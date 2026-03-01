#!/usr/bin/env python3
"""VLM Training Data Size Ablation Experiment.

Tests minimum number of VLM training samples needed for good performance.
This informs whether a closed-source VLM experiment (GPT-4V, Claude Vision) is cost-feasible.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/vlm_training_size_ablation.py

    # Quick mode (smaller sizes, 1 epoch, subset eval)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/vlm_training_size_ablation.py --quick
"""
import sys
import os
import json
import re
import io
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
try:
    import tensorflow as tf
except ImportError:
    tf = None  # ERQA benchmark excluded, tf not needed
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class UQTrainingSample:
    """Training sample for VLM judge."""
    question_id: str
    benchmark: str
    prompt: str
    response: str
    is_correct: bool
    dataset_index: int


# ============================================================
# DATA LOADING (copied from train_vlm_judge_combined.py)
# ============================================================

def parse_question_id(question_id: str) -> Optional[int]:
    """Extract dataset index from question_id."""
    match = re.search(r'_(\d+)$', question_id)
    if match:
        return int(match.group(1))
    return None


def load_erqa_from_tfrecord(tfrecord_path: str) -> list:
    """Load ERQA dataset from TFRecord file."""
    feature_description = {
        'answer': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.VarLenFeature(tf.string),
        'question_type': tf.io.VarLenFeature(tf.string),
        'visual_indices': tf.io.VarLenFeature(tf.int64),
        'question': tf.io.FixedLenFeature([], tf.string)
    }

    samples = []
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    for example_proto in dataset:
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        images_encoded = tf.sparse.to_dense(parsed['image/encoded']).numpy()

        if len(images_encoded) > 0:
            img_bytes = images_encoded[0]
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                img = None
        else:
            img = None

        samples.append({
            "image": img,
            "question": parsed['question'].numpy().decode('utf-8'),
            "answer": parsed['answer'].numpy().decode('utf-8'),
        })

    return samples


class ERQADataset:
    """Wrapper for ERQA TFRecord data."""
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_benchmark_dataset(benchmark: str):
    """Load benchmark dataset for image retrieval."""
    if benchmark == "vsr":
        return load_dataset("cambridgeltl/vsr_random", split="test")
    elif benchmark == "mmmu":
        return load_dataset("MMMU/MMMU", "Art", split="validation")
    elif benchmark == "charxiv":
        return load_dataset("princeton-nlp/CharXiv", split="validation")
    elif benchmark == "hallusionbench":
        return load_dataset("lmms-lab/HallusionBench", split="image")
    elif benchmark == "erqa":
        tfrecord_path = Path("data/erqa_repo/data/erqa.tfrecord")
        if tfrecord_path.exists():
            samples = load_erqa_from_tfrecord(str(tfrecord_path))
            return ERQADataset(samples)
        return None
    else:
        return None


def get_image_from_dataset(ds, idx: int, benchmark: str) -> Optional[Image.Image]:
    """Get image from dataset by index."""
    import requests
    from io import BytesIO

    try:
        if idx >= len(ds):
            return None
        row = ds[idx]

        if benchmark == "erqa":
            img = row.get("image")
        elif benchmark == "vsr":
            # Try image_link (COCO URL) first — row["image"] is often a filename string
            cache_dir = Path("data/vsr_images")
            cache_dir.mkdir(parents=True, exist_ok=True)
            url = row.get("image_link")
            if url:
                filename = url.split("/")[-1]
                cache_path = cache_dir / filename
                if cache_path.exists():
                    try:
                        return Image.open(cache_path).convert("RGB")
                    except Exception:
                        pass
                try:
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        img = Image.open(BytesIO(resp.content)).convert("RGB")
                        img.save(cache_path)
                        return img
                except Exception:
                    pass
            # Fallback to row["image"] if it's already a PIL Image
            img = row.get("image")
            if isinstance(img, Image.Image):
                return img.convert("RGB") if img.mode != "RGB" else img
            return None
        else:
            img = row.get("image")

        if img is None:
            return None
        if isinstance(img, Image.Image):
            return img.convert("RGB") if img.mode != "RGB" else img
        return None
    except Exception:
        return None


def load_training_samples(base_dir: Path) -> list[UQTrainingSample]:
    """Load all training samples from feature files."""
    samples = []

    for bench_dir in base_dir.iterdir():
        if not bench_dir.is_dir() or bench_dir.name == "smoke_test":
            continue

        benchmark = bench_dir.name
        pt_files = list(bench_dir.glob("*.pt"))

        for pt_file in pt_files:
            try:
                data = torch.load(pt_file, weights_only=False)
                question_id = data.get("question_id", "")
                idx = parse_question_id(question_id)

                if idx is None:
                    continue

                samples.append(UQTrainingSample(
                    question_id=question_id,
                    benchmark=benchmark,
                    prompt=data.get("prompt", ""),
                    response=data.get("response", ""),
                    is_correct=data.get("is_correct", False),
                    dataset_index=idx,
                ))
            except Exception:
                continue

    return samples


def load_test_ids(path: Path) -> set:
    """Load test IDs from probe evaluation."""
    with open(path) as f:
        test_ids = json.load(f)
    return {item["question_id"] for item in test_ids}


# ============================================================
# PROMPT FORMAT
# ============================================================

PROMPT_TEMPLATE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""


# ============================================================
# DATASET CLASS
# ============================================================

class VLMJudgeDataset(torch.utils.data.Dataset):
    """Dataset for VLM judge training with actual images."""

    def __init__(self, samples: list[UQTrainingSample], processor, max_length=2048):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self._datasets = {}
        self.fallback_image = Image.new('RGB', (336, 336), color='gray')
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 512 * 28 * 28

        # Preload datasets
        vision_benchmarks = set(s.benchmark for s in samples if s.dataset_index >= 0)
        for bench in vision_benchmarks:
            try:
                self._datasets[bench] = load_benchmark_dataset(bench)
            except Exception as e:
                print(f"    Failed to load {bench}: {e}")
                self._datasets[bench] = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if sample.dataset_index < 0:
            image = self.fallback_image
        else:
            ds = self._datasets.get(sample.benchmark)
            image = None
            if ds is not None:
                image = get_image_from_dataset(ds, sample.dataset_index, sample.benchmark)
            if image is None:
                image = self.fallback_image

        target = "ii" if sample.is_correct else "i"
        prompt = PROMPT_TEMPLATE.format(
            question=sample.prompt[:500],
            response=sample.response[:300]
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": target,
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        is_text_sample = sample.dataset_index < 0
        if is_text_sample:
            min_px = 256 * 28 * 28
            max_px = 256 * 28 * 28
        else:
            min_px = self.min_pixels
            max_px = self.max_pixels

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            min_pixels=min_px,
            max_pixels=max_px,
        )

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        assistant_token = 77091
        assistant_positions = (input_ids == assistant_token).nonzero(as_tuple=True)[0]
        if len(assistant_positions) > 0:
            answer_pos = assistant_positions[-1].item() + 2
            labels[:] = -100
            if answer_pos < len(labels):
                labels[answer_pos] = input_ids[answer_pos]
        else:
            labels[:-3] = -100

        result = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            if isinstance(pv, list):
                result["pixel_values"] = pv[0] if len(pv) > 0 else pv
            else:
                result["pixel_values"] = pv.squeeze(0) if pv.dim() > 3 else pv

        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]

        return result


# ============================================================
# EVALUATION
# ============================================================

def get_p_correct(model, processor, image, question, response, device):
    """Extract P(correct) from model logits."""
    prompt = PROMPT_TEMPLATE.format(question=question, response=response)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]

    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]

    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()


def evaluate_vlm_judge(
    model,
    processor,
    test_samples: list[UQTrainingSample],
    datasets_cache: dict,
    device: torch.device,
    fallback_image: Image.Image,
    max_samples: int = None,
) -> dict:
    """Evaluate VLM judge and compute metrics."""
    model.eval()

    if max_samples and len(test_samples) > max_samples:
        np.random.seed(42)
        indices = np.random.choice(len(test_samples), max_samples, replace=False)
        test_samples = [test_samples[i] for i in indices]

    all_preds = []
    all_labels = []
    per_benchmark = defaultdict(lambda: {"preds": [], "labels": []})

    for i, sample in enumerate(test_samples):
        if i % 50 == 0:
            print(f"  Evaluating {i}/{len(test_samples)}...")

        ds = datasets_cache.get(sample.benchmark)
        image = None
        if ds is not None:
            image = get_image_from_dataset(ds, sample.dataset_index, sample.benchmark)
        if image is None:
            image = fallback_image

        try:
            p_correct = get_p_correct(
                model, processor, image,
                sample.prompt[:500], sample.response[:300], device
            )
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            p_correct = 0.5

        all_preds.append(p_correct)
        all_labels.append(float(sample.is_correct))
        per_benchmark[sample.benchmark]["preds"].append(p_correct)
        per_benchmark[sample.benchmark]["labels"].append(float(sample.is_correct))

    results = {
        "auroc": roc_auc_score(all_labels, all_preds),
        "auprc": average_precision_score(all_labels, all_preds),
        "n_samples": len(all_labels),
        "n_correct": sum(all_labels),
        "base_rate": sum(all_labels) / len(all_labels),
    }

    results["per_benchmark"] = {}
    for bench, data in per_benchmark.items():
        if len(set(data["labels"])) < 2:
            continue
        results["per_benchmark"][bench] = {
            "auroc": roc_auc_score(data["labels"], data["preds"]),
            "n_samples": len(data["labels"]),
        }

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    results["brier"] = float(np.mean((preds - labels) ** 2))

    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for j in range(n_bins):
        in_bin = (preds >= bin_boundaries[j]) & (preds < bin_boundaries[j + 1])
        if in_bin.sum() == 0:
            continue
        bin_conf = preds[in_bin].mean()
        bin_acc = labels[in_bin].mean()
        ece += (in_bin.sum() / len(preds)) * abs(bin_acc - bin_conf)
    results["ece"] = ece

    return results


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_vlm_judge(
    train_samples: List[UQTrainingSample],
    model_name: str,
    output_dir: Path,
    num_epochs: int = 3,
    quick_mode: bool = False,
) -> float:
    """Train VLM judge and return training time in seconds."""
    start_time = time.time()

    print(f"\nLoading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Leave headroom on each GPU for fp32 conversions during backward pass
    max_memory = {
        0: "75GiB",
        1: "75GiB",
        2: "75GiB",
        3: "78GiB",
    }

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)

    print("Creating dataset...")
    train_dataset = VLMJudgeDataset(train_samples, processor)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="no",  # Don't save checkpoints during ablation
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
    )

    def collate_fn(batch):
        max_len = max(x["input_ids"].size(0) for x in batch)
        pad_token_id = processor.tokenizer.pad_token_id or 0

        input_ids = []
        attention_mask = []
        labels = []

        for x in batch:
            seq_len = x["input_ids"].size(0)
            pad_len = max_len - seq_len

            input_ids.append(torch.cat([
                torch.full((pad_len,), pad_token_id, dtype=x["input_ids"].dtype),
                x["input_ids"]
            ]))
            attention_mask.append(torch.cat([
                torch.zeros(pad_len, dtype=x["attention_mask"].dtype),
                x["attention_mask"]
            ]))
            labels.append(torch.cat([
                torch.full((pad_len,), -100, dtype=x["labels"].dtype),
                x["labels"]
            ]))

        result = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

        if "pixel_values" in batch[0]:
            result["pixel_values"] = torch.cat([x["pixel_values"] for x in batch], dim=0)

        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in batch])

        return result

    class MemoryCleanupCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 100 == 0:
                torch.cuda.empty_cache()
            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[MemoryCleanupCallback()],
    )

    print(f"Training with {len(train_samples)} samples, {num_epochs} epochs...")
    trainer.train()

    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    training_time = time.time() - start_time

    # Return model and processor for evaluation (avoid reload issues)
    return training_time, model, processor


def evaluate_model_directly(
    model,
    processor,
    test_samples: List[UQTrainingSample],
    max_eval_samples: int = None,
) -> dict:
    """Evaluate model directly without reloading."""
    model.eval()

    # Create temp dataset just to get the dataset cache
    temp_dataset = VLMJudgeDataset(test_samples[:1], processor)
    datasets_cache = temp_dataset._datasets
    fallback_image = temp_dataset.fallback_image
    device = next(model.parameters()).device

    results = evaluate_vlm_judge(
        model, processor, test_samples, datasets_cache, device, fallback_image,
        max_samples=max_eval_samples
    )

    return results


def cleanup_model(model, processor):
    """Clean up model after evaluation."""
    del model
    torch.cuda.empty_cache()


def run_single_size(size: int, output_base: Path, vision_train: list, vision_test: list,
                    model_name: str, num_epochs: int, max_eval_samples: int) -> dict:
    """Run training and evaluation for a single size. Called as subprocess to avoid CUDA issues."""
    import subprocess
    import sys

    # Run as subprocess to get clean CUDA context
    script = f'''
import sys
sys.path.insert(0, "/scratch/khayes/LLM")
import json
import numpy as np
import torch
from pathlib import Path

# Import the ablation module functions
exec(open("/scratch/khayes/LLM/scripts/vlm_training_size_ablation.py").read().split("if __name__")[0])

# Load data
base_dir = Path("data/features")
all_samples = load_training_samples(base_dir)
test_ids_path = Path("data/probe_results/test_ids.json")
test_ids = load_test_ids(test_ids_path) if test_ids_path.exists() else set()
vision_train = [s for s in all_samples if s.question_id not in test_ids]
vision_test = [s for s in all_samples if s.question_id in test_ids]

# Subsample
size = {size}
np.random.seed(42)
correct_samples = [s for s in vision_train if s.is_correct]
incorrect_samples = [s for s in vision_train if not s.is_correct]
n_each = size // 2
n_correct_sample = min(n_each, len(correct_samples))
n_incorrect_sample = min(n_each, len(incorrect_samples))
correct_idx = np.random.choice(len(correct_samples), n_correct_sample, replace=False)
incorrect_idx = np.random.choice(len(incorrect_samples), n_incorrect_sample, replace=False)
train_subset = [correct_samples[i] for i in correct_idx] + [incorrect_samples[i] for i in incorrect_idx]
np.random.shuffle(train_subset)

# Train
output_dir = Path("{output_base}/size_{size}")
output_dir.mkdir(exist_ok=True, parents=True)
training_time, model, processor = train_vlm_judge(
    train_subset, "{model_name}", output_dir, num_epochs={num_epochs}
)

# Evaluate
eval_results = evaluate_model_directly(model, processor, vision_test, max_eval_samples={max_eval_samples if max_eval_samples else 'None'})

# Output results as JSON
result = {{
    "size": size,
    "n_train_actual": len(train_subset),
    "n_correct": n_correct_sample,
    "n_incorrect": n_incorrect_sample,
    "training_time_sec": training_time,
    "training_time_min": training_time / 60,
    "vision_auroc": eval_results["auroc"],
    "vision_auprc": eval_results["auprc"],
    "vision_ece": eval_results["ece"],
    "vision_brier": eval_results["brier"],
}}
print("RESULT_JSON:" + json.dumps(result))
'''

    # Write temp script
    temp_script = output_base / f"_run_size_{size}.py"
    with open(temp_script, 'w') as f:
        f.write(script)

    # Run as subprocess
    env = dict(os.environ)
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    result = subprocess.run(
        [sys.executable, str(temp_script)],
        capture_output=True,
        text=True,
        env=env
    )

    # Parse output
    for line in result.stdout.split('\n'):
        if line.startswith('RESULT_JSON:'):
            return json.loads(line[len('RESULT_JSON:'):])

    # Error case
    print(f"ERROR for size {size}:")
    print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
    return None


# ============================================================
# CROSS-MODEL TRANSFER EVALUATION
# ============================================================

def run_cross_model_transfer(checkpoint_dir: Path, model_name: str, max_samples: int = 100) -> dict:
    """Run cross-model transfer test on a few benchmarks."""
    from datasets import load_dataset
    from tqdm import tqdm
    from PIL import Image

    print(f"\nRunning cross-model transfer test...")

    # Load models
    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)

    # Leave headroom on each GPU for fp32 conversions
    max_memory = {
        0: "75GiB",
        1: "75GiB",
        2: "75GiB",
        3: "78GiB",
    }

    # Load Qwen2.5-VL-72B for target model
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor as Qwen2Processor

    print("Loading Qwen2.5-VL-72B...")
    qwen_processor = Qwen2Processor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-72B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
    )

    # Load VLM judge
    from peft import PeftModel

    print("Loading VLM judge...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )
    judge_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    judge_model.eval()

    judge_processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)

    # Load VSR benchmark
    print("Loading VSR samples...")
    ds = load_dataset("cambridgeltl/vsr_random", split="test")

    samples = []
    import requests
    from io import BytesIO

    for idx in range(min(max_samples, len(ds))):
        row = ds[idx]
        # Download image from COCO
        url = row.get("image_link")
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                image = Image.open(BytesIO(resp.content)).convert("RGB")
            else:
                continue
        except:
            continue

        samples.append({
            "image": image,
            "caption": row["caption"],
            "label": row["label"] == 1,
        })

    print(f"Loaded {len(samples)} samples")

    # Generate Qwen responses and judge predictions
    preds = []
    labels = []
    fallback_image = Image.new('RGB', (336, 336), color='gray')

    for sample in tqdm(samples, desc="Cross-model transfer"):
        # Get Qwen response
        prompt = f'Is this statement about the image true or false? "{sample["caption"]}"\nAnswer with just "True" or "False".'

        messages = [{"role": "user", "content": [{"type": "image", "image": sample["image"]}, {"type": "text", "text": prompt}]}]
        text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_processor(text=[text], images=[sample["image"]], return_tensors="pt", padding=True)
        inputs = {k: v.to(qwen_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = qwen_model.generate(**inputs, max_new_tokens=20, do_sample=False)
        response = qwen_processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Check correctness
        response_lower = response.lower()
        resp_true = "true" in response_lower and "false" not in response_lower
        resp_false = "false" in response_lower and "true" not in response_lower
        gt_true = sample["label"]

        if resp_true:
            is_correct = gt_true
        elif resp_false:
            is_correct = not gt_true
        else:
            is_correct = False

        # Get judge prediction
        try:
            p_correct = get_p_correct(
                judge_model, judge_processor,
                sample["image"], prompt, response,
                next(judge_model.parameters()).device
            )
        except:
            p_correct = 0.5

        preds.append(p_correct)
        labels.append(float(is_correct))

    # Compute metrics
    results = {
        "auroc": roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5,
        "n_samples": len(labels),
        "target_accuracy": sum(labels) / len(labels),
    }

    # Clean up
    del qwen_model
    del judge_model
    del base_model
    torch.cuda.empty_cache()

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick mode: smaller sizes, 1 epoch")
    parser.add_argument("--skip-transfer", action="store_true", help="Skip cross-model transfer test")
    parser.add_argument("--size", type=int, help="Run only this size (for subprocess mode)")
    parser.add_argument("--run-all-subprocess", action="store_true", help="Run all sizes as separate subprocesses")
    args = parser.parse_args()

    # Config
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    output_base = Path("data/ablations/vlm_training_size_vsr_fixed")
    output_base.mkdir(parents=True, exist_ok=True)

    if args.quick:
        training_sizes = [100, 250, 500]
        num_epochs = 1
        max_eval_samples = 100
        max_transfer_samples = 50
    else:
        training_sizes = [100, 250, 500, 1000, 2000, 4200]
        num_epochs = 3
        max_eval_samples = None  # Use all
        max_transfer_samples = 100

    # If --run-all-subprocess, spawn separate processes for each size
    if args.run_all_subprocess:
        import subprocess
        import os

        print("=" * 70)
        print("VLM TRAINING SIZE ABLATION - SUBPROCESS MODE")
        print("=" * 70)
        print(f"\nRunning sizes {training_sizes} as separate subprocesses...")

        results = []
        for size in training_sizes:
            print(f"\n{'='*70}")
            print(f"SPAWNING SUBPROCESS FOR SIZE {size}")
            print(f"{'='*70}")

            cmd = [sys.executable, __file__, "--size", str(size)]
            if args.quick:
                cmd.append("--quick")
            if args.skip_transfer:
                cmd.append("--skip-transfer")

            env = dict(os.environ)
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

            proc = subprocess.run(cmd, env=env, capture_output=False)

            # Load result from file
            result_file = output_base / f"size_{size}" / "result.json"
            if result_file.exists():
                with open(result_file) as f:
                    result = json.load(f)
                results.append(result)
                print(f"Size {size}: AUROC = {result['vision_auroc']:.4f}")
            else:
                print(f"Size {size}: FAILED (no result file)")

        # Print summary and save
        print("\n" + "=" * 70)
        print("VLM TRAINING SIZE ABLATION RESULTS")
        print("=" * 70)

        print(f"\n{'Samples':<10} {'Vision AUROC':<15} {'Training Time':<15}")
        print("-" * 40)
        for r in results:
            print(f"{r['size']:<10} {r['vision_auroc']:.4f}          {r['training_time_min']:.1f} min")
        print("-" * 40)
        print(f"{'4200 (full)':<10} {'0.789 (baseline)':<15}")

        summary = {
            "experiment": "vlm_training_size_ablation",
            "model": model_name,
            "quick_mode": args.quick,
            "num_epochs": num_epochs,
            "results": results,
        }
        with open(output_base / "ablation_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {output_base / 'ablation_results.json'}")
        return

    # If --size is specified, run only that size
    if args.size:
        training_sizes = [args.size]

    print("=" * 70)
    print("VLM TRAINING DATA SIZE ABLATION EXPERIMENT")
    print("=" * 70)

    if args.quick:
        print("\nQUICK MODE: smaller sizes, 1 epoch, 100 eval samples")
    else:
        print(f"\nFULL MODE: {training_sizes} samples, 3 epochs")

    # Load all vision training samples
    print("\nLoading vision training data...")
    base_dir = Path("data/features")
    all_vision_samples = load_training_samples(base_dir)
    print(f"Total vision samples: {len(all_vision_samples)}")

    # Load test IDs
    test_ids_path = Path("data/probe_results/test_ids.json")
    if test_ids_path.exists():
        test_ids = load_test_ids(test_ids_path)
    else:
        test_ids = set()

    # Split into train/test
    vision_train = [s for s in all_vision_samples if s.question_id not in test_ids]
    vision_test = [s for s in all_vision_samples if s.question_id in test_ids]

    print(f"Vision train: {len(vision_train)}")
    print(f"Vision test: {len(vision_test)}")

    # Ensure class balance info
    n_correct = sum(1 for s in vision_train if s.is_correct)
    print(f"Train correct: {n_correct} ({100*n_correct/len(vision_train):.1f}%)")

    results = []

    for size in training_sizes:
        print("\n" + "=" * 70)
        print(f"TRAINING SIZE: {size}")
        print("=" * 70)

        # Subsample with class balance
        np.random.seed(42)
        correct_samples = [s for s in vision_train if s.is_correct]
        incorrect_samples = [s for s in vision_train if not s.is_correct]

        # Sample equal amounts from each class
        n_each = size // 2
        n_correct_sample = min(n_each, len(correct_samples))
        n_incorrect_sample = min(n_each, len(incorrect_samples))

        correct_idx = np.random.choice(len(correct_samples), n_correct_sample, replace=False)
        incorrect_idx = np.random.choice(len(incorrect_samples), n_incorrect_sample, replace=False)

        train_subset = [correct_samples[i] for i in correct_idx] + [incorrect_samples[i] for i in incorrect_idx]
        np.random.shuffle(train_subset)

        print(f"Subsampled: {len(train_subset)} ({n_correct_sample} correct, {n_incorrect_sample} incorrect)")

        # Output directory for this run
        run_dir = output_base / f"size_{size}"
        run_dir.mkdir(exist_ok=True)

        # Train
        training_time, model, processor = train_vlm_judge(
            train_subset,
            model_name,
            run_dir,
            num_epochs=num_epochs,
            quick_mode=args.quick,
        )
        print(f"Training time: {training_time/60:.1f} minutes")

        # Evaluate on vision test set (using model directly, no reload)
        print("\nEvaluating on vision test set...")
        eval_results = evaluate_model_directly(
            model, processor, vision_test,
            max_eval_samples=max_eval_samples
        )

        print(f"Vision AUROC: {eval_results['auroc']:.4f}")
        print(f"Vision AUPRC: {eval_results['auprc']:.4f}")

        # Clean up model before next training run
        cleanup_model(model, processor)

        # Cross-model transfer (optional)
        if not args.skip_transfer:
            transfer_results = run_cross_model_transfer(
                run_dir, model_name, max_samples=max_transfer_samples
            )
            cross_model_auroc = transfer_results["auroc"]
            print(f"Cross-model AUROC: {cross_model_auroc:.4f}")
        else:
            cross_model_auroc = None

        result = {
            "size": size,
            "n_train_actual": len(train_subset),
            "n_correct": n_correct_sample,
            "n_incorrect": n_incorrect_sample,
            "training_time_sec": training_time,
            "training_time_min": training_time / 60,
            "vision_auroc": eval_results["auroc"],
            "vision_auprc": eval_results["auprc"],
            "vision_ece": eval_results["ece"],
            "vision_brier": eval_results["brier"],
            "cross_model_auroc": cross_model_auroc,
        }
        results.append(result)

        # Save result for this size (for subprocess mode)
        with open(run_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)

        # Save intermediate results
        with open(output_base / "ablation_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print("VLM TRAINING SIZE ABLATION RESULTS")
    print("=" * 70)

    print(f"\n{'Samples':<10} {'Vision AUROC':<15} {'Cross-Model AUROC':<20} {'Training Time':<15}")
    print("-" * 60)
    for r in results:
        cm_auroc = f"{r['cross_model_auroc']:.3f}" if r['cross_model_auroc'] else "N/A"
        print(f"{r['size']:<10} {r['vision_auroc']:.4f}          {cm_auroc:<20} {r['training_time_min']:.1f} min")

    # Add baseline comparison
    print("-" * 60)
    print(f"{'4200 (full)':<10} {'0.789 (baseline)':<15} {'0.693 (baseline)':<20}")

    # Save final results
    summary = {
        "experiment": "vlm_training_size_ablation",
        "model": model_name,
        "quick_mode": args.quick,
        "num_epochs": num_epochs,
        "results": results,
        "baseline": {
            "size": 4200,
            "vision_auroc": 0.789,
            "cross_model_auroc": 0.693,
        }
    }

    with open(output_base / "ablation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_base / 'ablation_results.json'}")

    # Create plot
    sizes = [r["size"] for r in results]
    vision_aurocs = [r["vision_auroc"] for r in results]
    cross_model_aurocs = [r["cross_model_auroc"] for r in results if r["cross_model_auroc"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, vision_aurocs, 'o-', label='Vision Test AUROC', linewidth=2, markersize=8)
    if cross_model_aurocs and len(cross_model_aurocs) == len(sizes):
        ax.plot(sizes, cross_model_aurocs, 's--', label='Cross-Model AUROC', linewidth=2, markersize=8)

    ax.axhline(y=0.789, color='blue', linestyle=':', alpha=0.7, label='Baseline Vision (4200)')
    ax.axhline(y=0.693, color='orange', linestyle=':', alpha=0.7, label='Baseline Cross-Model (4200)')
    ax.axhline(y=0.70, color='gray', linestyle='--', alpha=0.5, label='Target (0.70)')

    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('VLM Judge Training Size Ablation', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])

    plt.tight_layout()
    plt.savefig(output_base / "vlm_training_size_ablation.png", dpi=150)
    print(f"Plot saved to {output_base / 'vlm_training_size_ablation.png'}")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Find minimum size for AUROC > 0.70
    min_viable = None
    for r in results:
        if r["vision_auroc"] >= 0.70:
            min_viable = r["size"]
            break

    if min_viable:
        print(f"Minimum viable size (AUROC >= 0.70): {min_viable} samples")
    else:
        print("No size achieved AUROC >= 0.70")

    # Find plateau point (90% of max performance)
    max_auroc = max(r["vision_auroc"] for r in results)
    plateau_threshold = 0.9 * max_auroc
    for r in results:
        if r["vision_auroc"] >= plateau_threshold:
            print(f"Plateau point (90% of max): {r['size']} samples (AUROC {r['vision_auroc']:.3f})")
            break


if __name__ == "__main__":
    main()
