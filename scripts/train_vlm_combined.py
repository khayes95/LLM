#!/usr/bin/env python3
"""
Train VLM Judge on COMBINED text + vision data.
"""
import sys
import json
import re
import io
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
import tensorflow as tf
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# CONFIG
# ============================================================

PROMPT_TEMPLATE = """Question: {question}
Answer: {response}
Is the answer correct? (i) No (ii) Yes"""


@dataclass
class UQTrainingSample:
    question_id: str
    benchmark: str
    prompt: str
    response: str
    is_correct: bool
    dataset_index: Optional[int] = None
    has_image: bool = False  # Whether this sample has an image


# ============================================================
# DATA LOADING
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
    try:
        if idx >= len(ds):
            return None
        row = ds[idx]

        if benchmark == "erqa":
            img = row.get("image")
        else:
            img = row.get("image")

        if img is None:
            return None
        if isinstance(img, Image.Image):
            return img.convert("RGB") if img.mode != "RGB" else img
        return None
    except Exception:
        return None


def load_vision_samples(base_dir: Path) -> list[UQTrainingSample]:
    """Load vision UQ training samples from .pt feature files."""
    samples = []
    for bench_dir in base_dir.iterdir():
        if not bench_dir.is_dir() or bench_dir.name == "smoke_test":
            continue
        benchmark = bench_dir.name
        print(f"  {benchmark}: ", end="")
        count = 0
        # Feature files are .pt (PyTorch tensors), not JSON
        for pt_file in bench_dir.glob("*.pt"):
            try:
                data = torch.load(pt_file, weights_only=False)

                question_id = data.get("question_id", "")
                idx = parse_question_id(question_id)

                if idx is None:
                    continue

                sample = UQTrainingSample(
                    question_id=question_id,
                    benchmark=benchmark,
                    prompt=data.get("prompt", ""),
                    response=data.get("response", ""),
                    is_correct=data.get("is_correct", False),
                    dataset_index=idx,
                    has_image=True,
                )
                samples.append(sample)
                count += 1
            except Exception as e:
                continue
        print(f"{count} files")
    return samples


def load_text_samples(train_path: Path) -> list[UQTrainingSample]:
    """Load text UQ training samples."""
    samples = []
    with open(train_path) as f:
        for line in f:
            data = json.loads(line)
            sample = UQTrainingSample(
                question_id=data["id"],
                benchmark=data.get("benchmark", "text"),
                prompt=data["input"] if isinstance(data["input"], str) else json.dumps(data["input"]),
                response=data["model_response"],
                is_correct=data["correct"] == 1,
                dataset_index=None,
                has_image=False,
            )
            samples.append(sample)
    return samples


# ============================================================
# DATASET
# ============================================================

class CombinedDataset(torch.utils.data.Dataset):
    """Combined text + vision dataset."""

    def __init__(
        self,
        samples: list[UQTrainingSample],
        processor,
        vision_benchmarks: list[str] = None,
        min_pixels: int = 256 * 28 * 28,  # Qwen3-VL uses 28x28 patches
        max_pixels: int = 256 * 28 * 28,  # Reduced from 512 to save memory
    ):
        self.samples = samples
        self.processor = processor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Load vision datasets
        self._datasets = {}
        if vision_benchmarks:
            for bench in vision_benchmarks:
                print(f"    Loaded {bench} dataset")
                self._datasets[bench] = load_benchmark_dataset(bench)

        # Create fallback image
        self.fallback_image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = "ii" if sample.is_correct else "i"

        # Get image
        if sample.has_image and sample.benchmark in self._datasets:
            ds = self._datasets[sample.benchmark]
            image = get_image_from_dataset(ds, sample.dataset_index, sample.benchmark)
            if image is None:
                image = self.fallback_image
        else:
            image = self.fallback_image

        # Build prompt
        # Truncate to match vision sample sizes (vision prompts are ~140 chars, responses ~4 chars)
        # Text samples are much longer and cause OOM when hit during training
        max_prompt = 200 if not sample.has_image else 500
        max_response = 100 if not sample.has_image else 300
        prompt = PROMPT_TEMPLATE.format(
            question=sample.prompt[:max_prompt],
            response=sample.response[:max_response]
        )

        # Build conversation
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

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        # Labels - mask everything except the assistant's answer token
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
        max_pixels=256 * 28 * 28,  # Reduced from 512 to save memory
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]

    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]

    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()


def evaluate_model(
    model,
    processor,
    test_samples: list[UQTrainingSample],
    datasets_cache: dict,
    device: torch.device,
    fallback_image: Image.Image,
    desc: str = "Evaluating",
) -> dict:
    """Evaluate model and compute metrics."""
    model.eval()

    all_preds = []
    all_labels = []
    per_benchmark = defaultdict(lambda: {"preds": [], "labels": []})

    for i, sample in enumerate(tqdm(test_samples, desc=desc)):
        # Get image
        if sample.has_image and sample.benchmark in datasets_cache:
            ds = datasets_cache.get(sample.benchmark)
            image = get_image_from_dataset(ds, sample.dataset_index, sample.benchmark)
            if image is None:
                image = fallback_image
        else:
            image = fallback_image

        try:
            p_correct = get_p_correct(
                model, processor, image,
                sample.prompt[:500], sample.response[:300], device
            )
        except Exception as e:
            p_correct = 0.5

        all_preds.append(p_correct)
        all_labels.append(float(sample.is_correct))
        per_benchmark[sample.benchmark]["preds"].append(p_correct)
        per_benchmark[sample.benchmark]["labels"].append(float(sample.is_correct))

    # Compute metrics
    results = {
        "auroc": roc_auc_score(all_labels, all_preds),
        "auprc": average_precision_score(all_labels, all_preds),
        "n_samples": len(all_labels),
        "n_correct": sum(all_labels),
        "base_rate": sum(all_labels) / len(all_labels),
    }

    # Per-benchmark
    results["per_benchmark"] = {}
    for bench, data in per_benchmark.items():
        if len(set(data["labels"])) < 2:
            continue
        results["per_benchmark"][bench] = {
            "auroc": roc_auc_score(data["labels"], data["preds"]),
            "n_samples": len(data["labels"]),
        }

    # ECE and Brier
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
        ece += in_bin.sum() / len(preds) * abs(bin_conf - bin_acc)
    results["ece"] = float(ece)

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("VLM JUDGE TRAINING (COMBINED TEXT + VISION)")
    print("=" * 70)

    # Config
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    output_dir = Path("data/vlm_judge_combined")
    output_dir.mkdir(exist_ok=True)

    # Load vision samples
    print("\nLoading vision training data...")
    vision_dir = Path("data/features")
    vision_samples = load_vision_samples(vision_dir)
    print(f"Vision samples: {len(vision_samples)}")

    # Load text samples
    print("\nLoading text training data...")
    text_train_path = Path("data/finetune/train_v2.jsonl")
    text_samples = load_text_samples(text_train_path)
    print(f"Text samples: {len(text_samples)}")

    # Load vision test IDs (list of dicts with question_id field)
    test_ids_path = Path("data/probe_results/test_ids.json")
    with open(test_ids_path) as f:
        test_ids_data = json.load(f)
        vision_test_ids = set(item["question_id"] for item in test_ids_data)

    # Split vision samples
    vision_train = [s for s in vision_samples if s.question_id not in vision_test_ids]
    vision_test = [s for s in vision_samples if s.question_id in vision_test_ids]

    # Load text test samples
    text_test_path = Path("data/finetune/test_v2.jsonl")
    text_test = load_text_samples(text_test_path)

    # Combine training data
    all_train = vision_train + text_samples
    print(f"\nCombined training: {len(all_train)} ({len(vision_train)} vision + {len(text_samples)} text)")
    print(f"Vision test: {len(vision_test)}")
    print(f"Text test: {len(text_test)}")

    # Class balance
    n_correct_train = sum(1 for s in all_train if s.is_correct)
    print(f"\nTrain class balance: {n_correct_train} correct ({100*n_correct_train/len(all_train):.1f}%)")

    # Load model
    print(f"\nLoading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Use all 4 GPUs with reduced memory allocation to leave room for fp32 conversion
    # The original script used 75/78 GiB but other GPU processes are using ~3GB on GPU 0
    max_memory = {
        0: "72GiB",
        1: "72GiB",
        2: "72GiB",
        3: "75GiB",
    }

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )

    # Note: Gradient checkpointing causes OOM during backward pass recomputation
    # with this model. The original training worked without it.

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)

    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    # Create dataset
    print("\nCreating dataset...")
    vision_benchmarks = list(set(s.benchmark for s in vision_train if s.has_image))
    train_dataset = CombinedDataset(all_train, processor, vision_benchmarks)

    # Training arguments (matches original train_vlm_judge.py)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,  # Same as original
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,  # Same as original
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        bf16_full_eval=True,  # Keep bf16 precision during forward to avoid fp32 conversion OOM
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,  # Reduce memory pressure
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
    )

    # Collator
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

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[MemoryCleanupCallback()],
    )

    # Train
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Total samples: {len(all_train)}")
    print(f"Steps per epoch: {len(all_train) // training_args.gradient_accumulation_steps}")

    trainer.train()

    # Save
    print(f"\nSaving LoRA adapter to {output_dir}...")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    # Evaluate
    print("\n" + "=" * 50)
    print("EVALUATING ON TEST SETS")
    print("=" * 50)

    datasets_cache = train_dataset._datasets
    fallback_image = train_dataset.fallback_image
    device = next(model.parameters()).device

    # Evaluate on vision test
    print("\n--- Vision Test Set ---")
    vision_results = evaluate_model(
        model, processor, vision_test, datasets_cache, device, fallback_image, "Vision"
    )
    print(f"Vision AUROC: {vision_results['auroc']:.4f}")

    # Evaluate on text test
    print("\n--- Text Test Set ---")
    text_results = evaluate_model(
        model, processor, text_test, {}, device, fallback_image, "Text"
    )
    print(f"Text AUROC: {text_results['auroc']:.4f}")

    # Combined results
    results = {
        "vision": vision_results,
        "text": text_results,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("COMBINED VLM JUDGE RESULTS")
    print("=" * 70)
    print(f"\nVision Test AUROC: {vision_results['auroc']:.4f}")
    print(f"Text Test AUROC: {text_results['auroc']:.4f}")

    print("\n--- Comparison ---")
    print(f"Vision-only trained (Option B on vision): 0.766")
    print(f"Vision-only trained (Option B on text):   0.811")
    print(f"Combined trained (vision): {vision_results['auroc']:.4f}")
    print(f"Combined trained (text):   {text_results['auroc']:.4f}")

    # Save results
    results_path = output_dir / "combined_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
