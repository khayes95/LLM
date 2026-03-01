#!/usr/bin/env python3
"""
Retrain VLM Judge with FIXED VSR image loading.

This is a copy of train_vlm_judge_combined.py with the VSR fix already applied.
The fix downloads images from COCO URLs via image_link field.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_vlm_judge_vsr_fixed.py
"""
import sys
import json
import re
import io
import requests
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

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# DATA LOADING (with VSR fix)
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

        # Get first image
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
    """Wrapper for ERQA TFRecord data to match HuggingFace dataset interface."""
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


VSR_IMAGE_CACHE_DIR = Path("data/vsr_images")

def get_image_from_dataset(ds, idx: int, benchmark: str) -> Optional[Image.Image]:
    """Get image from dataset by index. FIXED for VSR with disk caching."""
    from io import BytesIO

    try:
        if idx >= len(ds):
            return None
        row = ds[idx]

        if benchmark == "erqa":
            img = row.get("image")
        elif benchmark == "vsr":
            # VSR FIX: Download from COCO URL with disk caching
            img = row.get("image")
            if isinstance(img, Image.Image):
                return img.convert("RGB") if img.mode != "RGB" else img
            # Check disk cache first
            url = row.get("image_link")
            if url:
                filename = url.split("/")[-1]
                cache_path = VSR_IMAGE_CACHE_DIR / filename
                if cache_path.exists():
                    return Image.open(cache_path).convert("RGB")
                # Download and cache
                try:
                    resp = requests.get(url, timeout=30)
                    if resp.status_code == 200:
                        img = Image.open(BytesIO(resp.content)).convert("RGB")
                        VSR_IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        img.save(cache_path)
                        return img
                except Exception:
                    pass
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
        print(f"  {benchmark}: {len(pt_files)} files")

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


def load_text_samples(train_path: Path) -> list[UQTrainingSample]:
    """Load text UQ training samples (no images)."""
    samples = []
    with open(train_path) as f:
        for line in f:
            data = json.loads(line)
            prompt = data["input"] if isinstance(data["input"], str) else json.dumps(data["input"])
            response = data["model_response"]

            samples.append(UQTrainingSample(
                question_id=data["id"],
                benchmark="text_" + data.get("benchmark", "unknown"),
                prompt=prompt,
                response=response,
                is_correct=data["correct"] == 1,
                dataset_index=-1,
            ))
    return samples


# ============================================================
# PROMPT FORMAT
# ============================================================

PROMPT_TEMPLATE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""


# ============================================================
# DATASET
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
                print(f"    Loaded {bench} dataset")
            except Exception as e:
                print(f"    Failed to load {bench}: {e}")
                self._datasets[bench] = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get image
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
        im_end_token = 151645

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
) -> dict:
    """Evaluate VLM judge and compute metrics."""
    model.eval()

    all_preds = []
    all_labels = []
    per_benchmark = defaultdict(lambda: {"preds": [], "labels": []})

    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
    print(f"Token IDs: i={token_i}, ii={token_ii}")

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
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("VLM JUDGE RETRAINING - VSR FIX (Qwen3-VL-8B LoRA)")
    print("=" * 70)
    print("\nThis run uses FIXED VSR image loading (downloads from COCO URLs)")
    print("Previous run had VSR using gray fallback images\n")

    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    output_dir = Path("data/vlm_judge_vsr_fixed")
    output_dir.mkdir(exist_ok=True)

    # Load VISION samples
    print("Loading vision training data...")
    base_dir = Path("data/features")
    vision_samples = load_training_samples(base_dir)
    print(f"Vision samples: {len(vision_samples)}")

    # Load TEXT samples
    print("\nLoading text training data...")
    text_train_path = Path("data/finetune/train_v2.jsonl")
    text_samples = load_text_samples(text_train_path)
    print(f"Text samples: {len(text_samples)}")

    # Load vision test IDs
    test_ids_path = Path("data/probe_results/test_ids.json")
    if test_ids_path.exists():
        test_ids = load_test_ids(test_ids_path)
        print(f"Loaded {len(test_ids)} vision test IDs")
    else:
        print("WARNING: No test_ids.json found")
        test_ids = set()

    # Split vision samples
    vision_train = [s for s in vision_samples if s.question_id not in test_ids]
    vision_test = [s for s in vision_samples if s.question_id in test_ids]

    # Load text test samples
    text_test_path = Path("data/finetune/test_v2.jsonl")
    text_test = load_text_samples(text_test_path)
    print(f"Text test samples: {len(text_test)}")

    # Combine training
    train_samples = vision_train + text_samples
    test_samples = vision_test

    print(f"\nCombined training: {len(train_samples)} ({len(vision_train)} vision + {len(text_samples)} text)")
    print(f"Vision test: {len(vision_test)}")
    print(f"Text test: {len(text_test)}")

    # Count VSR samples
    vsr_train = sum(1 for s in train_samples if s.benchmark == "vsr")
    print(f"\nVSR training samples (now with real images): {vsr_train}")

    n_correct_train = sum(1 for s in train_samples if s.is_correct)
    print(f"Train correct: {n_correct_train} ({100*n_correct_train/len(train_samples):.1f}%)")

    # Load model
    print(f"\nLoading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    num_gpus = torch.cuda.device_count()
    max_memory = {i: "70GiB" for i in range(num_gpus)}
    print(f"  Using {num_gpus} GPUs with {max_memory}")

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

    # Enable gradient checkpointing to reduce activation memory
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    # Create datasets
    print("\nCreating datasets (VSR will download images from COCO)...")
    train_dataset = VLMJudgeDataset(train_samples, processor)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        bf16_full_eval=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        optim="adamw_bnb_8bit",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
            if state.global_step % 10 == 0:
                torch.cuda.empty_cache()
            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[MemoryCleanupCallback()],
    )

    print("\n" + "=" * 50)
    print("STARTING TRAINING (with fixed VSR images)")
    print("=" * 50)
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

    trainer.train()

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

    print("\n--- Vision Test Set ---")
    vision_results = evaluate_vlm_judge(
        model, processor, vision_test, datasets_cache, device, fallback_image
    )

    print("\n--- Text Test Set ---")
    text_results = evaluate_vlm_judge(
        model, processor, text_test, datasets_cache, device, fallback_image
    )

    print("\n" + "=" * 70)
    print("VLM JUDGE RESULTS (VSR FIXED)")
    print("=" * 70)

    print("\nVISION TEST:")
    print(f"  AUROC: {vision_results['auroc']:.4f}")
    print(f"  AUPRC: {vision_results['auprc']:.4f}")
    print(f"  ECE: {vision_results['ece']:.4f}")

    if "vsr" in vision_results.get("per_benchmark", {}):
        print(f"  VSR AUROC: {vision_results['per_benchmark']['vsr']['auroc']:.4f}")

    print("\nTEXT TEST:")
    print(f"  AUROC: {text_results['auroc']:.4f}")
    print(f"  AUPRC: {text_results['auprc']:.4f}")
    print(f"  ECE: {text_results['ece']:.4f}")

    print("\n" + "-" * 50)
    print("COMPARISON TO PREVIOUS (broken VSR)")
    print("-" * 50)
    print(f"Previous vision AUROC: 0.789")
    print(f"Previous VSR AUROC:    0.637 (gray images)")
    print(f"Fixed vision AUROC:    {vision_results['auroc']:.4f}")
    if "vsr" in vision_results.get("per_benchmark", {}):
        print(f"Fixed VSR AUROC:       {vision_results['per_benchmark']['vsr']['auroc']:.4f}")

    # Save results
    combined_results = {
        "vision": vision_results,
        "text": text_results,
        "fix_applied": "VSR images now downloaded from COCO URLs",
    }
    results_path = output_dir / "vsr_fixed_results.json"
    with open(results_path, "w") as f:
        json.dump(combined_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
