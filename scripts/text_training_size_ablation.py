#!/usr/bin/env python3
"""Text Training Data Size Ablation Experiment.

Tests minimum number of TEXT training samples needed for good performance.
Runs two experiments:
1. VLM Judge on Text: Qwen3-VL-8B with gray placeholder images
2. Pure LLM Judge: Qwen2.5-7B-Instruct (text-only, no vision)

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/text_training_size_ablation.py --vlm
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/text_training_size_ablation.py --llm
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/text_training_size_ablation.py --both
"""
import sys
import os
import json
import time
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    Qwen3VLForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TextTrainingSample:
    """Training sample for text judge."""
    question_id: str
    benchmark: str
    prompt: str
    response: str
    is_correct: bool


PROMPT_TEMPLATE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""


# ============================================================
# DATA LOADING
# ============================================================

def load_text_samples(train_path: Path, test_path: Path) -> tuple[list, list]:
    """Load text training samples from JSONL files."""
    train_samples = []
    test_samples = []

    with open(train_path) as f:
        for line in f:
            data = json.loads(line)
            # Handle both string and dict input formats
            if isinstance(data.get("input"), dict):
                prompt = data["input"].get("question", str(data["input"]))
            else:
                prompt = data.get("input", "")

            train_samples.append(TextTrainingSample(
                question_id=data.get("id", ""),
                benchmark=data.get("benchmark", "unknown"),
                prompt=prompt,
                response=data.get("output", ""),
                is_correct=data.get("correct", False),
            ))

    with open(test_path) as f:
        for line in f:
            data = json.loads(line)
            if isinstance(data.get("input"), dict):
                prompt = data["input"].get("question", str(data["input"]))
            else:
                prompt = data.get("input", "")

            test_samples.append(TextTrainingSample(
                question_id=data.get("id", ""),
                benchmark=data.get("benchmark", "unknown"),
                prompt=prompt,
                response=data.get("output", ""),
                is_correct=data.get("correct", False),
            ))

    return train_samples, test_samples


# ============================================================
# VLM JUDGE (with gray image)
# ============================================================

class VLMTextDataset(torch.utils.data.Dataset):
    """Dataset for VLM judge training on text with gray placeholder image."""

    def __init__(self, samples: list[TextTrainingSample], processor, max_length=2048):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.gray_image = Image.new('RGB', (224, 224), color='gray')
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 256 * 28 * 28  # Fixed size for text

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        target = "ii" if sample.is_correct else "i"
        prompt = PROMPT_TEMPLATE.format(
            question=sample.prompt[:200],  # Shorter for text to avoid OOM
            response=sample.response[:100]
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.gray_image},
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
            images=[self.gray_image],
            return_tensors="pt",
            padding=True,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        # Find assistant token and mask everything before the answer
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


def get_vlm_p_correct(model, processor, question, response, device):
    """Extract P(correct) from VLM model logits."""
    gray_image = Image.new('RGB', (224, 224), color='gray')
    prompt = PROMPT_TEMPLATE.format(question=question, response=response)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": gray_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=[gray_image],
        return_tensors="pt",
        padding=True,
        min_pixels=256 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]

    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]

    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()


def train_vlm_text_judge(
    train_samples: List[TextTrainingSample],
    model_name: str,
    output_dir: Path,
    num_epochs: int = 3,
) -> tuple:
    """Train VLM judge on text samples with gray images."""
    start_time = time.time()

    print(f"\nLoading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

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
    train_dataset = VLMTextDataset(train_samples, processor)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="no",
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
    return training_time, model, processor


def evaluate_vlm_text_judge(
    model,
    processor,
    test_samples: List[TextTrainingSample],
    max_samples: int = None,
) -> dict:
    """Evaluate VLM judge on text samples."""
    model.eval()
    device = next(model.parameters()).device

    if max_samples and len(test_samples) > max_samples:
        np.random.seed(42)
        indices = np.random.choice(len(test_samples), max_samples, replace=False)
        test_samples = [test_samples[i] for i in indices]

    all_preds = []
    all_labels = []

    for i, sample in enumerate(test_samples):
        if i % 50 == 0:
            print(f"  Evaluating {i}/{len(test_samples)}...")

        try:
            p_correct = get_vlm_p_correct(
                model, processor,
                sample.prompt[:200], sample.response[:100], device
            )
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            p_correct = 0.5

        all_preds.append(p_correct)
        all_labels.append(float(sample.is_correct))

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    results = {
        "auroc": roc_auc_score(labels, preds),
        "auprc": average_precision_score(labels, preds),
        "n_samples": len(labels),
        "brier": float(np.mean((preds - labels) ** 2)),
    }

    # ECE
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
# PURE LLM JUDGE (no vision)
# ============================================================

class LLMTextDataset(torch.utils.data.Dataset):
    """Dataset for pure LLM judge training on text."""

    def __init__(self, samples: list[TextTrainingSample], tokenizer, max_length=1024):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        target = "ii" if sample.is_correct else "i"
        prompt = PROMPT_TEMPLATE.format(
            question=sample.prompt[:500],
            response=sample.response[:300]
        )

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        # Find where assistant response starts and mask everything before
        # Look for the pattern that ends the user turn
        text_tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # For Qwen2.5, find "assistant" token or similar marker
        # Simpler approach: mask all but last 3 tokens (answer token + end tokens)
        labels[:-3] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "labels": labels,
        }


def get_llm_p_correct(model, tokenizer, question, response, device):
    """Extract P(correct) from LLM model logits."""
    prompt = PROMPT_TEMPLATE.format(question=question, response=response)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]

    token_i = tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = tokenizer.encode("ii", add_special_tokens=False)[-1]

    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()


def train_llm_text_judge(
    train_samples: List[TextTrainingSample],
    model_name: str,
    output_dir: Path,
    num_epochs: int = 3,
) -> tuple:
    """Train pure LLM judge on text samples."""
    start_time = time.time()

    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Smaller model, can fit on fewer GPUs
    max_memory = {
        0: "40GiB",
        1: "40GiB",
        2: "40GiB",
        3: "40GiB",
    }

    model = AutoModelForCausalLM.from_pretrained(
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
    train_dataset = LLMTextDataset(train_samples, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,  # Larger batch for text-only
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
    )

    def collate_fn(batch):
        max_len = max(x["input_ids"].size(0) for x in batch)
        pad_token_id = tokenizer.pad_token_id or 0

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

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    print(f"Training with {len(train_samples)} samples, {num_epochs} epochs...")
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    training_time = time.time() - start_time
    return training_time, model, tokenizer


def evaluate_llm_text_judge(
    model,
    tokenizer,
    test_samples: List[TextTrainingSample],
    max_samples: int = None,
) -> dict:
    """Evaluate pure LLM judge on text samples."""
    model.eval()
    device = next(model.parameters()).device

    if max_samples and len(test_samples) > max_samples:
        np.random.seed(42)
        indices = np.random.choice(len(test_samples), max_samples, replace=False)
        test_samples = [test_samples[i] for i in indices]

    all_preds = []
    all_labels = []

    for i, sample in enumerate(test_samples):
        if i % 50 == 0:
            print(f"  Evaluating {i}/{len(test_samples)}...")

        try:
            p_correct = get_llm_p_correct(
                model, tokenizer,
                sample.prompt[:500], sample.response[:300], device
            )
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            p_correct = 0.5

        all_preds.append(p_correct)
        all_labels.append(float(sample.is_correct))

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    results = {
        "auroc": roc_auc_score(labels, preds),
        "auprc": average_precision_score(labels, preds),
        "n_samples": len(labels),
        "brier": float(np.mean((preds - labels) ** 2)),
    }

    # ECE
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

def run_vlm_ablation(train_samples, test_samples, output_base, training_sizes, num_epochs):
    """Run VLM text ablation experiment."""
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    results = []

    for size in training_sizes:
        print("\n" + "=" * 70)
        print(f"VLM TEXT ABLATION - SIZE: {size}")
        print("=" * 70)

        # Subsample with class balance
        np.random.seed(42)
        correct_samples = [s for s in train_samples if s.is_correct]
        incorrect_samples = [s for s in train_samples if not s.is_correct]

        n_each = size // 2
        n_correct_sample = min(n_each, len(correct_samples))
        n_incorrect_sample = min(n_each, len(incorrect_samples))

        correct_idx = np.random.choice(len(correct_samples), n_correct_sample, replace=False)
        incorrect_idx = np.random.choice(len(incorrect_samples), n_incorrect_sample, replace=False)

        train_subset = [correct_samples[i] for i in correct_idx] + [incorrect_samples[i] for i in incorrect_idx]
        np.random.shuffle(train_subset)

        print(f"Subsampled: {len(train_subset)} ({n_correct_sample} correct, {n_incorrect_sample} incorrect)")

        run_dir = output_base / f"vlm_size_{size}"
        run_dir.mkdir(exist_ok=True)

        # Train
        training_time, model, processor = train_vlm_text_judge(
            train_subset, model_name, run_dir, num_epochs=num_epochs
        )
        print(f"Training time: {training_time/60:.1f} minutes")

        # Evaluate
        print("\nEvaluating...")
        eval_results = evaluate_vlm_text_judge(model, processor, test_samples)
        print(f"Text AUROC: {eval_results['auroc']:.4f}")

        # Clean up
        del model
        torch.cuda.empty_cache()

        result = {
            "size": size,
            "n_train_actual": len(train_subset),
            "training_time_min": training_time / 60,
            "text_auroc": eval_results["auroc"],
            "text_auprc": eval_results["auprc"],
            "text_ece": eval_results["ece"],
            "text_brier": eval_results["brier"],
        }
        results.append(result)

        with open(run_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)

    return results


def run_llm_ablation(train_samples, test_samples, output_base, training_sizes, num_epochs):
    """Run pure LLM text ablation experiment."""
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    results = []

    for size in training_sizes:
        print("\n" + "=" * 70)
        print(f"PURE LLM TEXT ABLATION - SIZE: {size}")
        print("=" * 70)

        # Subsample with class balance
        np.random.seed(42)
        correct_samples = [s for s in train_samples if s.is_correct]
        incorrect_samples = [s for s in train_samples if not s.is_correct]

        n_each = size // 2
        n_correct_sample = min(n_each, len(correct_samples))
        n_incorrect_sample = min(n_each, len(incorrect_samples))

        correct_idx = np.random.choice(len(correct_samples), n_correct_sample, replace=False)
        incorrect_idx = np.random.choice(len(incorrect_samples), n_incorrect_sample, replace=False)

        train_subset = [correct_samples[i] for i in correct_idx] + [incorrect_samples[i] for i in incorrect_idx]
        np.random.shuffle(train_subset)

        print(f"Subsampled: {len(train_subset)} ({n_correct_sample} correct, {n_incorrect_sample} incorrect)")

        run_dir = output_base / f"llm_size_{size}"
        run_dir.mkdir(exist_ok=True)

        # Train
        training_time, model, tokenizer = train_llm_text_judge(
            train_subset, model_name, run_dir, num_epochs=num_epochs
        )
        print(f"Training time: {training_time/60:.1f} minutes")

        # Evaluate
        print("\nEvaluating...")
        eval_results = evaluate_llm_text_judge(model, tokenizer, test_samples)
        print(f"Text AUROC: {eval_results['auroc']:.4f}")

        # Clean up
        del model
        torch.cuda.empty_cache()

        result = {
            "size": size,
            "n_train_actual": len(train_subset),
            "training_time_min": training_time / 60,
            "text_auroc": eval_results["auroc"],
            "text_auprc": eval_results["auprc"],
            "text_ece": eval_results["ece"],
            "text_brier": eval_results["brier"],
        }
        results.append(result)

        with open(run_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", action="store_true", help="Run VLM text ablation")
    parser.add_argument("--llm", action="store_true", help="Run pure LLM text ablation")
    parser.add_argument("--both", action="store_true", help="Run both ablations")
    parser.add_argument("--size", type=int, help="Run only this size")
    args = parser.parse_args()

    if not any([args.vlm, args.llm, args.both]):
        print("Please specify --vlm, --llm, or --both")
        sys.exit(1)

    # Config
    training_sizes = [100, 250, 500, 1000, 2000]
    if args.size:
        training_sizes = [args.size]
    num_epochs = 3

    output_base = Path("data/ablations/text_training_size")
    output_base.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading text training data...")
    train_path = Path("data/finetune/train_v2.jsonl")
    test_path = Path("data/finetune/test_v2.jsonl")

    train_samples, test_samples = load_text_samples(train_path, test_path)
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    n_correct = sum(1 for s in train_samples if s.is_correct)
    print(f"Train correct: {n_correct} ({100*n_correct/len(train_samples):.1f}%)")

    all_results = {}

    # Run VLM ablation
    if args.vlm or args.both:
        print("\n" + "=" * 70)
        print("VLM TEXT ABLATION (Qwen3-VL-8B + gray image)")
        print("=" * 70)

        vlm_results = run_vlm_ablation(
            train_samples, test_samples, output_base, training_sizes, num_epochs
        )
        all_results["vlm"] = vlm_results

        print("\n" + "-" * 40)
        print("VLM TEXT ABLATION RESULTS")
        print("-" * 40)
        for r in vlm_results:
            print(f"Size {r['size']}: AUROC = {r['text_auroc']:.4f}, Time = {r['training_time_min']:.1f} min")

    # Run LLM ablation
    if args.llm or args.both:
        print("\n" + "=" * 70)
        print("PURE LLM TEXT ABLATION (Qwen2.5-7B)")
        print("=" * 70)

        llm_results = run_llm_ablation(
            train_samples, test_samples, output_base, training_sizes, num_epochs
        )
        all_results["llm"] = llm_results

        print("\n" + "-" * 40)
        print("PURE LLM TEXT ABLATION RESULTS")
        print("-" * 40)
        for r in llm_results:
            print(f"Size {r['size']}: AUROC = {r['text_auroc']:.4f}, Time = {r['training_time_min']:.1f} min")

    # Save all results
    with open(output_base / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_base / 'ablation_results.json'}")

    # Create comparison plot if both were run
    if args.both and "vlm" in all_results and "llm" in all_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        vlm_sizes = [r["size"] for r in all_results["vlm"]]
        vlm_aurocs = [r["text_auroc"] for r in all_results["vlm"]]
        llm_sizes = [r["size"] for r in all_results["llm"]]
        llm_aurocs = [r["text_auroc"] for r in all_results["llm"]]

        ax.plot(vlm_sizes, vlm_aurocs, 'o-', label='VLM (Qwen3-VL-8B)', linewidth=2, markersize=8)
        ax.plot(llm_sizes, llm_aurocs, 's--', label='LLM (Qwen2.5-7B)', linewidth=2, markersize=8)

        ax.set_xlabel('Number of Training Samples', fontsize=12)
        ax.set_ylabel('Text AUROC', fontsize=12)
        ax.set_title('Text Judge Training Size Ablation', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_xticks(vlm_sizes)
        ax.set_xticklabels([str(s) for s in vlm_sizes])

        plt.tight_layout()
        plt.savefig(output_base / "text_training_size_ablation.png", dpi=150)
        print(f"Plot saved to {output_base / 'text_training_size_ablation.png'}")


if __name__ == "__main__":
    main()
