#!/usr/bin/env python3
"""Unified UQ classifier training from predictions.jsonl files.

Loads data from runs/{model}_combined/*/predictions.jsonl format.
Works with GPT-5-mini, Qwen3-VL, InternVL, or any model with same output format.

The UQ classifier is a text-only model that predicts whether a response is correct
based on the question and answer text (no images needed for judging).

Usage:
    # Train on GPT-5-mini data
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_uq_unified.py \
        --data_dir runs/gpt5_mini_combined \
        --output_dir uq_models/gpt5_mini_uq

    # Train on Qwen3-VL data
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_uq_unified.py \
        --data_dir runs/qwen3_vl_combined \
        --output_dir uq_models/qwen3_vl_uq

    # Cross-model: Train on GPT-5, test on Qwen
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_uq_unified.py \
        --data_dir runs/gpt5_mini_combined \
        --test_data_dir runs/qwen3_vl_combined \
        --output_dir uq_models/gpt5_to_qwen_uq
"""
import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model

# Check if Qwen3-VL is available
try:
    from transformers import Qwen3VLForConditionalGeneration
    HAS_QWEN3_VL = True
except ImportError:
    HAS_QWEN3_VL = False

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# DATA LOADING
# ============================================================

@dataclass
class UQSample:
    """Training sample for UQ classifier."""
    id: str
    benchmark: str
    question: str
    response: str
    is_correct: bool
    confidence: Optional[float] = None


# Benchmarks to exclude from training (too easy, too hard, etc.)
EXCLUDED_BENCHMARKS = {
    "triviaqa",  # 97% accuracy - too easy
    "babilong",  # 93% accuracy - too easy
    "vsr",       # 87% accuracy - too easy
    "aokvqa",    # 92% accuracy - too easy
    "erqa",      # Multi-image benchmark, random AUROC - known limitation
    "tutorbench",  # 98% accuracy - too easy
    "healthbench", # 93% accuracy - too easy
}


def extract_question_text(input_data: dict) -> str:
    """Extract question text from input field."""
    if isinstance(input_data, str):
        return input_data

    # Handle dict with various field names
    if isinstance(input_data, dict):
        # Try common field names
        for key in ["question", "query", "query_cot", "prompt", "text"]:
            if key in input_data and input_data[key]:
                val = input_data[key]
                if isinstance(val, str):
                    return val

        # Try to get from messages if present
        if "messages" in input_data:
            for msg in input_data["messages"]:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        # Extract text parts
                        texts = [p.get("text", "") for p in content if isinstance(p, dict) and "text" in p]
                        return " ".join(texts)

        # Fallback: stringify dict (minus images)
        clean = {k: v for k, v in input_data.items() if k != "images"}
        return json.dumps(clean)[:1000]

    return str(input_data)[:1000]


def load_predictions_jsonl(path: Path) -> list[dict]:
    """Load predictions from JSONL file."""
    samples = []
    with open(path) as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def load_samples_from_dir(data_dir: Path, exclude_benchmarks: set = None) -> list[UQSample]:
    """Load all UQ samples from a model's combined output directory."""
    if exclude_benchmarks is None:
        exclude_benchmarks = EXCLUDED_BENCHMARKS

    samples = []

    for bench_dir in data_dir.iterdir():
        if not bench_dir.is_dir():
            continue

        benchmark = bench_dir.name

        # Skip excluded benchmarks
        if benchmark in exclude_benchmarks:
            print(f"  Skipping {benchmark} (excluded)")
            continue

        pred_file = bench_dir / "predictions.jsonl"
        if not pred_file.exists():
            continue

        predictions = load_predictions_jsonl(pred_file)
        bench_samples = []

        for pred in predictions:
            # Get correctness from score
            score = pred.get("score", {})
            correct = score.get("correct", -1)

            # Skip uncertain samples
            if correct not in (0, 1):
                continue

            # Extract question and response
            question = extract_question_text(pred.get("input", {}))

            # Get response - prefer full response_text, fall back to prediction.answer
            response = pred.get("response_text") or ""
            if not response:
                prediction = pred.get("prediction", {})
                if isinstance(prediction, dict):
                    response = prediction.get("answer") or ""
                else:
                    response = str(prediction)
            if not response:
                continue

            # Get confidence if available
            prediction = pred.get("prediction", {})
            confidence = None
            if isinstance(prediction, dict):
                confidence = prediction.get("confidence")

            bench_samples.append(UQSample(
                id=str(pred.get("id", "")),
                benchmark=benchmark,
                question=question[:2000],  # Limit length
                response=response[:1000],
                is_correct=bool(correct == 1),
                confidence=confidence,
            ))

        print(f"  {benchmark}: {len(bench_samples)} samples")
        samples.extend(bench_samples)

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

class UQDataset(torch.utils.data.Dataset):
    """Dataset for UQ classifier training."""

    def __init__(self, samples: list[UQSample], tokenizer, max_length=1024, is_vlm=False):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_vlm = is_vlm

        # For VLM models, we need a dummy image
        if is_vlm:
            self.dummy_image = Image.new('RGB', (336, 336), color='gray')
            # Minimal image resolution for VLM
            self.min_pixels = 256 * 28 * 28
            self.max_pixels = 256 * 28 * 28

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Target: "i" for incorrect, "ii" for correct
        target = "ii" if sample.is_correct else "i"

        prompt = PROMPT_TEMPLATE.format(
            question=sample.question[:800],
            response=sample.response[:400]
        )

        if self.is_vlm:
            # VLM path: use processor with dummy image
            # Use just {"type": "image"} placeholder in messages - the actual image
            # is passed separately to the processor
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # Placeholder only
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": target,
                }
            ]

            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # Process with VLM processor - pass image separately
            inputs = self.tokenizer(
                text=[text],
                images=[self.dummy_image],
                return_tensors="pt",
                padding=True,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )

            # Labels - mask everything except the assistant's answer token
            input_ids = inputs["input_ids"][0]
            labels = input_ids.clone()

            # Find the assistant token and train on the token after it
            assistant_token = 77091  # 'assistant' token for Qwen
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
                "labels": labels,
            }

            # Handle pixel_values
            if "pixel_values" in inputs:
                pv = inputs["pixel_values"]
                if isinstance(pv, list):
                    result["pixel_values"] = pv[0] if len(pv) > 0 else pv
                else:
                    result["pixel_values"] = pv.squeeze(0) if pv.dim() > 3 else pv

            # Handle image_grid_thw if present
            if "image_grid_thw" in inputs:
                result["image_grid_thw"] = inputs["image_grid_thw"]

            return result

        else:
            # Text-only path
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target}
            ]

            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

            # Create labels - mask everything except the assistant's answer
            input_ids = inputs["input_ids"].squeeze(0)
            labels = input_ids.clone()
            labels[:-3] = -100  # Keep last 3 tokens (answer + end tokens)

            return {
                "input_ids": input_ids,
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": labels,
            }


def collate_fn(batch, pad_token_id=0, is_vlm=False):
    """Collate function with left-padding for causal LM."""
    max_len = max(x["input_ids"].size(0) for x in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for x in batch:
        seq_len = x["input_ids"].size(0)
        pad_len = max_len - seq_len

        # Left-pad for causal LM
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

    # Handle VLM-specific fields
    if is_vlm and "pixel_values" in batch[0]:
        # Stack pixel_values - they should all have the same shape for dummy images
        pixel_values = [x["pixel_values"] for x in batch]
        result["pixel_values"] = torch.stack(pixel_values)

        # Handle image_grid_thw if present
        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in batch], dim=0)

    return result


# ============================================================
# EVALUATION
# ============================================================

def get_p_correct(model, tokenizer, question: str, response: str, device) -> float:
    """Extract P(correct) from model logits."""
    prompt = PROMPT_TEMPLATE.format(question=question[:800], response=response[:400])

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]

    # Get token IDs for "i" and "ii"
    token_i = tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = tokenizer.encode("ii", add_special_tokens=False)[-1]

    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()  # P(correct) = P("ii")


def evaluate_uq_classifier(
    model,
    tokenizer,
    test_samples: list[UQSample],
    device: torch.device,
) -> dict:
    """Evaluate UQ classifier and compute metrics."""
    model.eval()

    all_preds = []
    all_labels = []
    per_benchmark = defaultdict(lambda: {"preds": [], "labels": []})

    for i, sample in enumerate(test_samples):
        if i % 100 == 0:
            print(f"  Evaluating {i}/{len(test_samples)}...")

        try:
            p_correct = get_p_correct(
                model, tokenizer,
                sample.question, sample.response, device
            )
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            p_correct = 0.5

        all_preds.append(p_correct)
        all_labels.append(float(sample.is_correct))
        per_benchmark[sample.benchmark]["preds"].append(p_correct)
        per_benchmark[sample.benchmark]["labels"].append(float(sample.is_correct))

    # Compute overall metrics
    results = {
        "auroc": roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5,
        "auprc": average_precision_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5,
        "n_samples": len(all_labels),
        "n_correct": int(sum(all_labels)),
        "base_rate": sum(all_labels) / len(all_labels) if all_labels else 0.5,
    }

    # Per-benchmark metrics
    results["per_benchmark"] = {}
    for bench, data in per_benchmark.items():
        if len(set(data["labels"])) < 2:
            continue
        results["per_benchmark"][bench] = {
            "auroc": roc_auc_score(data["labels"], data["preds"]),
            "n_samples": len(data["labels"]),
            "n_correct": int(sum(data["labels"])),
        }

    # ECE and Brier
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    results["brier"] = float(np.mean((preds - labels) ** 2))

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
    results["ece"] = float(ece)

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Unified UQ classifier training")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to model's combined output dir (e.g., runs/gpt5_mini_combined)")
    parser.add_argument("--test_data_dir", type=str, default=None,
                        help="Optional: separate test data dir for cross-model evaluation")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for trained model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Base model for UQ classifier")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train/test split ratio (ignored if test_data_dir provided)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split")
    args = parser.parse_args()

    print("=" * 70)
    print("UNIFIED UQ CLASSIFIER TRAINING")
    print("=" * 70)
    print(f"\nData dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Base model: {args.model_name}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    print("\nLoading training data...")
    data_dir = Path(args.data_dir)
    all_samples = load_samples_from_dir(data_dir)
    print(f"\nTotal samples: {len(all_samples)}")

    # Count correct/incorrect
    n_correct = sum(1 for s in all_samples if s.is_correct)
    n_incorrect = len(all_samples) - n_correct
    print(f"Correct: {n_correct} ({100*n_correct/len(all_samples):.1f}%)")
    print(f"Incorrect: {n_incorrect} ({100*n_incorrect/len(all_samples):.1f}%)")

    # Split or load test data
    if args.test_data_dir:
        # Cross-model evaluation
        print(f"\nLoading test data from {args.test_data_dir}...")
        test_dir = Path(args.test_data_dir)
        train_samples = all_samples
        test_samples = load_samples_from_dir(test_dir)
        print(f"Test samples: {len(test_samples)}")
    else:
        # Random split
        np.random.seed(args.seed)
        indices = np.random.permutation(len(all_samples))
        split = int(args.train_ratio * len(all_samples))
        train_samples = [all_samples[i] for i in indices[:split]]
        test_samples = [all_samples[i] for i in indices[split:]]

    print(f"\nTrain: {len(train_samples)}, Test: {len(test_samples)}")

    n_correct_train = sum(1 for s in train_samples if s.is_correct)
    n_correct_test = sum(1 for s in test_samples if s.is_correct)
    print(f"Train correct: {n_correct_train} ({100*n_correct_train/len(train_samples):.1f}%)")
    print(f"Test correct: {n_correct_test} ({100*n_correct_test/len(test_samples):.1f}%)")

    # Load model
    print(f"\nLoading {args.model_name}...")

    # Detect if this is a VLM model
    is_vlm = "VL" in args.model_name or "vl" in args.model_name

    if is_vlm:
        # Use processor for VLM models
        tokenizer = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        # Ensure pad token for text-only models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Dynamically allocate based on visible GPUs
    num_gpus = torch.cuda.device_count()
    max_memory = {i: "78GiB" for i in range(num_gpus)}

    if is_vlm and HAS_QWEN3_VL and "Qwen3-VL" in args.model_name:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
        )

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)

    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = UQDataset(train_samples, tokenizer, is_vlm=is_vlm)

    # Calculate steps for ~25% checkpoints
    # With batch_size=4, grad_accum=8, effective_batch=32
    # steps_per_epoch = len(train_samples) / effective_batch
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(train_samples) // effective_batch
    total_steps = steps_per_epoch * args.epochs
    save_steps = max(1, total_steps // 4)  # Save every ~25%

    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total training steps: {total_steps}")
    print(f"  Saving every {save_steps} steps (~25%)")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=5,  # Keep more checkpoints for ~25% intervals
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
    )

    # Memory cleanup callback
    class MemoryCleanupCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 100 == 0:
                torch.cuda.empty_cache()
            return control

    # Get pad_token_id (handle both processor and tokenizer)
    if is_vlm:
        pad_token_id = tokenizer.tokenizer.pad_token_id
    else:
        pad_token_id = tokenizer.pad_token_id

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda batch: collate_fn(batch, pad_token_id, is_vlm=is_vlm),
        callbacks=[MemoryCleanupCallback()],
    )

    # Train
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.grad_accum}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"Learning rate: {args.learning_rate}")

    trainer.train()

    # Save
    print(f"\nSaving LoRA adapter to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Evaluate
    print("\n" + "=" * 50)
    print("EVALUATING ON TEST SET")
    print("=" * 50)

    device = next(model.parameters()).device
    results = evaluate_uq_classifier(model, tokenizer, test_samples, device)

    # Print results
    print("\n" + "=" * 70)
    print("UQ CLASSIFIER RESULTS")
    print("=" * 70)
    print(f"\nOverall AUROC: {results['auroc']:.4f}")
    print(f"Overall AUPRC: {results['auprc']:.4f}")
    print(f"ECE: {results['ece']:.4f}")
    print(f"Brier: {results['brier']:.4f}")
    print(f"Base rate: {results['base_rate']:.2%}")

    print("\nPer-benchmark AUROC:")
    for bench, metrics in sorted(results["per_benchmark"].items()):
        print(f"  {bench}: {metrics['auroc']:.4f} (n={metrics['n_samples']}, {metrics['n_correct']} correct)")

    # Interpret results
    print("\n" + "-" * 50)
    if results["auroc"] > 0.75:
        print("✓ AUROC > 0.75: Strong signal! UQ classifier works well.")
    elif results["auroc"] > 0.65:
        print("⚠ AUROC 0.65-0.75: Moderate signal.")
    elif results["auroc"] > 0.55:
        print("⚠ AUROC 0.55-0.65: Weak signal.")
    else:
        print("✗ AUROC ~0.50: No signal. Model may need more data or tuning.")

    # Save results
    results["config"] = {
        "data_dir": str(args.data_dir),
        "test_data_dir": str(args.test_data_dir) if args.test_data_dir else None,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save sample IDs for reproducibility
    train_ids = [{"id": s.id, "benchmark": s.benchmark} for s in train_samples]
    test_ids = [{"id": s.id, "benchmark": s.benchmark} for s in test_samples]

    with open(output_dir / "train_ids.json", "w") as f:
        json.dump(train_ids, f)
    with open(output_dir / "test_ids.json", "w") as f:
        json.dump(test_ids, f)

    print(f"\nTraining complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
