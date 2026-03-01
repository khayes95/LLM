#!/usr/bin/env python3
"""Retrain the best unified UQ model with optimal hyperparameters.

Uses r=32 LoRA rank and combined (longer+CoT+metadata) prompt template,
which ablations showed gives 0.867 AUROC vs 0.831 baseline.

This script reuses the data loading and training logic from train_best_uq.py
but with the better configuration.

Usage:
    # Full training (4 GPUs, ~2-3 hours)
    CUDA_VISIBLE_DEVICES=0,4,5,7 python scripts/retrain_best_v2.py

    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/retrain_best_v2.py --smoke_test
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
IMAGE_CACHE_DIR = Path("data/training_images")
OUTPUT_DIR = "uq_models/best_unified_v2"

# The combined prompt with longer context (from ablations)
PROMPT_TEMPLATE = """Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes"""

Q_TRUNCATION = 1500  # longer context
R_TRUNCATION = 800

# Data sources (same as train_best_uq.py)
DATA_SOURCES = {
    "gpt5mini": {"dir": "runs/gpt5_mini_combined", "type": "combined"},
    "gpt52": {"dir": "runs", "prefix": "gpt52_high_", "type": "prefixed"},
    "qwen35": {"dir": "runs", "prefix": "qwen35_397b_", "type": "prefixed"},
}

VLM_BENCHMARKS = {
    "charxiv", "mmmu", "mmstar", "hallusionbench", "mathverse",
    "mathvision", "mathvista", "realworldqa", "vizwiz", "hle_multimodal", "mmvet",
}

EXCLUDED = {
    "triviaqa", "babilong", "vsr", "aokvqa", "erqa", "tutorbench",
    "healthbench", "arc", "oolong",
}


@dataclass
class Sample:
    id: str
    benchmark: str
    source_model: str
    question: str
    response: str
    is_correct: bool
    has_image: bool


def extract_question_text(input_data) -> str:
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, dict):
        for key in ["question", "query", "query_cot", "prompt", "text"]:
            if key in input_data and input_data[key]:
                val = input_data[key]
                if isinstance(val, str):
                    return val
        if "messages" in input_data:
            for msg in input_data["messages"]:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        texts = [p.get("text", "") for p in content
                                 if isinstance(p, dict) and "text" in p]
                        return " ".join(texts)
        clean = {k: v for k, v in input_data.items() if k != "images"}
        return json.dumps(clean)[:2000]
    return str(input_data)[:2000]


def load_combined_dir(data_dir, source_model, max_per_benchmark=None):
    samples = []
    data_path = Path(data_dir)
    if not data_path.exists():
        return samples

    for bench_dir in sorted(data_path.iterdir()):
        if not bench_dir.is_dir():
            continue
        benchmark = bench_dir.name
        if benchmark in EXCLUDED:
            continue

        pred_file = bench_dir / "predictions.jsonl"
        if not pred_file.exists():
            continue

        bench_samples = []
        with open(pred_file) as f:
            for line in f:
                try:
                    pred = json.loads(line)
                except json.JSONDecodeError:
                    continue

                score = pred.get("score", {})
                if isinstance(score, dict):
                    correct = score.get("correct", -1)
                else:
                    correct = score
                if correct not in (0, 1):
                    continue

                question = extract_question_text(pred.get("input", {}))
                response = pred.get("response_text", "") or str(pred.get("prediction", ""))
                if not question or not response:
                    continue

                bench_samples.append(Sample(
                    id=str(pred.get("id", "")),
                    benchmark=benchmark,
                    source_model=source_model,
                    question=question[:4000],
                    response=response[:2000],
                    is_correct=bool(correct == 1),
                    has_image=benchmark in VLM_BENCHMARKS,
                ))

        if max_per_benchmark and len(bench_samples) > max_per_benchmark:
            np.random.seed(42)
            idx = np.random.choice(len(bench_samples), max_per_benchmark, replace=False)
            bench_samples = [bench_samples[i] for i in idx]
        samples.extend(bench_samples)

    return samples


def load_prefixed_runs(runs_dir, prefix, source_model, max_per_benchmark=None):
    samples = []
    runs_path = Path(runs_dir)

    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.name.startswith(prefix):
            continue
        benchmark = run_dir.name[len(prefix):]
        if benchmark in EXCLUDED:
            continue

        pred_file = run_dir / "predictions.jsonl"
        if not pred_file.exists():
            continue

        bench_samples = []
        with open(pred_file) as f:
            for line in f:
                try:
                    pred = json.loads(line)
                except json.JSONDecodeError:
                    continue

                score = pred.get("score", {})
                if isinstance(score, dict):
                    correct = score.get("correct", -1)
                else:
                    correct = score
                if correct not in (0, 1):
                    continue

                question = extract_question_text(pred.get("input", {}))
                response = pred.get("response_text", "")
                if not response:
                    response = str(pred.get("prediction", {}).get("answer", ""))
                if not question or not response:
                    continue

                bench_samples.append(Sample(
                    id=str(pred.get("id", "")),
                    benchmark=benchmark,
                    source_model=source_model,
                    question=question[:4000],
                    response=response[:2000],
                    is_correct=bool(correct == 1),
                    has_image=benchmark in VLM_BENCHMARKS,
                ))

        if max_per_benchmark and len(bench_samples) > max_per_benchmark:
            np.random.seed(42)
            idx = np.random.choice(len(bench_samples), max_per_benchmark, replace=False)
            bench_samples = [bench_samples[i] for i in idx]
        samples.extend(bench_samples)

    return samples


def load_all_samples(max_per_benchmark=None):
    all_samples = []
    for model_name, config in DATA_SOURCES.items():
        print(f"\nLoading {model_name}...")
        if config["type"] == "combined":
            samples = load_combined_dir(config["dir"], model_name, max_per_benchmark)
        else:
            samples = load_prefixed_runs(config["dir"], config["prefix"], model_name, max_per_benchmark)
        print(f"  {model_name}: {len(samples)} samples")
        all_samples.extend(samples)
    return all_samples


class UnifiedUQDataset(torch.utils.data.Dataset):
    def __init__(self, samples, processor, max_length=2048):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.fallback_image = Image.new('RGB', (336, 336), color='gray')
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 512 * 28 * 28

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if sample.has_image:
            cache_path = IMAGE_CACHE_DIR / sample.benchmark / f"{sample.id}.jpg"
            if cache_path.exists():
                try:
                    image = Image.open(cache_path).convert("RGB")
                except Exception:
                    image = self.fallback_image
            else:
                image = self.fallback_image
            min_px = self.min_pixels
            max_px = self.max_pixels
        else:
            image = self.fallback_image
            min_px = 256 * 28 * 28
            max_px = 256 * 28 * 28

        target = "ii" if sample.is_correct else "i"

        prompt = PROMPT_TEMPLATE.format(
            question=sample.question[:Q_TRUNCATION],
            response=sample.response[:R_TRUNCATION],
            benchmark=sample.benchmark,
            source_model=sample.source_model,
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
            {"role": "assistant", "content": target},
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True,
            min_pixels=min_px, max_pixels=max_px,
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


def custom_collate(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    result = {}
    for key in ["input_ids", "attention_mask", "labels"]:
        padded = []
        pad_val = 0 if key == "attention_mask" else -100 if key == "labels" else 0
        for b in batch:
            t = b[key]
            if t.size(0) < max_len:
                padding = torch.full((max_len - t.size(0),), pad_val, dtype=t.dtype)
                t = torch.cat([t, padding])
            padded.append(t)
        result[key] = torch.stack(padded)

    if "pixel_values" in batch[0]:
        result["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    if "image_grid_thw" in batch[0]:
        result["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)

    return result


def evaluate(model, processor, test_samples, device):
    model.eval()
    fallback = Image.new('RGB', (224, 224), color='gray')
    all_preds, all_labels = [], []
    per_benchmark = defaultdict(lambda: {"preds": [], "labels": []})

    for i, sample in enumerate(test_samples):
        if i % 100 == 0:
            print(f"  Eval {i}/{len(test_samples)}...")

        if sample.has_image:
            cache_path = IMAGE_CACHE_DIR / sample.benchmark / f"{sample.id}.jpg"
            if cache_path.exists():
                try:
                    image = Image.open(cache_path).convert("RGB")
                except Exception:
                    image = fallback
            else:
                image = fallback
        else:
            image = fallback

        prompt = PROMPT_TEMPLATE.format(
            question=sample.question[:Q_TRUNCATION],
            response=sample.response[:R_TRUNCATION],
            benchmark=sample.benchmark,
            source_model=sample.source_model,
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[image], return_tensors="pt", padding=True,
            min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
            token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
            probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
            p_correct = probs[1].item()
        except Exception as e:
            if i < 5:
                print(f"  Error: {e}")
            p_correct = 0.5

        all_preds.append(p_correct)
        all_labels.append(float(sample.is_correct))
        per_benchmark[sample.benchmark]["preds"].append(p_correct)
        per_benchmark[sample.benchmark]["labels"].append(float(sample.is_correct))

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    results = {
        "auroc": float(roc_auc_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
        "brier": float(brier_score_loss(labels, preds)),
        "n_samples": len(labels),
    }

    # VLM vs Text
    vlm_idx = [i for i, s in enumerate(test_samples) if s.has_image]
    txt_idx = [i for i, s in enumerate(test_samples) if not s.has_image]

    for label, idx in [("vlm", vlm_idx), ("text", txt_idx)]:
        if idx and len(set(labels[idx])) > 1:
            results[f"{label}_auroc"] = float(roc_auc_score(labels[idx], preds[idx]))

    # Per-benchmark
    results["per_benchmark"] = {}
    for bench, data in per_benchmark.items():
        bl = np.array(data["labels"])
        bp = np.array(data["preds"])
        if len(set(bl)) > 1:
            results["per_benchmark"][bench] = {
                "auroc": float(roc_auc_score(bl, bp)),
                "n": len(bl),
            }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    # Load all samples
    print("Loading samples...")
    max_per = 5 if args.smoke_test else None
    samples = load_all_samples(max_per_benchmark=max_per)
    print(f"\nTotal: {len(samples)} samples")

    # Train/test split (stratified by benchmark)
    benchmarks = [s.benchmark for s in samples]
    train_samples, test_samples = train_test_split(
        samples, test_size=args.test_fraction, random_state=args.seed,
        stratify=benchmarks,
    )
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    # Save split info
    split_info = {
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "n_train_vlm": sum(1 for s in train_samples if s.has_image),
        "n_test_vlm": sum(1 for s in test_samples if s.has_image),
        "train_ids": [s.id for s in train_samples],
        "test_ids": [s.id for s in test_samples],
        "config": {
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "prompt": "combined (longer + CoT + metadata)",
            "q_truncation": Q_TRUNCATION,
            "r_truncation": R_TRUNCATION,
            "seed": args.seed,
        }
    }
    with open(f"{args.output_dir}/split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)

    # Load model
    print("\nLoading model...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Datasets
    train_dataset = UnifiedUQDataset(train_samples, processor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=custom_collate, num_workers=4, pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training
    print(f"\nTraining: {args.epochs} epochs, {len(train_loader)} steps/epoch, "
          f"grad_accum={args.grad_accum}")
    print(f"Config: LoRA r={args.lora_r}, alpha={args.lora_alpha}, lr={args.learning_rate}")
    print(f"Prompt: combined (longer context 1500/800 + CoT + metadata)")

    best_auroc = 0
    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            try:
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum
                loss.backward()
            except Exception as e:
                print(f"  Error batch {batch_idx}: {e}")
                optimizer.zero_grad()
                continue

            epoch_loss += outputs.loss.item()
            n_batches += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 20 == 0:
                    avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
                    elapsed = time.time() - start_time
                    print(f"  [Epoch {epoch+1}/{args.epochs}] Step {global_step}/{total_steps} "
                          f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                          f"elapsed={elapsed/60:.1f}m")

        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
        print(f"\n  Epoch {epoch+1}/{args.epochs} complete. Avg loss: {avg_loss:.4f}")

        # Evaluate
        print(f"  Evaluating on test set...")
        results = evaluate(model, processor, test_samples, device)
        auroc = results["auroc"]
        print(f"  Test AUROC: {auroc:.4f} | VLM: {results.get('vlm_auroc', 'N/A')} | "
              f"Text: {results.get('text_auroc', 'N/A')}")

        if auroc > best_auroc:
            best_auroc = auroc
            print(f"  New best! Saving checkpoint...")
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)

            results["epoch"] = epoch + 1
            results["avg_loss"] = avg_loss
            with open(f"{args.output_dir}/results.json", 'w') as f:
                json.dump(results, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best AUROC: {best_auroc:.4f}")
    print(f"Checkpoint: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
