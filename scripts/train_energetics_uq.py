#!/usr/bin/env python3
"""Domain-adapt the unified UQ model to the energetics benchmark.

Continues LoRA fine-tuning from the best_unified checkpoint on energetics
benchmark data. Splits by question_id so test questions are truly unseen.

Usage:
    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/train_energetics_uq.py \
        --output_dir uq_models/energetics_uq_smoke --smoke_test

    # Full training (1 GPU, ~15-20 min)
    CUDA_VISIBLE_DEVICES=0 python scripts/train_energetics_uq.py \
        --output_dir uq_models/energetics_uq

    # Custom split ratio
    CUDA_VISIBLE_DEVICES=0 python scripts/train_energetics_uq.py \
        --output_dir uq_models/energetics_uq --test_fraction 0.25
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_BASE_CHECKPOINT = "uq_models/best_unified"
DEFAULT_INPUT = "/scratch/khayes/energetics_bench/scoring/results/uq_input.jsonl"

PROMPT_TEMPLATE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""

GRAY_IMAGE = Image.new('RGB', (336, 336), color='gray')


# ============================================================
# DATA
# ============================================================

@dataclass
class Sample:
    id: str
    question_id: str
    model: str
    question: str
    response: str
    is_correct: bool


def load_energetics_data(input_path: str):
    """Load energetics QA data from JSONL."""
    samples = []
    with open(input_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            if "question" not in row or "response" not in row:
                continue

            is_correct = row.get("is_correct", False)
            if isinstance(is_correct, str):
                is_correct = is_correct.lower() in ("true", "1", "yes")

            samples.append(Sample(
                id=f"{row.get('question_id', f'q{i}')}_{row.get('model', 'unknown')}",
                question_id=row.get("question_id", f"q{i}"),
                model=row.get("model", "unknown"),
                question=row["question"],
                response=row["response"],
                is_correct=bool(is_correct),
            ))

    return samples


def split_by_question(samples, test_fraction=0.2, seed=42):
    """Split by question_id so no question appears in both train and test."""
    question_ids = sorted(set(s.question_id for s in samples))
    print(f"Unique questions: {len(question_ids)}")

    train_qids, test_qids = train_test_split(
        question_ids, test_size=test_fraction, random_state=seed
    )
    train_qids = set(train_qids)
    test_qids = set(test_qids)

    train = [s for s in samples if s.question_id in train_qids]
    test = [s for s in samples if s.question_id in test_qids]

    return train, test


# ============================================================
# DATASET
# ============================================================

class EnergeticsUQDataset(torch.utils.data.Dataset):
    """Dataset for energetics UQ training (text-only, gray placeholder image)."""

    def __init__(self, samples: list[Sample], processor, max_length=2048):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.image = GRAY_IMAGE
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 256 * 28 * 28  # fixed for text-only

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        target = "ii" if sample.is_correct else "i"

        prompt = PROMPT_TEMPLATE.format(
            question=sample.question[:500],
            response=sample.response[:300],
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": self.image},
                {"type": "text", "text": prompt},
            ]},
            {"role": "assistant", "content": target},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=[self.image],
            return_tensors="pt",
            padding=True,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        # Labels: mask everything except the answer token
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

def evaluate_model(model, processor, test_samples: list[Sample], device):
    """Evaluate model on test set, report overall + per-model AUROC."""
    model.eval()

    all_preds, all_labels = [], []
    per_model = defaultdict(lambda: {"preds": [], "labels": []})

    for i, sample in enumerate(test_samples):
        if i % 100 == 0:
            print(f"  Eval {i}/{len(test_samples)}...")

        prompt = PROMPT_TEMPLATE.format(
            question=sample.question[:500],
            response=sample.response[:300],
        )

        messages = [{"role": "user", "content": [
            {"type": "image", "image": GRAY_IMAGE},
            {"type": "text", "text": prompt},
        ]}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[GRAY_IMAGE], return_tensors="pt", padding=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28,
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
        per_model[sample.model]["preds"].append(p_correct)
        per_model[sample.model]["labels"].append(float(sample.is_correct))

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    results = {
        "auroc": float(roc_auc_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
        "brier": float(brier_score_loss(labels, preds)),
        "n_samples": len(labels),
        "n_correct": int(sum(labels)),
        "mean_p_correct": float(np.mean(preds)),
    }

    # ECE
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, 11)
    for j in range(10):
        in_bin = (preds >= bin_boundaries[j]) & (preds < bin_boundaries[j + 1])
        if in_bin.sum() == 0:
            continue
        ece += float((in_bin.sum() / len(preds)) * abs(labels[in_bin].mean() - preds[in_bin].mean()))
    results["ece"] = ece

    # Per-model
    results["per_model"] = {}
    for model_name, data in sorted(per_model.items()):
        mp, ml = np.array(data["preds"]), np.array(data["labels"])
        entry = {"n_samples": len(ml), "accuracy": float(ml.mean()),
                 "mean_p_correct": float(mp.mean())}
        if len(set(ml)) > 1:
            entry["auroc"] = float(roc_auc_score(ml, mp))
        results["per_model"][model_name] = entry

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Domain-adapt UQ model to energetics")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Input JSONL with energetics QA pairs")
    parser.add_argument("--output_dir", default="uq_models/energetics_uq",
                        help="Output directory for adapted model")
    parser.add_argument("--base_checkpoint", default=DEFAULT_BASE_CHECKPOINT,
                        help="Base UQ checkpoint to continue from")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("ENERGETICS UQ DOMAIN ADAPTATION")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Base checkpoint: {args.base_checkpoint}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}, LR: {args.learning_rate}")
    print(f"Test fraction: {args.test_fraction}")
    print()

    # --- Step 1: Load data ---
    print("STEP 1: Loading energetics data...")
    all_samples = load_energetics_data(args.input)

    if args.smoke_test:
        # Take a small subset but preserve question diversity
        qids = sorted(set(s.question_id for s in all_samples))[:20]
        all_samples = [s for s in all_samples if s.question_id in set(qids)]
        print(f"SMOKE TEST: using {len(all_samples)} samples from {len(qids)} questions")

    n_correct = sum(s.is_correct for s in all_samples)
    by_model = defaultdict(int)
    for s in all_samples:
        by_model[s.model] += 1

    print(f"Total: {len(all_samples)} samples")
    print(f"Correct: {n_correct}/{len(all_samples)} ({100*n_correct/len(all_samples):.1f}%)")
    for m, c in sorted(by_model.items()):
        print(f"  {m}: {c}")

    # --- Step 2: Split by question ---
    print(f"\nSTEP 2: Splitting by question_id ({1-args.test_fraction:.0%}/{args.test_fraction:.0%})...")
    train_samples, test_samples = split_by_question(
        all_samples, test_fraction=args.test_fraction
    )

    train_qids = set(s.question_id for s in train_samples)
    test_qids = set(s.question_id for s in test_samples)
    print(f"Train: {len(train_samples)} samples ({len(train_qids)} questions)")
    print(f"Test:  {len(test_samples)} samples ({len(test_qids)} questions)")
    assert len(train_qids & test_qids) == 0, "Question leak between train and test!"

    # --- Step 3: Load model from checkpoint ---
    print(f"\nSTEP 3: Loading model from {args.base_checkpoint}...")
    from transformers import (
        Qwen3VLForConditionalGeneration, AutoProcessor,
        TrainingArguments, Trainer, TrainerCallback,
    )
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s)")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load existing LoRA weights
    lora_path = args.base_checkpoint
    subdirs = [
        d for d in sorted(os.listdir(lora_path))
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(lora_path, d))
    ]
    if subdirs and not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        lora_path = os.path.join(lora_path, subdirs[-1])
    print(f"Loading LoRA from: {lora_path}")

    model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --- Step 4: Create datasets ---
    print("\nSTEP 4: Creating datasets...")
    train_dataset = EnergeticsUQDataset(train_samples, processor)

    def collate_fn(batch):
        max_len = max(x["input_ids"].size(0) for x in batch)
        pad_token_id = processor.tokenizer.pad_token_id or 0

        input_ids, attention_mask, labels_list = [], [], []
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
            labels_list.append(torch.cat([
                torch.full((pad_len,), -100, dtype=x["labels"].dtype),
                x["labels"]
            ]))

        result = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels_list),
        }
        if "pixel_values" in batch[0]:
            result["pixel_values"] = torch.cat([x["pixel_values"] for x in batch], dim=0)
        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in batch])
        return result

    # --- Step 5: Train ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save split info
    split_info = {
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "train_question_ids": sorted(train_qids),
        "test_question_ids": sorted(test_qids),
        "base_checkpoint": args.base_checkpoint,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        bf16_full_eval=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
    )

    class MemoryCleanupCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 50 == 0:
                torch.cuda.empty_cache()
            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[MemoryCleanupCallback()],
    )

    eff_batch = args.batch_size * args.grad_accum * max(num_gpus, 1)
    steps_per_epoch = len(train_dataset) // eff_batch

    print(f"\n{'='*50}")
    print("STARTING DOMAIN ADAPTATION")
    print(f"{'='*50}")
    print(f"Epochs: {args.epochs}, LR: {args.learning_rate}")
    print(f"Effective batch: {eff_batch}, Steps/epoch: ~{steps_per_epoch}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save
    print(f"\nSaving to {output_dir}...")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    # --- Step 6: Evaluate ---
    print(f"\n{'='*50}")
    print("EVALUATING ON HELD-OUT QUESTIONS")
    print(f"{'='*50}")

    device = next(model.parameters()).device
    results = evaluate_model(model, processor, test_samples, device)

    print(f"\nOverall AUROC: {results['auroc']:.4f}")
    print(f"Brier score:   {results['brier']:.4f}")
    print(f"ECE:           {results['ece']:.4f}")
    print(f"Mean P(correct): {results['mean_p_correct']:.4f}")

    print(f"\nPer-model:")
    for model_name, data in sorted(results["per_model"].items()):
        auroc = data.get("auroc", "N/A")
        if isinstance(auroc, float):
            print(f"  {model_name}: AUROC={auroc:.4f} (n={data['n_samples']}, "
                  f"acc={data['accuracy']:.1%}, mean_p={data['mean_p_correct']:.3f})")
        else:
            print(f"  {model_name}: {auroc} (n={data['n_samples']})")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")

    # --- Step 7: Score ALL data (for the energetics project to use) ---
    print(f"\n{'='*50}")
    print("SCORING ALL SAMPLES WITH ADAPTED MODEL")
    print(f"{'='*50}")

    all_rescored = load_energetics_data(args.input)
    if args.smoke_test:
        qids = sorted(set(s.question_id for s in all_rescored))[:20]
        all_rescored = [s for s in all_rescored if s.question_id in set(qids)]

    output_scored = output_dir / "energetics_scored_adapted.jsonl"
    all_scores = []

    with open(output_scored, "w") as f_out:
        for i, sample in enumerate(all_rescored):
            if i % 100 == 0:
                print(f"  Scoring {i}/{len(all_rescored)}...")

            prompt = PROMPT_TEMPLATE.format(
                question=sample.question[:500],
                response=sample.response[:300],
            )

            messages = [{"role": "user", "content": [
                {"type": "image", "image": GRAY_IMAGE},
                {"type": "text", "text": prompt},
            ]}]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[GRAY_IMAGE], return_tensors="pt", padding=True,
                min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28,
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
            except Exception:
                p_correct = 0.5

            all_scores.append(p_correct)

            in_train = sample.question_id in train_qids
            out_row = {
                "question_id": sample.question_id,
                "model": sample.model,
                "is_correct": sample.is_correct,
                "uq_score": round(p_correct, 6),
                "split": "train" if in_train else "test",
            }
            f_out.write(json.dumps(out_row) + "\n")

    print(f"Scored {len(all_rescored)} samples → {output_scored}")
    print(f"Mean UQ score: {np.mean(all_scores):.4f}")

    # Quick AUROC on test-only
    test_scores = [(s, sc) for s, sc in zip(all_rescored, all_scores)
                   if s.question_id in test_qids]
    if test_scores:
        t_labels = [float(s.is_correct) for s, _ in test_scores]
        t_preds = [sc for _, sc in test_scores]
        if len(set(t_labels)) > 1:
            print(f"Test-only AUROC: {roc_auc_score(t_labels, t_preds):.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
