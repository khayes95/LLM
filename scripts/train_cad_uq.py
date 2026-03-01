#!/usr/bin/env python3
"""Train a CAD-domain UQ model to predict CadQuery code correctness.

Adapted from train_best_uq.py. Trains Qwen3-VL-8B-Instruct + LoRA on
(NL description, CadQuery code, label) triples to predict whether
generated code is correct (compiles, valid geometry, etc.).

v1: Text-only (gray placeholder images). The VLM backbone is preserved
so v2 can incorporate rendered CAD images without retraining from scratch.

Input data format (JSONL):
    {"prompt": "Create a 10x10x10 cube", "code": "import cadquery as cq\n...", "label": 1}
    {"prompt": "Make a sphere r=5", "code": "import cadquery as cq\n...", "label": 0}

Optional fields:
    - "id": unique sample identifier (auto-generated if missing)
    - "source": metadata about where the sample came from
    - "error": error message if label=0 (not used in training, just metadata)

Usage:
    # Full training (4 GPUs, ~1-2 hours depending on dataset size)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_cad_uq.py \
        --data_file data/cad_triples.jsonl \
        --output_dir uq_models/cad_uq_v1

    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/train_cad_uq.py \
        --data_file data/cad_triples.jsonl \
        --output_dir uq_models/cad_uq_smoke --smoke_test

    # Resume from checkpoint
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_cad_uq.py \
        --data_file data/cad_triples.jsonl \
        --output_dir uq_models/cad_uq_v1 \
        --resume_from_checkpoint uq_models/cad_uq_v1/checkpoint-100
"""
import argparse
import json
import os
import sys
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

PROMPT_TEMPLATE = """Description: {prompt}

CadQuery Code:
```python
{code}
```

Is the code correct? (i) No (ii) Yes"""


# ============================================================
# DATA LOADING
# ============================================================

@dataclass
class Sample:
    id: str
    prompt: str
    code: str
    is_correct: bool


def load_data(data_file: str, max_samples: Optional[int] = None) -> list[Sample]:
    """Load (prompt, code, label) triples from JSONL."""
    samples = []
    with open(data_file) as f:
        for i, line in enumerate(f):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = row.get("prompt", "")
            code = row.get("code", "")
            label = row.get("label", -1)

            if not prompt or not code or label not in (0, 1):
                continue

            samples.append(Sample(
                id=str(row.get("id", i)),
                prompt=prompt[:2000],
                code=code[:3000],
                is_correct=bool(label == 1),
            ))

    if max_samples and len(samples) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in idx]

    return samples


# ============================================================
# TRAINING DATASET
# ============================================================

class CadUQDataset(torch.utils.data.Dataset):
    """Dataset for CAD UQ training. Text-only v1 with gray placeholder images."""

    def __init__(self, samples: list[Sample], processor, max_length=2048):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        # Gray placeholder — keeps VLM backbone active for future image support
        self.placeholder_image = Image.new('RGB', (336, 336), color='gray')
        # Fixed small resolution for placeholder (no wasted compute on fake pixels)
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 256 * 28 * 28

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        target = "ii" if sample.is_correct else "i"

        text = PROMPT_TEMPLATE.format(
            prompt=sample.prompt[:800],
            code=sample.code[:1200],
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": self.placeholder_image},
                {"type": "text", "text": text},
            ]},
            {"role": "assistant", "content": target},
        ]

        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[chat_text],
            images=[self.placeholder_image],
            return_tensors="pt",
            padding=True,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        # Labels: mask everything except the answer token
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        assistant_token = 77091  # Qwen3-VL assistant header token
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
    """Evaluate trained model on test set."""
    model.eval()
    placeholder = Image.new('RGB', (336, 336), color='gray')

    all_preds, all_labels = [], []

    for i, sample in enumerate(test_samples):
        if i % 100 == 0:
            print(f"  Eval {i}/{len(test_samples)}...")

        text = PROMPT_TEMPLATE.format(
            prompt=sample.prompt[:800],
            code=sample.code[:1200],
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": placeholder},
            {"type": "text", "text": text},
        ]}]

        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[chat_text], images=[placeholder], return_tensors="pt",
            padding=True, min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28,
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
                print(f"  Error on sample {i}: {e}")
            p_correct = 0.5

        all_preds.append(p_correct)
        all_labels.append(float(sample.is_correct))

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    results = {
        "auroc": float(roc_auc_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
        "auprc": float(average_precision_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
        "brier": float(brier_score_loss(labels, preds)),
        "ece": 0.0,
        "n_samples": len(labels),
        "n_correct": int(sum(labels)),
        "accuracy_rate": float(labels.mean()),
    }

    # ECE (10-bin)
    bin_boundaries = np.linspace(0, 1, 11)
    for j in range(10):
        in_bin = (preds >= bin_boundaries[j]) & (preds < bin_boundaries[j + 1])
        if in_bin.sum() == 0:
            continue
        results["ece"] += float(
            (in_bin.sum() / len(preds)) * abs(labels[in_bin].mean() - preds[in_bin].mean())
        )

    # Calibration by confidence bucket
    results["calibration_buckets"] = {}
    for j in range(10):
        lo, hi = bin_boundaries[j], bin_boundaries[j + 1]
        in_bin = (preds >= lo) & (preds < hi)
        if in_bin.sum() > 0:
            results["calibration_buckets"][f"{lo:.1f}-{hi:.1f}"] = {
                "count": int(in_bin.sum()),
                "mean_pred": float(preds[in_bin].mean()),
                "mean_actual": float(labels[in_bin].mean()),
            }

    return results


# ============================================================
# BEST-OF-N SCORING (standalone utility)
# ============================================================

def score_candidates(model, processor, prompt: str, candidates: list[str], device) -> list[float]:
    """Score multiple candidate codes for a single prompt. Returns P(correct) for each.

    This is the core function for Best-of-N selection at inference time.
    """
    model.eval()
    placeholder = Image.new('RGB', (336, 336), color='gray')
    scores = []

    for code in candidates:
        text = PROMPT_TEMPLATE.format(prompt=prompt[:800], code=code[:1200])
        messages = [{"role": "user", "content": [
            {"type": "image", "image": placeholder},
            {"type": "text", "text": text},
        ]}]

        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[chat_text], images=[placeholder], return_tensors="pt",
            padding=True, min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
        token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
        probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
        scores.append(probs[1].item())

    return scores


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train CAD-domain UQ model")
    parser.add_argument("--data_file", type=str, required=True,
                        help="JSONL file with {prompt, code, label} triples")
    parser.add_argument("--output_dir", type=str, default="uq_models/cad_uq_v1")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick test with 20 samples")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true",
                        help="Load checkpoint and evaluate only, no training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for --eval_only mode")
    args = parser.parse_args()

    max_samples = 20 if args.smoke_test else None

    print("=" * 70)
    print("TRAINING CAD UQ MODEL")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Data:  {args.data_file}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}, LR: {args.learning_rate}, LoRA r: {args.lora_r}")
    print()

    # --- Step 1: Load data ---
    print("STEP 1: Loading data...")
    all_samples = load_data(args.data_file, max_samples=max_samples)

    n_correct = sum(1 for s in all_samples if s.is_correct)
    print(f"Total: {len(all_samples)} samples")
    print(f"Correct: {n_correct} ({100 * n_correct / len(all_samples):.1f}%)")
    print(f"Incorrect: {len(all_samples) - n_correct} ({100 * (len(all_samples) - n_correct) / len(all_samples):.1f}%)")

    # --- Step 2: Train/test split ---
    print(f"\nSTEP 2: Creating train/test split ({1 - args.test_fraction:.0%}/{args.test_fraction:.0%})...")
    indices = list(range(len(all_samples)))
    labels_for_split = [int(s.is_correct) for s in all_samples]

    try:
        train_idx, test_idx = train_test_split(
            indices, test_size=args.test_fraction, random_state=42,
            stratify=labels_for_split,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            indices, test_size=args.test_fraction, random_state=42,
        )

    train_samples = [all_samples[i] for i in train_idx]
    test_samples = [all_samples[i] for i in test_idx]
    print(f"Train: {len(train_samples)} | Test: {len(test_samples)}")

    # Save split info
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "split_info.json", "w") as f:
        json.dump({
            "data_file": args.data_file,
            "n_train": len(train_samples),
            "n_test": len(test_samples),
            "train_ids": [s.id for s in train_samples],
            "test_ids": [s.id for s in test_samples],
        }, f)

    # --- Step 3: Load model ---
    print(f"\nSTEP 3: Loading {MODEL_NAME}...")
    from transformers import (
        Qwen3VLForConditionalGeneration, AutoProcessor,
        TrainingArguments, Trainer, TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model, PeftModel

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    num_gpus = torch.cuda.device_count()
    max_memory = {i: "78GiB" for i in range(num_gpus)}
    print(f"Using {num_gpus} GPUs")

    if args.eval_only and args.checkpoint:
        # Load fine-tuned checkpoint
        print(f"Loading checkpoint: {args.checkpoint}")
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
            max_memory=max_memory, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
            max_memory=max_memory, trust_remote_code=True,
        )
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if args.eval_only:
        print("\n--eval_only: Skipping training, running evaluation...")
        device = next(model.parameters()).device
        results = evaluate_model(model, processor, test_samples, device)
        _print_results(results)
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_dir / 'eval_results.json'}")
        return

    # --- Step 4: Create dataset ---
    print("\nSTEP 4: Creating dataset...")
    train_dataset = CadUQDataset(train_samples, processor)

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
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
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
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
    )

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

    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)
    eff_batch = args.batch_size * args.grad_accum * max(num_gpus, 1)
    print(f"Epochs: {args.epochs}, Effective batch: {eff_batch}")
    print(f"Steps/epoch: ~{len(train_dataset) // eff_batch}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save
    print(f"\nSaving to {output_dir}...")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    # --- Step 6: Evaluate ---
    print("\n" + "=" * 50)
    print("EVALUATING ON TEST SET")
    print("=" * 50)

    device = next(model.parameters()).device
    results = evaluate_model(model, processor, test_samples, device)
    _print_results(results)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")


def _print_results(results):
    print(f"\nAUROC:  {results['auroc']:.4f}")
    print(f"AUPRC:  {results['auprc']:.4f}")
    print(f"Brier:  {results['brier']:.4f}")
    print(f"ECE:    {results['ece']:.4f}")
    print(f"N:      {results['n_samples']} ({results['n_correct']} correct, "
          f"{results['accuracy_rate']:.1%} rate)")

    if results.get("calibration_buckets"):
        print("\nCalibration:")
        for bucket, data in sorted(results["calibration_buckets"].items()):
            print(f"  {bucket}: n={data['count']}, "
                  f"pred={data['mean_pred']:.3f}, actual={data['mean_actual']:.3f}")


if __name__ == "__main__":
    main()
