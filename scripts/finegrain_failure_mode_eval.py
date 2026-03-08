#!/usr/bin/env python3
"""Per-failure-mode evaluation of finetuned FineGRAIN models.

For each of the 5 CV folds, loads the finetuned checkpoint and scores
the held-out model's data, then aggregates per-failure-mode AUROC
across all folds.

Usage:
    # Smoke test (1 fold, 5 samples per failure mode)
    CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_failure_mode_eval.py --smoke_test

    # Full evaluation (all 5 folds)
    CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_failure_mode_eval.py
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CV_DIR = Path("data/finegrain_uq/exp1_human_cv")
DATA_DIR = Path("data/finegrain_uq/finetune")
OUTPUT_DIR = Path("data/finegrain_uq/failure_mode_eval")

FOLDS = ["flux", "sd3.5_large", "sd3.5_medium", "sd3_m", "sd3_xl"]

PROMPT_TEMPLATE = """Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes"""

Q_TRUNC = 1500
R_TRUNC = 800


def load_samples(model_name):
    """Load JSONL samples for a given model, extract failure mode from question."""
    path = DATA_DIR / f"human_{model_name}.jsonl"
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            # Extract failure mode from question text
            q = d["question"]
            fm = "unknown"
            if "Failure mode being tested: " in q:
                fm = q.split("Failure mode being tested: ")[1].split("\n")[0]
            d["failure_mode"] = fm
            samples.append(d)
    return samples


def score_samples(model, processor, samples, batch_size=1):
    """Score samples and return list of (score, label, failure_mode)."""
    device = model.device

    # Get token IDs for "i" (No) and "ii" (Yes)
    yes_token = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
    no_token = processor.tokenizer.encode("i", add_special_tokens=False)[-1]

    results = []
    for i, sample in enumerate(samples):
        if i % 50 == 0:
            print(f"  Scoring {i}/{len(samples)}...", flush=True)

        # Load image
        img_path = sample.get("image_path", "")
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        prompt = PROMPT_TEMPLATE.format(
            benchmark=sample["benchmark"],
            source_model=sample["source_model"],
            question=sample["question"][:Q_TRUNC],
            response=sample["response"][:R_TRUNC],
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            yes_logit = logits[yes_token].item()
            no_logit = logits[no_token].item()

            # P(correct) = softmax over yes/no
            max_l = max(yes_logit, no_logit)
            p_yes = np.exp(yes_logit - max_l) / (np.exp(yes_logit - max_l) + np.exp(no_logit - max_l))

        results.append({
            "score": float(p_yes),
            "label": int(sample["is_correct"]),
            "failure_mode": sample["failure_mode"],
            "id": sample["id"],
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--folds", nargs="+", default=None,
                        help="Specific folds to evaluate (default: all)")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = args.folds or FOLDS
    max_per_fm = 5 if args.smoke_test else None

    print(f"Loading base model: {BASE_MODEL}")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    all_results = []
    fold_summaries = {}

    for fold_name in folds:
        print(f"\n{'='*60}")
        print(f"FOLD: {fold_name} (held-out model)")
        print(f"{'='*60}")

        checkpoint = CV_DIR / f"fold_{fold_name}" / "checkpoint-best"
        if not checkpoint.exists():
            print(f"  Checkpoint not found: {checkpoint}, skipping")
            continue

        # Load LoRA adapter
        print(f"  Loading checkpoint: {checkpoint}")
        model = PeftModel.from_pretrained(base_model, str(checkpoint))
        model.eval()

        # Load held-out model data
        samples = load_samples(fold_name)
        print(f"  Loaded {len(samples)} samples for {fold_name}")

        if max_per_fm:
            # Subsample per failure mode for smoke test
            by_fm = defaultdict(list)
            for s in samples:
                by_fm[s["failure_mode"]].append(s)
            samples = []
            for fm, fm_samples in by_fm.items():
                samples.extend(fm_samples[:max_per_fm])
            print(f"  Smoke test: {len(samples)} samples")

        # Score
        t0 = time.time()
        results = score_samples(model, processor, samples)
        elapsed = time.time() - t0
        print(f"  Scored in {elapsed:.1f}s")

        all_results.extend(results)

        # Per-failure-mode for this fold
        by_fm = defaultdict(lambda: {"scores": [], "labels": []})
        for r in results:
            by_fm[r["failure_mode"]]["scores"].append(r["score"])
            by_fm[r["failure_mode"]]["labels"].append(r["label"])

        fold_summary = {}
        for fm, data in sorted(by_fm.items()):
            n = len(data["labels"])
            n_pos = sum(data["labels"])
            if n_pos == 0 or n_pos == n:
                auroc = float("nan")
            else:
                auroc = roc_auc_score(data["labels"], data["scores"])
            fold_summary[fm] = {"auroc": auroc, "n": n, "n_correct": n_pos}

        fold_summaries[fold_name] = fold_summary

        # Unload adapter
        del model
        torch.cuda.empty_cache()

    # Aggregate across all folds
    print(f"\n{'='*60}")
    print("AGGREGATED PER-FAILURE-MODE (across all folds)")
    print(f"{'='*60}")

    by_fm_all = defaultdict(lambda: {"scores": [], "labels": []})
    for r in all_results:
        by_fm_all[r["failure_mode"]]["scores"].append(r["score"])
        by_fm_all[r["failure_mode"]]["labels"].append(r["label"])

    aggregate = {}
    for fm, data in sorted(by_fm_all.items()):
        n = len(data["labels"])
        n_pos = sum(data["labels"])
        if n_pos == 0 or n_pos == n:
            auroc = float("nan")
        else:
            auroc = roc_auc_score(data["labels"], data["scores"])
        acc = sum(1 for s, l in zip(data["scores"], data["labels"])
                  if (s >= 0.5) == bool(l)) / n
        aggregate[fm] = {
            "auroc": auroc,
            "accuracy": acc,
            "n": n,
            "n_correct": n_pos,
            "failure_rate": 1.0 - n_pos / n,  # is_correct=False means failure
        }

    # Sort by AUROC descending
    sorted_fms = sorted(aggregate.items(),
                        key=lambda x: x[1]["auroc"] if not np.isnan(x[1]["auroc"]) else -1,
                        reverse=True)

    print(f"\n{'Failure Mode':<50s} {'AUROC':>7s} {'Acc':>6s} {'N':>5s} {'FailRate':>8s}")
    print("-" * 80)
    aurocs = []
    for fm, m in sorted_fms:
        auroc_s = f"{m['auroc']:.3f}" if not np.isnan(m["auroc"]) else "  NaN"
        print(f"{fm:<50s} {auroc_s:>7s} {m['accuracy']:>6.1%} {m['n']:>5d} {m['failure_rate']:>8.1%}")
        if not np.isnan(m["auroc"]):
            aurocs.append(m["auroc"])

    overall_scores = [r["score"] for r in all_results]
    overall_labels = [r["label"] for r in all_results]
    overall_auroc = roc_auc_score(overall_labels, overall_scores)

    print(f"\nOverall AUROC: {overall_auroc:.4f}")
    print(f"Mean per-FM AUROC: {np.mean(aurocs):.4f} (std: {np.std(aurocs):.4f})")
    print(f"Min: {np.min(aurocs):.4f}, Max: {np.max(aurocs):.4f}")
    print(f"Total samples: {len(all_results)}")

    # Save
    output = {
        "overall_auroc": overall_auroc,
        "mean_per_fm_auroc": float(np.mean(aurocs)),
        "std_per_fm_auroc": float(np.std(aurocs)),
        "n_total": len(all_results),
        "per_failure_mode": {fm: m for fm, m in sorted_fms},
        "per_fold": fold_summaries,
    }

    out_path = output_dir / "failure_mode_breakdown.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
