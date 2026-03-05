#!/usr/bin/env python3
"""Score FineGRAIN new T2I models (with LLM judge labels) using our UQ calibrator.

For models evaluated by the FineGRAIN VLM+LLM pipeline (Molmo-72B + Llama-70B),
we score the same images with our UQ model and compare predictions.

Uses LLM judge boolean as pseudo-ground-truth. Computes:
  - AUROC of UQ P(failure) vs judge boolean
  - Cohen's kappa (agreement)
  - Spearman correlation of P(failure) vs judge severity score
  - Per-failure-mode and per-model breakdowns
  - T2I model ranking correlation

Usage:
    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_newmodels.py --smoke_test

    # Single model
    CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_newmodels.py --models flux2_dev

    # All models
    CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_newmodels.py --all
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
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, cohen_kappa_score,
)

# Import shared components from finegrain_uq_eval
sys.path.insert(0, str(Path(__file__).parent))
from finegrain_uq_eval import (
    load_model, score_finegrain_sample, compute_metrics, compute_bootstrap_ci,
    PROMPT_TEMPLATES, FAILURE_MODE_DESCRIPTIONS, BASE_MODEL, DEFAULT_CHECKPOINT,
)

# ============================================================
# CONFIG
# ============================================================

FINEGRAIN_DEV_DIR = Path("/scratch/khayes/finegrain_dev")
JUDGED_DIR = FINEGRAIN_DEV_DIR / "data/results/vlm_evaluation"

# All models with judged JSON files
ALL_MODELS = [
    "flux", "flux2_dev", "flux2_pro", "gemini_image", "gemini_image_native",
    "gpt_image1", "gpt_image1_resized", "gpt_image15", "hidream",
    "nano_banana2", "qwen", "sd1", "sd2", "seedream", "wan22",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_judged_data(model_name, max_samples=None):
    """Load judged JSON data for a model from finegrain_dev.

    Returns list of dicts with keys:
        prompt_text, failure_mode, image_path, judge_boolean, judge_score, model
    """
    json_path = JUDGED_DIR / f"metadata_{model_name}_tailored_vllm_judged.json"
    if not json_path.exists():
        print(f"  WARNING: {json_path} not found, skipping {model_name}")
        return []

    with open(json_path) as f:
        data = json.load(f)

    samples = []
    skipped_status = 0
    skipped_image = 0
    skipped_eval = 0

    for entry in data:
        # Filter to success status only
        status = entry.get("status", "success")
        if status != "success":
            skipped_status += 1
            continue

        # Resolve image path
        rel_path = entry.get("image_path", "")
        abs_path = FINEGRAIN_DEV_DIR / rel_path
        if not abs_path.exists():
            skipped_image += 1
            continue

        # Extract judge evaluation
        judge_eval = entry.get("llm_evaluation_tailored", {})
        if not judge_eval or "boolean" not in judge_eval or judge_eval["boolean"] is None:
            skipped_eval += 1
            continue

        samples.append({
            "prompt_text": entry["prompt"],
            "failure_mode": entry["failure_mode"],
            "image_path": str(abs_path),
            "judge_boolean": int(judge_eval["boolean"]),  # 1=failure, 0=compliant
            "judge_score": float(judge_eval.get("score", 0)),
            "judge_reasoning": judge_eval.get("reasoning", ""),
            "model": model_name,
            "index": entry.get("index", -1),
        })

    if skipped_status or skipped_image or skipped_eval:
        print(f"  {model_name}: loaded {len(samples)}, skipped "
              f"{skipped_status} non-success, {skipped_image} missing images, "
              f"{skipped_eval} missing eval")

    if max_samples and len(samples) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in sorted(indices)]

    return samples


# ============================================================
# METRICS
# ============================================================

def compute_agreement_metrics(judge_labels, uq_scores, judge_scores=None):
    """Compute agreement between UQ model and LLM judge."""
    judge_labels = np.array(judge_labels)
    uq_scores = np.array(uq_scores)
    failure_scores = 1.0 - uq_scores  # Higher = more likely failure

    results = {}

    # AUROC
    if len(set(judge_labels)) > 1:
        results["auroc"] = roc_auc_score(judge_labels, failure_scores)
    else:
        results["auroc"] = float("nan")

    # Binary predictions at optimal threshold
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (failure_scores >= t).astype(int)
        if judge_labels.sum() > 0:
            t_f1 = f1_score(judge_labels, preds, zero_division=0)
            if t_f1 > best_f1:
                best_f1, best_thresh = t_f1, t
    results["best_f1"] = best_f1
    results["best_threshold"] = best_thresh

    # Cohen's kappa at optimal threshold
    preds = (failure_scores >= best_thresh).astype(int)
    results["cohens_kappa"] = cohen_kappa_score(judge_labels, preds)
    results["accuracy"] = accuracy_score(judge_labels, preds)
    results["agreement_rate"] = float((preds == judge_labels).mean())

    # Correlation with judge severity score
    if judge_scores is not None:
        judge_scores = np.array(judge_scores)
        if np.std(judge_scores) > 0 and np.std(failure_scores) > 0:
            rho, p_val = spearmanr(failure_scores, judge_scores)
            results["spearman_rho_vs_score"] = rho
            results["spearman_p_vs_score"] = p_val

    results["n_samples"] = len(judge_labels)
    results["n_failures_judge"] = int(judge_labels.sum())
    results["failure_rate_judge"] = float(judge_labels.mean())
    results["mean_p_compliant"] = float(uq_scores.mean())

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Score FineGRAIN new models with UQ calibrator"
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt_variant", default="combined",
                        choices=list(PROMPT_TEMPLATES.keys()))
    parser.add_argument("--output_dir", default="data/finegrain_uq/newmodels")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to evaluate")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all 15 models")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run on 5 samples per model")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    if args.all:
        models = ALL_MODELS
    elif args.models:
        models = args.models
    else:
        # Default: a representative subset
        models = ["flux2_dev", "gpt_image15", "wan22"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_samples = 5 if args.smoke_test else args.max_samples

    print("=" * 70)
    print("FineGRAIN x UQ: New Model Evaluation")
    print(f"Models: {models}")
    print(f"Prompt variant: {args.prompt_variant}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 70)

    # Load UQ model
    print("\nLoading UQ model...")
    model, processor = load_model(args.checkpoint)
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    all_results = {}
    all_judge_labels = []
    all_uq_scores = []
    all_judge_scores = []

    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Scoring model: {model_name}")
        print(f"{'='*50}")

        samples = load_judged_data(model_name, max_samples=max_samples)
        if not samples:
            print(f"  No samples loaded for {model_name}, skipping")
            continue

        print(f"  Loaded {len(samples)} samples "
              f"({sum(s['judge_boolean'] for s in samples)} failures by judge)")

        # Score each sample
        scored_path = output_dir / f"scored_{model_name}.jsonl"
        uq_scores = []
        judge_labels = []
        judge_scores = []
        start_time = time.time()

        with open(scored_path, "w") as f_out:
            for i, sample in enumerate(samples):
                try:
                    # Reuse the scoring function from finegrain_uq_eval
                    # It expects a dict with prompt_text, failure_mode, image_path, model
                    p_correct = score_finegrain_sample(
                        model, processor, device, sample, args.prompt_variant
                    )
                except Exception as e:
                    print(f"  ERROR on sample {i}: {e}")
                    p_correct = 0.5

                uq_scores.append(p_correct)
                judge_labels.append(sample["judge_boolean"])
                judge_scores.append(sample["judge_score"])

                record = {
                    "idx": i,
                    "model": model_name,
                    "failure_mode": sample["failure_mode"],
                    "judge_boolean": sample["judge_boolean"],
                    "judge_score": sample["judge_score"],
                    "p_compliant": p_correct,
                    "prompt_text": sample["prompt_text"][:200],
                    "index": sample["index"],
                }
                f_out.write(json.dumps(record) + "\n")

                if (i + 1) % 50 == 0 or (i + 1) == len(samples):
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    print(f"  [{i+1}/{len(samples)}] {rate:.1f} samples/s")

        elapsed = time.time() - start_time

        # Compute metrics for this model
        metrics = compute_agreement_metrics(
            judge_labels, uq_scores, judge_scores
        )
        metrics["time_s"] = elapsed
        all_results[model_name] = metrics

        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  Cohen's kappa: {metrics['cohens_kappa']:.4f}")
        print(f"  Agreement: {metrics['agreement_rate']:.4f}")
        if "spearman_rho_vs_score" in metrics:
            print(f"  Spearman rho vs score: {metrics['spearman_rho_vs_score']:.4f}")

        # Accumulate for overall metrics
        all_judge_labels.extend(judge_labels)
        all_uq_scores.extend(uq_scores)
        all_judge_scores.extend(judge_scores)

    # Overall metrics across all models
    if all_judge_labels:
        print(f"\n{'='*70}")
        print("OVERALL RESULTS (all models combined)")
        print(f"{'='*70}")

        overall = compute_agreement_metrics(
            all_judge_labels, all_uq_scores, all_judge_scores
        )
        all_results["_overall"] = overall

        print(f"Total samples: {overall['n_samples']}")
        print(f"AUROC: {overall['auroc']:.4f}")
        print(f"Cohen's kappa: {overall['cohens_kappa']:.4f}")
        print(f"Agreement rate: {overall['agreement_rate']:.4f}")
        print(f"Best F1: {overall['best_f1']:.4f} (threshold={overall['best_threshold']:.2f})")

        # Model ranking correlation
        model_uq_means = {}
        model_judge_rates = {}
        for m in models:
            if m in all_results and m != "_overall":
                model_uq_means[m] = all_results[m]["mean_p_compliant"]
                model_judge_rates[m] = all_results[m]["failure_rate_judge"]

        if len(model_uq_means) >= 3:
            ordered = sorted(model_uq_means.keys())
            uq_vals = [1 - model_uq_means[m] for m in ordered]  # failure proxy
            judge_vals = [model_judge_rates[m] for m in ordered]
            rho, p = spearmanr(uq_vals, judge_vals)
            tau, p_tau = kendalltau(uq_vals, judge_vals)
            all_results["_ranking"] = {
                "spearman_rho": rho, "spearman_p": p,
                "kendall_tau": tau, "kendall_p": p_tau,
                "n_models": len(ordered),
            }
            print(f"\nModel ranking correlation:")
            print(f"  Spearman rho: {rho:.4f} (p={p:.4f})")
            print(f"  Kendall tau:  {tau:.4f} (p={p_tau:.4f})")

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
