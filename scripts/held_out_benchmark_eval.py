#!/usr/bin/env python3
"""Held-out benchmark evaluation: leave-K-out cross-validation analysis for the UQ calibrator.

CPU-only analysis of existing scored test data to assess per-benchmark generalization.
Groups test samples by benchmark and simulates leave-K-out folds:
  - For each fold, K=5 benchmarks are "held out" and the rest are "in-distribution"
  - Reports AUROC on in-dist vs held-out benchmarks using existing p_correct scores
  - Computes per-benchmark AUROC to identify easy/hard benchmarks

Since the current model was trained on ALL benchmarks, there are no truly held-out
benchmarks. But uniform per-benchmark AUROCs suggest the model generalizes well,
while large variance suggests benchmark-specific overfitting. This analysis informs
whether a full retrain with held-out benchmarks is needed.

Usage:
    python scripts/held_out_benchmark_eval.py
    python scripts/held_out_benchmark_eval.py --smoke_test
    python scripts/held_out_benchmark_eval.py --scored_dir data/use_cases/scored_test_only_v2 \
        --split_info uq_models/best_unified/split_info.json \
        --output_dir data/use_cases/results_test_only_v2
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scored(path: str) -> list:
    """Load a scored JSONL file, coercing types as needed."""
    samples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)

            # is_correct: may be int, float, or string
            ic = row.get("is_correct")
            if isinstance(ic, str):
                row["is_correct"] = int(float(ic))
            else:
                row["is_correct"] = int(ic)

            # p_correct: may be float or string
            pc = row.get("p_correct")
            if pc is None:
                row["p_correct"] = None
            elif isinstance(pc, str):
                row["p_correct"] = float(pc)
            else:
                row["p_correct"] = float(pc)

            # verbalized_confidence: may be float, string, or "None"
            vc = row.get("verbalized_confidence")
            if vc is None or vc == "None" or vc == "none":
                row["verbalized_confidence"] = None
            elif isinstance(vc, str):
                row["verbalized_confidence"] = float(vc)
            else:
                row["verbalized_confidence"] = float(vc)

            samples.append(row)
    return samples


def load_all_scored(scored_dir: str) -> list:
    """Load all three scored JSONL files and combine."""
    files = [
        "gpt5mini_scored.jsonl",
        "gpt52_scored.jsonl",
        "qwen35_scored.jsonl",
    ]
    all_samples = []
    for fname in files:
        path = os.path.join(scored_dir, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping.")
            continue
        samples = load_scored(path)
        print(f"  Loaded {len(samples):,} samples from {fname}")
        all_samples.extend(samples)
    return all_samples


def filter_test_only(samples: list, split_info_path: str) -> list:
    """Filter samples to test-only IDs using split_info.json."""
    with open(split_info_path) as f:
        split_info = json.load(f)
    test_ids = set(split_info["test_ids"])
    train_ids = set(split_info.get("train_ids", []))
    # Support both benchmark-prefixed and bare IDs
    filtered = []
    for s in samples:
        bare_id = s["id"]
        bench = s.get("benchmark", "unknown")
        prefixed_id = f"{bench}_{bare_id}"
        if prefixed_id in test_ids or (bare_id in test_ids and prefixed_id not in train_ids):
            filtered.append(s)
    print(f"  Filtered to {len(filtered):,} test-only samples "
          f"(from {len(test_ids):,} test IDs)")
    return filtered


# ---------------------------------------------------------------------------
# AUROC computation (safe)
# ---------------------------------------------------------------------------

def safe_auroc(labels, scores):
    """Compute AUROC, returning None if undefined (single class or <2 samples)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(labels) < 2:
        return None
    unique = np.unique(labels)
    if len(unique) < 2:
        return None
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Per-benchmark analysis
# ---------------------------------------------------------------------------

def per_benchmark_auroc(samples: list) -> dict:
    """Compute AUROC for each benchmark individually.

    Returns dict mapping benchmark -> {auroc, n_samples, n_correct, n_incorrect, accuracy}.
    """
    by_bench = defaultdict(list)
    for s in samples:
        by_bench[s["benchmark"]].append(s)

    results = {}
    for bench in sorted(by_bench.keys()):
        group = by_bench[bench]
        labels = np.array([s["is_correct"] for s in group])
        scores = np.array([s["p_correct"] for s in group])
        auroc = safe_auroc(labels, scores)
        n_correct = int(labels.sum())
        n_total = len(labels)
        results[bench] = {
            "auroc": auroc,
            "n_samples": n_total,
            "n_correct": n_correct,
            "n_incorrect": n_total - n_correct,
            "accuracy": round(n_correct / n_total, 4) if n_total > 0 else None,
        }
    return results


# ---------------------------------------------------------------------------
# Per-benchmark, per-target-model analysis
# ---------------------------------------------------------------------------

def per_benchmark_model_auroc(samples: list) -> dict:
    """Compute AUROC for each (benchmark, target_model) pair.

    Returns nested dict: benchmark -> target_model -> {auroc, n_samples, ...}.
    """
    by_key = defaultdict(list)
    for s in samples:
        key = (s["benchmark"], s.get("target_model", "unknown"))
        by_key[key].append(s)

    results = {}
    for (bench, model), group in sorted(by_key.items()):
        labels = np.array([s["is_correct"] for s in group])
        scores = np.array([s["p_correct"] for s in group])
        auroc = safe_auroc(labels, scores)
        n_correct = int(labels.sum())
        n_total = len(labels)
        if bench not in results:
            results[bench] = {}
        results[bench][model] = {
            "auroc": auroc,
            "n_samples": n_total,
            "n_correct": n_correct,
            "n_incorrect": n_total - n_correct,
            "accuracy": round(n_correct / n_total, 4) if n_total > 0 else None,
        }
    return results


# ---------------------------------------------------------------------------
# Leave-K-out cross-validation
# ---------------------------------------------------------------------------

def leave_k_out_folds(benchmarks: list, k: int, seed: int = 42) -> list:
    """Generate folds for leave-K-out cross-validation.

    Each fold holds out K benchmarks and keeps the rest as in-distribution.
    Benchmarks are shuffled deterministically and assigned to non-overlapping folds.

    Returns list of dicts with keys: fold_idx, held_out, in_dist.
    """
    rng = random.Random(seed)
    shuffled = list(benchmarks)
    rng.shuffle(shuffled)

    n_folds = len(shuffled) // k
    folds = []
    for i in range(n_folds):
        start = i * k
        end = start + k
        held_out = shuffled[start:end]
        in_dist = [b for b in shuffled if b not in set(held_out)]
        folds.append({
            "fold_idx": i,
            "held_out": sorted(held_out),
            "in_dist": sorted(in_dist),
        })

    # Handle remainder: if benchmarks don't divide evenly, the leftover
    # benchmarks form one last (smaller) fold.
    remainder_start = n_folds * k
    if remainder_start < len(shuffled):
        held_out = shuffled[remainder_start:]
        in_dist = [b for b in shuffled if b not in set(held_out)]
        folds.append({
            "fold_idx": n_folds,
            "held_out": sorted(held_out),
            "in_dist": sorted(in_dist),
        })

    return folds


def evaluate_fold(samples: list, fold: dict) -> dict:
    """Evaluate AUROC on in-dist and held-out benchmarks for one fold."""
    held_out_set = set(fold["held_out"])

    in_dist_samples = [s for s in samples if s["benchmark"] not in held_out_set]
    held_out_samples = [s for s in samples if s["benchmark"] in held_out_set]

    in_dist_labels = np.array([s["is_correct"] for s in in_dist_samples])
    in_dist_scores = np.array([s["p_correct"] for s in in_dist_samples])
    held_out_labels = np.array([s["is_correct"] for s in held_out_samples])
    held_out_scores = np.array([s["p_correct"] for s in held_out_samples])

    in_dist_auroc = safe_auroc(in_dist_labels, in_dist_scores)
    held_out_auroc = safe_auroc(held_out_labels, held_out_scores)

    gap = None
    if in_dist_auroc is not None and held_out_auroc is not None:
        gap = round(in_dist_auroc - held_out_auroc, 4)

    return {
        "fold_idx": fold["fold_idx"],
        "held_out_benchmarks": fold["held_out"],
        "in_dist_benchmarks": fold["in_dist"],
        "n_in_dist": len(in_dist_samples),
        "n_held_out": len(held_out_samples),
        "in_dist_auroc": round(in_dist_auroc, 4) if in_dist_auroc is not None else None,
        "held_out_auroc": round(held_out_auroc, 4) if held_out_auroc is not None else None,
        "gap": gap,
    }


def run_leave_k_out(samples: list, k: int = 5, seed: int = 42) -> list:
    """Run full leave-K-out cross-validation over benchmarks."""
    benchmarks = sorted(set(s["benchmark"] for s in samples))
    n_full_folds = len(benchmarks) // k
    n_remainder = len(benchmarks) % k
    total_folds = n_full_folds + (1 if n_remainder > 0 else 0)
    print(f"\n  {len(benchmarks)} unique benchmarks, K={k} "
          f"-> {n_full_folds} full folds"
          + (f" + 1 partial fold ({n_remainder} benchmarks)" if n_remainder else ""))

    folds = leave_k_out_folds(benchmarks, k, seed=seed)
    fold_results = []
    for fold in folds:
        result = evaluate_fold(samples, fold)
        fold_results.append(result)

    return fold_results


# ---------------------------------------------------------------------------
# Verbalized confidence baseline
# ---------------------------------------------------------------------------

def verbalized_baseline_auroc(samples: list) -> dict:
    """Compute per-benchmark AUROC using verbalized confidence as a baseline."""
    by_bench = defaultdict(list)
    for s in samples:
        if s["verbalized_confidence"] is not None:
            by_bench[s["benchmark"]].append(s)

    results = {}
    for bench in sorted(by_bench.keys()):
        group = by_bench[bench]
        labels = np.array([s["is_correct"] for s in group])
        scores = np.array([s["verbalized_confidence"] for s in group])
        auroc = safe_auroc(labels, scores)
        results[bench] = {
            "auroc": auroc,
            "n_samples": len(group),
        }
    return results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_per_benchmark_table(per_bench: dict, verb_baseline: dict):
    """Print a markdown-style table of per-benchmark AUROCs."""
    print("\n" + "=" * 90)
    print("PER-BENCHMARK AUROC (test set, all target models pooled)")
    print("=" * 90)
    header = (f"| {'Benchmark':<20} | {'N':>6} | {'Acc':>6} | "
              f"{'AUROC':>7} | {'Verb AUROC':>10} | {'Delta':>7} | {'Rating':<8} |")
    print(header)
    sep = (f"|{'-' * 22}|{'-' * 8}|{'-' * 8}|"
           f"{'-' * 9}|{'-' * 12}|{'-' * 9}|{'-' * 10}|")
    print(sep)

    for bench in sorted(per_bench.keys()):
        info = per_bench[bench]
        auroc = info["auroc"]
        n = info["n_samples"]
        acc = info["accuracy"]

        # Verbalized baseline
        vb = verb_baseline.get(bench, {})
        vb_auroc = vb.get("auroc")

        # Delta: calibrator - verbalized
        delta = None
        if auroc is not None and vb_auroc is not None:
            delta = auroc - vb_auroc

        # Rating based on AUROC
        if auroc is None:
            rating = "N/A"
        elif auroc >= 0.80:
            rating = "EASY"
        elif auroc >= 0.65:
            rating = "MEDIUM"
        else:
            rating = "HARD"

        auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
        vb_str = f"{vb_auroc:.4f}" if vb_auroc is not None else "N/A"
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        acc_str = f"{acc:.3f}" if acc is not None else "N/A"

        print(f"| {bench:<20} | {n:>6} | {acc_str:>6} | "
              f"{auroc_str:>7} | {vb_str:>10} | {delta_str:>7} | {rating:<8} |")

    # Summary statistics
    valid_aurocs = [v["auroc"] for v in per_bench.values() if v["auroc"] is not None]
    if valid_aurocs:
        print(f"\nSummary: mean AUROC = {np.mean(valid_aurocs):.4f}, "
              f"std = {np.std(valid_aurocs):.4f}, "
              f"min = {np.min(valid_aurocs):.4f}, "
              f"max = {np.max(valid_aurocs):.4f}")
        cv = np.std(valid_aurocs) / np.mean(valid_aurocs) if np.mean(valid_aurocs) > 0 else float("inf")
        print(f"  Coefficient of variation: {cv:.3f}")


def print_per_benchmark_model_table(per_bench_model: dict):
    """Print a markdown-style table of per-(benchmark, model) AUROCs."""
    print("\n" + "=" * 90)
    print("PER-BENCHMARK, PER-TARGET-MODEL AUROC")
    print("=" * 90)

    # Collect all models
    all_models = set()
    for bench_data in per_bench_model.values():
        all_models.update(bench_data.keys())
    all_models = sorted(all_models)

    # Header
    model_cols = "".join(f" | {m:>12}" for m in all_models)
    header = f"| {'Benchmark':<20}{model_cols} |"
    print(header)
    sep_parts = f"|{'-' * 22}" + "".join(f"|{'-' * 14}" for _ in all_models) + "|"
    print(sep_parts)

    for bench in sorted(per_bench_model.keys()):
        bench_data = per_bench_model[bench]
        parts = f"| {bench:<20}"
        for model in all_models:
            info = bench_data.get(model)
            if info and info["auroc"] is not None:
                parts += f" | {info['auroc']:>10.4f}  "
            elif info:
                parts += f" | {'N/A':>10}  "
            else:
                parts += f" | {'--':>10}  "
        parts += "|"
        print(parts)


def print_fold_table(fold_results: list):
    """Print a markdown-style table of leave-K-out fold results."""
    print("\n" + "=" * 90)
    print("LEAVE-K-OUT CROSS-VALIDATION (simulated on existing scores)")
    print("=" * 90)
    header = (f"| {'Fold':>4} | {'In-Dist N':>9} | {'Held-Out N':>10} | "
              f"{'In-Dist AUROC':>13} | {'Held-Out AUROC':>14} | {'Gap':>7} |")
    print(header)
    sep = (f"|{'-' * 6}|{'-' * 11}|{'-' * 12}|"
           f"{'-' * 15}|{'-' * 16}|{'-' * 9}|")
    print(sep)

    gaps = []
    for fr in fold_results:
        in_str = f"{fr['in_dist_auroc']:.4f}" if fr["in_dist_auroc"] is not None else "N/A"
        ho_str = f"{fr['held_out_auroc']:.4f}" if fr["held_out_auroc"] is not None else "N/A"
        gap_str = f"{fr['gap']:+.4f}" if fr["gap"] is not None else "N/A"

        print(f"| {fr['fold_idx']:>4} | {fr['n_in_dist']:>9,} | {fr['n_held_out']:>10,} | "
              f"{in_str:>13} | {ho_str:>14} | {gap_str:>7} |")
        print(f"|      | Held out: {', '.join(fr['held_out_benchmarks'])}")

        if fr["gap"] is not None:
            gaps.append(fr["gap"])

    if gaps:
        print(f"\nMean gap (in-dist - held-out): {np.mean(gaps):+.4f} "
              f"(std: {np.std(gaps):.4f})")
        print(f"  Positive gap = in-dist scores better than held-out (expected)")
        print(f"  Small gap = good generalization")


def print_overall_summary(samples: list):
    """Print overall aggregate metrics."""
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s["p_correct"] for s in samples])
    auroc = safe_auroc(labels, scores)

    # Verbalized baseline (where available)
    verb_samples = [s for s in samples if s["verbalized_confidence"] is not None]
    verb_auroc = None
    if verb_samples:
        vl = np.array([s["is_correct"] for s in verb_samples])
        vs = np.array([s["verbalized_confidence"] for s in verb_samples])
        verb_auroc = safe_auroc(vl, vs)

    print("\n" + "=" * 90)
    print("OVERALL TEST SET SUMMARY")
    print("=" * 90)
    print(f"  Total test samples:   {len(samples):,}")
    print(f"  Correct:              {int(labels.sum()):,} ({labels.mean():.1%})")
    print(f"  Incorrect:            {int((1 - labels).sum()):,} ({1 - labels.mean():.1%})")
    if auroc is not None:
        print(f"  Calibrator AUROC:     {auroc:.4f}")
    else:
        print(f"  Calibrator AUROC:     N/A")
    if verb_auroc is not None:
        print(f"  Verbalized AUROC:     {verb_auroc:.4f} "
              f"({len(verb_samples):,} samples with verb. conf.)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Held-out benchmark evaluation for UQ calibrator (CPU-only analysis)"
    )
    parser.add_argument(
        "--scored_dir",
        default="data/use_cases/scored_test_only_v2",
        help="Directory with scored JSONL files (default: data/use_cases/scored_test_only_v2)",
    )
    parser.add_argument(
        "--split_info",
        default="uq_models/best_unified/split_info.json",
        help="Path to split_info.json with train/test IDs "
             "(default: uq_models/best_unified/split_info.json)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/use_cases/results_test_only_v2",
        help="Directory to write results JSON (default: data/use_cases/results_test_only_v2)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of benchmarks to hold out per fold (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fold assignment (default: 42)",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run on 5 samples per benchmark for quick testing",
    )
    args = parser.parse_args()

    print("=" * 90)
    print("HELD-OUT BENCHMARK EVALUATION (CPU-only analysis of existing scores)")
    print("=" * 90)

    # ------------------------------------------------------------------
    # 1. Load scored data and filter to test set
    # ------------------------------------------------------------------
    print("\n[1/6] Loading scored data...")
    all_samples = load_all_scored(args.scored_dir)
    if not all_samples:
        print("ERROR: No scored data found. Exiting.")
        sys.exit(1)

    print(f"\n[2/6] Filtering to test-only samples...")
    test_samples = filter_test_only(all_samples, args.split_info)
    if not test_samples:
        print("ERROR: No test samples after filtering. Exiting.")
        sys.exit(1)

    # Remove samples with missing p_correct
    n_before = len(test_samples)
    test_samples = [s for s in test_samples if s["p_correct"] is not None]
    if len(test_samples) < n_before:
        print(f"  Removed {n_before - len(test_samples)} samples with missing p_correct")

    # Smoke test: subsample to 5 per benchmark
    if args.smoke_test:
        print("\n  ** SMOKE TEST: limiting to 5 samples per benchmark **")
        by_bench = defaultdict(list)
        for s in test_samples:
            by_bench[s["benchmark"]].append(s)
        test_samples = []
        for bench, group in by_bench.items():
            test_samples.extend(group[:5])
        print(f"  Smoke test: {len(test_samples)} samples across "
              f"{len(by_bench)} benchmarks")

    # ------------------------------------------------------------------
    # 2. Overall summary
    # ------------------------------------------------------------------
    print("\n[3/6] Computing overall metrics...")
    print_overall_summary(test_samples)

    # ------------------------------------------------------------------
    # 3. Per-benchmark AUROC
    # ------------------------------------------------------------------
    print("\n[4/6] Computing per-benchmark metrics...")
    per_bench = per_benchmark_auroc(test_samples)
    verb_baseline = verbalized_baseline_auroc(test_samples)
    print_per_benchmark_table(per_bench, verb_baseline)

    # Per-benchmark, per-model breakdown
    per_bench_model = per_benchmark_model_auroc(test_samples)
    print_per_benchmark_model_table(per_bench_model)

    # ------------------------------------------------------------------
    # 4. Leave-K-out cross-validation
    # ------------------------------------------------------------------
    print(f"\n[5/6] Running leave-{args.k}-out cross-validation (seed={args.seed})...")
    fold_results = run_leave_k_out(test_samples, k=args.k, seed=args.seed)
    print_fold_table(fold_results)

    # ------------------------------------------------------------------
    # 5. Identify hard vs easy benchmarks
    # ------------------------------------------------------------------
    print(f"\n[6/6] Benchmark difficulty ranking...")
    ranked = sorted(
        [(b, info) for b, info in per_bench.items() if info["auroc"] is not None],
        key=lambda x: x[1]["auroc"],
    )

    skipped = [(b, info) for b, info in per_bench.items() if info["auroc"] is None]
    if skipped:
        print(f"\n  SKIPPED ({len(skipped)} benchmarks, single-class or too few samples):")
        for b, info in skipped:
            print(f"    {b:<20} N={info['n_samples']}, "
                  f"correct={info['n_correct']}, incorrect={info['n_incorrect']}")

    n_show = min(5, len(ranked))

    print(f"\n  HARDEST benchmarks (lowest AUROC):")
    for b, info in ranked[:n_show]:
        print(f"    {b:<20} AUROC={info['auroc']:.4f}  "
              f"(N={info['n_samples']}, acc={info['accuracy']:.3f})")

    print(f"\n  EASIEST benchmarks (highest AUROC):")
    for b, info in ranked[-n_show:]:
        print(f"    {b:<20} AUROC={info['auroc']:.4f}  "
              f"(N={info['n_samples']}, acc={info['accuracy']:.3f})")

    # Uniformity assessment
    valid_aurocs = [info["auroc"] for _, info in ranked]
    mean_auroc = float(np.mean(valid_aurocs))
    std_auroc = float(np.std(valid_aurocs))
    cv = std_auroc / mean_auroc if mean_auroc > 0 else float("inf")

    print(f"\n  Uniformity assessment:")
    print(f"    Mean per-benchmark AUROC: {mean_auroc:.4f}")
    print(f"    Std:  {std_auroc:.4f}")
    print(f"    CV:   {cv:.3f}")
    if cv < 0.10:
        verdict = "LOW variance: model generalizes uniformly across benchmarks."
    elif cv < 0.20:
        verdict = "MODERATE variance: some benchmark-specific effects."
    else:
        verdict = "HIGH variance: significant benchmark-specific performance differences."
    print(f"    -> {verdict}")

    # Fold gap assessment
    fold_gaps = [fr["gap"] for fr in fold_results if fr["gap"] is not None]
    mean_gap = float(np.mean(fold_gaps)) if fold_gaps else None
    std_gap = float(np.std(fold_gaps)) if fold_gaps else None
    if fold_gaps:
        print(f"\n  Leave-K-out gap assessment:")
        print(f"    Mean gap (in-dist - held-out): {mean_gap:+.4f} "
              f"(std: {std_gap:.4f})")
        if abs(mean_gap) < 0.02:
            gap_verdict = ("NEGLIGIBLE gap: full retrain with held-out "
                           "benchmarks likely unnecessary.")
        elif abs(mean_gap) < 0.05:
            gap_verdict = ("SMALL gap: model generalizes reasonably, "
                           "but full held-out retrain could confirm.")
        else:
            gap_verdict = ("NOTABLE gap: consider full retrain with "
                           "truly held-out benchmarks.")
        print(f"    -> {gap_verdict}")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "held_out_eval.json")

    # Compute overall AUROC for the output
    overall_labels = np.array([s["is_correct"] for s in test_samples])
    overall_scores = np.array([s["p_correct"] for s in test_samples])
    overall_auroc = safe_auroc(overall_labels, overall_scores)

    output = {
        "config": {
            "scored_dir": args.scored_dir,
            "split_info": args.split_info,
            "k": args.k,
            "seed": args.seed,
            "smoke_test": args.smoke_test,
            "n_test_samples": len(test_samples),
            "n_benchmarks": len(per_bench),
        },
        "overall": {
            "auroc": round(overall_auroc, 4) if overall_auroc is not None else None,
            "n_samples": len(test_samples),
            "n_correct": int(overall_labels.sum()),
            "accuracy": round(float(overall_labels.mean()), 4),
        },
        "per_benchmark": per_bench,
        "per_benchmark_model": per_bench_model,
        "verbalized_baseline": verb_baseline,
        "leave_k_out": {
            "k": args.k,
            "n_folds": len(fold_results),
            "folds": fold_results,
            "mean_gap": round(mean_gap, 4) if mean_gap is not None else None,
            "std_gap": round(std_gap, 4) if std_gap is not None else None,
        },
        "difficulty_ranking": [
            {"benchmark": b, **info} for b, info in ranked
        ],
        "uniformity": {
            "mean_auroc": round(mean_auroc, 4) if valid_aurocs else None,
            "std_auroc": round(std_auroc, 4) if valid_aurocs else None,
            "cv": round(cv, 4) if cv != float("inf") else None,
            "verdict": verdict if valid_aurocs else None,
        },
        "gap_assessment": {
            "mean_gap": round(mean_gap, 4) if mean_gap is not None else None,
            "std_gap": round(std_gap, 4) if std_gap is not None else None,
            "verdict": gap_verdict if fold_gaps else None,
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
