#!/usr/bin/env python3
"""Compute proper scoring rules beyond AUROC for all UQ methods.

Metrics per method:
  - AUROC (discrimination)
  - Brier score: mean((p - y)^2), lower is better
  - Log loss: -mean(y*log(p) + (1-y)*log(1-p)), lower is better
  - ECE at 10, 15, 20 bins
  - MCE (Maximum Calibration Error)
  - Reliability diagram data

Computed for: combined, per-target-model, per-benchmark.

Input:  data/use_cases/scored_test_only/{gpt5mini,gpt52,qwen35}_scored.jsonl
Output: data/use_cases/results_test_only/scoring_rules.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = "/scratch/khayes/LLM"
SCORED_DIR = os.path.join(BASE_DIR, "data/use_cases/scored_test_only")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/use_cases/results_test_only")

INPUT_FILES = [
    "gpt5mini_scored.jsonl",
    "gpt52_scored.jsonl",
    "qwen35_scored.jsonl",
]

# Method name -> field in JSONL
METHODS = {
    "calibrator": "p_correct",
    "platt": "p_platt_verbalized",
    "isotonic": "p_isotonic_verbalized",
    "length": "p_length_baseline",
    "combined": "p_combined_baseline",
    "zeroshot": "p_zeroshot",
    "verbalized": "verbalized_confidence",
}

ECE_BINS_LIST = [10, 15, 20]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC using the Mann-Whitney U statistic (no sklearn)."""
    if len(y_true) == 0:
        return float("nan")
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U
    # For each positive, count how many negatives it beats
    # Use broadcasting for speed; fall back to sorted approach for large arrays
    if len(pos) * len(neg) < 1e8:
        # Direct comparison (fast for moderate sizes)
        u = np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])
    else:
        # Sorted approach for very large arrays
        all_scores = np.concatenate([pos, neg])
        all_labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        order = np.argsort(all_scores)
        all_labels = all_labels[order]
        ranks = np.arange(1, len(all_labels) + 1, dtype=np.float64)
        # Handle ties: assign average rank
        sorted_scores = all_scores[order]
        i = 0
        while i < len(sorted_scores):
            j = i
            while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
                j += 1
            avg_rank = (ranks[i] + ranks[j - 1]) / 2.0
            ranks[i:j] = avg_rank
            i = j
        pos_rank_sum = np.sum(ranks[all_labels == 1])
        n_pos = len(pos)
        n_neg = len(neg)
        u = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    auroc = u / (len(pos) * len(neg))
    return float(auroc)


def compute_brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Brier score: mean((p - y)^2)."""
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_score - y_true) ** 2))


def compute_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Log loss with clipping."""
    if len(y_true) == 0:
        return float("nan")
    eps = 1e-7
    p = np.clip(y_score, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def compute_ece_mce(y_true: np.ndarray, y_score: np.ndarray, n_bins: int):
    """Expected and Maximum Calibration Error.

    Returns (ece, mce, reliability_data).
    reliability_data is a list of dicts with bin_center, bin_count, mean_predicted, mean_actual.
    """
    if len(y_true) == 0:
        return float("nan"), float("nan"), []

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = []
    weighted_abs_errors = []
    max_abs_error = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (y_score >= lo) & (y_score < hi)
        else:
            # Last bin is inclusive on both sides
            mask = (y_score >= lo) & (y_score <= hi)
        count = int(np.sum(mask))
        if count == 0:
            reliability.append({
                "bin_center": float((lo + hi) / 2.0),
                "bin_count": 0,
                "mean_predicted": None,
                "mean_actual": None,
            })
            continue
        mean_p = float(np.mean(y_score[mask]))
        mean_y = float(np.mean(y_true[mask]))
        abs_err = abs(mean_p - mean_y)
        weighted_abs_errors.append(abs_err * count / total)
        if abs_err > max_abs_error:
            max_abs_error = abs_err
        reliability.append({
            "bin_center": float((lo + hi) / 2.0),
            "bin_count": count,
            "mean_predicted": mean_p,
            "mean_actual": mean_y,
        })

    ece = float(sum(weighted_abs_errors)) if weighted_abs_errors else float("nan")
    mce = float(max_abs_error) if weighted_abs_errors else float("nan")
    return ece, mce, reliability


def compute_all_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute all scoring rules for one (method, slice) pair."""
    n = len(y_true)
    if n == 0:
        return {"n": 0, "error": "no samples"}

    result = {
        "n": int(n),
        "n_positive": int(np.sum(y_true)),
        "n_negative": int(n - np.sum(y_true)),
        "base_rate": float(np.mean(y_true)),
        "auroc": compute_auroc(y_true, y_score),
        "brier_score": compute_brier(y_true, y_score),
        "log_loss": compute_logloss(y_true, y_score),
    }

    # ECE / MCE at multiple bin counts + reliability data
    for n_bins in ECE_BINS_LIST:
        ece, mce, rel = compute_ece_mce(y_true, y_score, n_bins)
        result[f"ece_{n_bins}"] = ece
        if n_bins == ECE_BINS_LIST[0]:
            # MCE reported once (from finest standard bin count, 10)
            result["mce"] = mce
        result[f"reliability_{n_bins}"] = rel

    # Also compute MCE at 20 bins (finest resolution)
    _, mce_20, _ = compute_ece_mce(y_true, y_score, 20)
    result["mce_20"] = mce_20

    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data(scored_dir: str, smoke_test: bool = False) -> list[dict]:
    """Load all scored JSONL files, return list of dicts."""
    records = []
    for fname in INPUT_FILES:
        path = os.path.join(scored_dir, fname)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))
        if smoke_test and len(records) >= 50:
            break
    if smoke_test:
        records = records[:50]
    return records


def extract_arrays(records: list[dict], method_field: str):
    """Extract (y_true, y_score) arrays, skipping records where field is None."""
    y_list = []
    p_list = []
    for r in records:
        p = r.get(method_field)
        y = r.get("is_correct")
        if p is None or y is None:
            continue
        y_list.append(float(y))
        p_list.append(float(p))
    return np.array(y_list, dtype=np.float64), np.array(p_list, dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-benchmark worker (for multiprocessing)
# ---------------------------------------------------------------------------

def compute_benchmark_metrics(bench_records_tuple: tuple) -> tuple:
    """Worker function: compute metrics for one benchmark slice.

    Args:
        bench_records_tuple: (benchmark_name, records_for_benchmark)

    Returns:
        (benchmark_name, {method: metrics_dict})
    """
    bench, records = bench_records_tuple
    result = {}
    for method_name, field in METHODS.items():
        y, p = extract_arrays(records, field)
        result[method_name] = compute_all_metrics(y, p)
    return bench, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute proper scoring rules for UQ methods")
    parser.add_argument("--scored_dir", default=SCORED_DIR, help="Directory with scored JSONL files")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--smoke_test", action="store_true", help="Run on tiny subset (50 samples)")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers (default: cpu_count)")
    args = parser.parse_args()

    n_workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))

    print(f"Loading data from {args.scored_dir} ...")
    records = load_all_data(args.scored_dir, smoke_test=args.smoke_test)
    print(f"Loaded {len(records)} records" + (" (smoke test)" if args.smoke_test else ""))

    if not records:
        print("ERROR: No records loaded. Exiting.")
        sys.exit(1)

    # ---- Group records ----
    by_model = defaultdict(list)
    by_bench = defaultdict(list)
    for r in records:
        by_model[r["target_model"]].append(r)
        by_bench[r["benchmark"]].append(r)

    results = {
        "metadata": {
            "total_samples": len(records),
            "target_models": sorted(by_model.keys()),
            "benchmarks": sorted(by_bench.keys()),
            "methods": list(METHODS.keys()),
            "ece_bins": ECE_BINS_LIST,
            "smoke_test": args.smoke_test,
        },
        "combined": {},
        "per_model": {},
        "per_benchmark": {},
    }

    # ---- Combined (all data) ----
    print("\nComputing combined metrics ...")
    for method_name, field in METHODS.items():
        y, p = extract_arrays(records, field)
        results["combined"][method_name] = compute_all_metrics(y, p)

    # ---- Per target model ----
    print("Computing per-model metrics ...")
    for model_name, model_records in sorted(by_model.items()):
        results["per_model"][model_name] = {}
        for method_name, field in METHODS.items():
            y, p = extract_arrays(model_records, field)
            results["per_model"][model_name][method_name] = compute_all_metrics(y, p)

    # ---- Per benchmark (parallelized) ----
    bench_items = sorted(by_bench.items())
    print(f"Computing per-benchmark metrics ({len(bench_items)} benchmarks, {n_workers} workers) ...")

    with ProcessPoolExecutor(max_workers=min(n_workers, len(bench_items))) as pool:
        bench_results = pool.map(compute_benchmark_metrics, bench_items)

    for bench, metrics in bench_results:
        results["per_benchmark"][bench] = metrics

    # ---- Save ----
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "scoring_rules.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ---- Print summary table ----
    print_summary(results)


def print_summary(results: dict):
    """Print a formatted summary table to stdout."""
    methods = results["metadata"]["methods"]

    def fmt(v, width=8):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "   N/A".ljust(width)
        return f"{v:.4f}".rjust(width)

    # --- Combined table ---
    print("\n" + "=" * 90)
    print("COMBINED (all target models)")
    print("=" * 90)
    header = f"{'Method':<14} {'N':>6} {'AUROC':>8} {'Brier':>8} {'LogLoss':>8} {'ECE-10':>8} {'ECE-15':>8} {'ECE-20':>8} {'MCE':>8}"
    print(header)
    print("-" * len(header))
    for m in methods:
        d = results["combined"].get(m, {})
        n = d.get("n", 0)
        print(
            f"{m:<14} {n:>6} "
            f"{fmt(d.get('auroc'))} "
            f"{fmt(d.get('brier_score'))} "
            f"{fmt(d.get('log_loss'))} "
            f"{fmt(d.get('ece_10'))} "
            f"{fmt(d.get('ece_15'))} "
            f"{fmt(d.get('ece_20'))} "
            f"{fmt(d.get('mce'))}"
        )

    # --- Per model table ---
    for model_name in sorted(results["per_model"].keys()):
        print(f"\n{'=' * 90}")
        print(f"TARGET MODEL: {model_name}")
        print("=" * 90)
        print(header)
        print("-" * len(header))
        for m in methods:
            d = results["per_model"][model_name].get(m, {})
            n = d.get("n", 0)
            print(
                f"{m:<14} {n:>6} "
                f"{fmt(d.get('auroc'))} "
                f"{fmt(d.get('brier_score'))} "
                f"{fmt(d.get('log_loss'))} "
                f"{fmt(d.get('ece_10'))} "
                f"{fmt(d.get('ece_15'))} "
                f"{fmt(d.get('ece_20'))} "
                f"{fmt(d.get('mce'))}"
            )

    # --- Per benchmark AUROC summary ---
    print(f"\n{'=' * 120}")
    print("PER-BENCHMARK AUROC")
    print("=" * 120)
    bench_header = f"{'Benchmark':<20} {'N':>5}"
    for m in methods:
        bench_header += f" {m:>12}"
    print(bench_header)
    print("-" * len(bench_header))
    for bench in sorted(results["per_benchmark"].keys()):
        bd = results["per_benchmark"][bench]
        # Get N from any method that has it
        n = 0
        for m in methods:
            if bd.get(m, {}).get("n", 0) > 0:
                n = bd[m]["n"]
                break
        row = f"{bench:<20} {n:>5}"
        for m in methods:
            auroc = bd.get(m, {}).get("auroc")
            if auroc is None or (isinstance(auroc, float) and np.isnan(auroc)):
                row += "          N/A"
            else:
                row += f" {auroc:>12.4f}"
        print(row)

    # --- Per benchmark Brier summary ---
    print(f"\n{'=' * 120}")
    print("PER-BENCHMARK BRIER SCORE (lower is better)")
    print("=" * 120)
    print(bench_header)
    print("-" * len(bench_header))
    for bench in sorted(results["per_benchmark"].keys()):
        bd = results["per_benchmark"][bench]
        n = 0
        for m in methods:
            if bd.get(m, {}).get("n", 0) > 0:
                n = bd[m]["n"]
                break
        row = f"{bench:<20} {n:>5}"
        for m in methods:
            brier = bd.get(m, {}).get("brier_score")
            if brier is None or (isinstance(brier, float) and np.isnan(brier)):
                row += "          N/A"
            else:
                row += f" {brier:>12.4f}"
        print(row)

    print()


if __name__ == "__main__":
    main()
