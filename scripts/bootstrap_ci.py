#!/usr/bin/env python3
"""
Compute bootstrap confidence intervals for all UQ baselines.

For each UQ signal (calibrator, verbalized, Platt-scaled, length, combined, zero-shot),
computes AUROC with 95% confidence intervals using bootstrap resampling.

This produces a summary table that can be included in the report.

Usage:
    python scripts/bootstrap_ci.py --scored_dir data/use_cases/scored_unified
    python scripts/bootstrap_ci.py --scored_dir data/use_cases/scored_unified --smoke_test
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def load_scored(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def bootstrap_auroc(y_true, y_score, n_bootstrap=2000, ci=0.95, seed=42):
    """Compute AUROC with bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(y_true)

    if n < 10 or len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": n}

    # Point estimate
    auroc = roc_auc_score(y_true, y_score)

    # Bootstrap
    aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_boot = y_true[idx]
        s_boot = y_score[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        aurocs.append(roc_auc_score(y_boot, s_boot))

    aurocs = np.array(aurocs)
    alpha = (1 - ci) / 2
    ci_low = np.percentile(aurocs, 100 * alpha)
    ci_high = np.percentile(aurocs, 100 * (1 - alpha))

    return {
        "auroc": float(auroc),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n": n,
        "n_bootstrap": len(aurocs),
    }


def compute_one_baseline(args_tuple):
    """Worker function for parallel bootstrap computation."""
    name, y, scores = args_tuple
    result = bootstrap_auroc(y, scores)
    result["method"] = name
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_unified")
    parser.add_argument("--output", default="data/use_cases/results_unified/bootstrap_ci.json")
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    targets = ["gpt5mini", "gpt52", "qwen35"]

    # Methods to evaluate (score field name → display name)
    methods = [
        ("p_correct", "Calibrator (ours)"),
        ("verbalized_confidence", "Verbalized (raw)"),
        ("p_platt_verbalized", "Verbalized (Platt)"),
        ("p_isotonic_verbalized", "Verbalized (Isotonic)"),
        ("p_length_baseline", "Response length"),
        ("p_combined_baseline", "Combined (verb+len)"),
        ("p_zeroshot", "Zero-shot base model"),
    ]

    all_results = {}

    for target in targets:
        path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if not path.exists():
            print(f"Skipping {target}: not found")
            continue

        data = load_scored(path)
        if args.smoke_test:
            data = data[:200]

        print(f"\n{'='*60}")
        print(f"Bootstrap CI for {target} ({len(data)} samples)")
        print(f"{'='*60}")

        target_results = []

        # Prepare all tasks
        tasks = []
        for field, label in methods:
            valid = [d for d in data
                     if d.get(field) is not None
                     and d.get("is_correct") is not None]
            if len(valid) < 20:
                print(f"  {label:30s} — skipped (only {len(valid)} samples)")
                continue

            y = np.array([d["is_correct"] for d in valid])
            scores = np.array([d[field] for d in valid])
            tasks.append((label, y, scores))

        # Run bootstrap in parallel
        n_workers = min(len(tasks), multiprocessing.cpu_count())
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(compute_one_baseline, tasks))

        # Sort by AUROC descending
        results.sort(key=lambda x: x["auroc"], reverse=True)

        # Print table
        print(f"\n  {'Method':<30s} {'AUROC':>8s} {'95% CI':>16s} {'N':>6s}")
        print(f"  {'—'*30} {'—'*8} {'—'*16} {'—'*6}")
        for r in results:
            ci_str = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
            marker = " ***" if r["method"] == "Calibrator (ours)" else ""
            print(f"  {r['method']:<30s} {r['auroc']:>8.4f} {ci_str:>16s} {r['n']:>6d}{marker}")

        all_results[target] = results

    # Also compute combined (all targets)
    print(f"\n{'='*60}")
    print(f"Combined (all targets)")
    print(f"{'='*60}")
    combined_data = []
    for target in targets:
        path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if path.exists():
            combined_data.extend(load_scored(path))

    if args.smoke_test:
        combined_data = combined_data[:500]

    tasks = []
    for field, label in methods:
        valid = [d for d in combined_data
                 if d.get(field) is not None
                 and d.get("is_correct") is not None]
        if len(valid) < 20:
            continue
        y = np.array([d["is_correct"] for d in valid])
        scores = np.array([d[field] for d in valid])
        tasks.append((label, y, scores))

    n_workers = min(len(tasks), multiprocessing.cpu_count())
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(compute_one_baseline, tasks))

    results.sort(key=lambda x: x["auroc"], reverse=True)

    print(f"\n  {'Method':<30s} {'AUROC':>8s} {'95% CI':>16s} {'N':>6s}")
    print(f"  {'—'*30} {'—'*8} {'—'*16} {'—'*6}")
    for r in results:
        ci_str = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        marker = " ***" if r["method"] == "Calibrator (ours)" else ""
        print(f"  {r['method']:<30s} {r['auroc']:>8.4f} {ci_str:>16s} {r['n']:>6d}{marker}")

    all_results["combined"] = results

    # Save
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
