#!/usr/bin/env python3
"""Compute bootstrap confidence intervals for v3 test-only AUROC.

Also computes verbalized baseline comparison and significance tests.

Usage:
    python scripts/bootstrap_ci_v3.py
    python scripts/bootstrap_ci_v3.py --scored_dir data/use_cases/scored_test_only_v3
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load_scored_data(scored_dir):
    """Load scored JSONL files and return combined data."""
    data = {}
    combined = []
    for target in ["gpt5mini", "gpt52", "qwen35"]:
        fpath = Path(scored_dir) / f"{target}_scored.jsonl"
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        samples = [json.loads(line) for line in open(fpath)]
        data[target] = samples
        combined.extend(samples)
        print(f"  {target}: {len(samples)} samples")
    print(f"  Combined: {len(combined)} samples")
    return data, combined


def bootstrap_auroc(labels, scores, n_bootstrap=10000, seed=42):
    """Compute bootstrap CI for AUROC."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    labels = np.array(labels)
    scores = np.array(scores)

    base_auroc = roc_auc_score(labels, scores)
    boot_aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b = labels[idx]
        s_b = scores[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_aurocs.append(roc_auc_score(y_b, s_b))

    boot_aurocs = np.array(boot_aurocs)
    ci_lower = np.percentile(boot_aurocs, 2.5)
    ci_upper = np.percentile(boot_aurocs, 97.5)
    return {
        "auroc": float(base_auroc),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "std": float(np.std(boot_aurocs)),
        "n_bootstrap": len(boot_aurocs),
    }


def permutation_test(labels, scores1, scores2, n_perm=10000, seed=42):
    """Permutation test for difference in AUROC between two scoring methods."""
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    auroc1 = roc_auc_score(labels, scores1)
    auroc2 = roc_auc_score(labels, scores2)
    observed_diff = auroc1 - auroc2

    count = 0
    for _ in range(n_perm):
        swap = rng.rand(len(labels)) > 0.5
        s1_perm = np.where(swap, scores2, scores1)
        s2_perm = np.where(swap, scores1, scores2)
        diff = roc_auc_score(labels, s1_perm) - roc_auc_score(labels, s2_perm)
        if diff >= observed_diff:
            count += 1

    p_value = (count + 1) / (n_perm + 1)
    return {
        "auroc_1": float(auroc1),
        "auroc_2": float(auroc2),
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "n_permutations": n_perm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scored_dir",
        default="data/use_cases/scored_test_only_v3",
    )
    parser.add_argument(
        "--output",
        default="data/use_cases/results_test_only_v3/bootstrap_ci_v3.json",
    )
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    args = parser.parse_args()

    print("=== Bootstrap CI for v3 test-only AUROC ===\n")

    # Load data
    print("Loading scored data...")
    per_target, combined = load_scored_data(args.scored_dir)

    if not combined:
        print("ERROR: No data loaded")
        sys.exit(1)

    # Combined AUROC + CI
    labels = [int(s["is_correct"]) for s in combined]
    cal_scores = [float(s["p_correct"]) for s in combined]

    print(f"\nComputing bootstrap CIs (n={args.n_bootstrap})...")
    results = {}

    # Overall
    results["combined"] = bootstrap_auroc(labels, cal_scores, args.n_bootstrap)
    print(f"  Combined AUROC: {results['combined']['auroc']:.4f} "
          f"[{results['combined']['ci_lower']:.4f}, {results['combined']['ci_upper']:.4f}]")

    # Per-target
    results["per_target"] = {}
    for target, samples in per_target.items():
        t_labels = [int(s["is_correct"]) for s in samples]
        t_scores = [float(s["p_correct"]) for s in samples]
        results["per_target"][target] = bootstrap_auroc(t_labels, t_scores, args.n_bootstrap)
        r = results["per_target"][target]
        print(f"  {target}: {r['auroc']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")

    # Per-benchmark
    print("\nPer-benchmark AUROC:")
    bench_data = {}
    for s in combined:
        b = s.get("benchmark", "unknown")
        bench_data.setdefault(b, {"labels": [], "scores": []})
        bench_data[b]["labels"].append(int(s["is_correct"]))
        bench_data[b]["scores"].append(float(s["p_correct"]))

    results["per_benchmark"] = {}
    for bench in sorted(bench_data.keys()):
        bd = bench_data[bench]
        n = len(bd["labels"])
        n_pos = sum(bd["labels"])
        if n_pos == 0 or n_pos == n:
            print(f"  {bench}: N={n}, skipping (all same class)")
            continue
        r = bootstrap_auroc(bd["labels"], bd["scores"], min(args.n_bootstrap, 5000))
        results["per_benchmark"][bench] = r
        results["per_benchmark"][bench]["n_samples"] = n
        results["per_benchmark"][bench]["accuracy"] = n_pos / n
        print(f"  {bench}: {r['auroc']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] (N={n})")

    # Verbalized baseline comparison (if available)
    if any("verbalized_confidence" in s for s in combined):
        verb_scores = []
        cal_scores_filtered = []
        labels_filtered = []
        for s in combined:
            vc = s.get("verbalized_confidence")
            if vc is not None:
                verb_scores.append(float(vc))
                cal_scores_filtered.append(float(s["p_correct"]))
                labels_filtered.append(int(s["is_correct"]))

        if len(verb_scores) > 50:
            results["verbalized_baseline"] = bootstrap_auroc(
                labels_filtered, verb_scores, args.n_bootstrap
            )
            results["significance_vs_verbalized"] = permutation_test(
                labels_filtered, cal_scores_filtered, verb_scores
            )
            print(f"\n  Verbalized AUROC: {results['verbalized_baseline']['auroc']:.4f}")
            print(f"  p-value (cal > verb): {results['significance_vs_verbalized']['p_value']:.6f}")

    # Summary stats
    bench_aurocs = [v["auroc"] for v in results["per_benchmark"].values()]
    results["summary"] = {
        "n_test_samples": len(combined),
        "n_benchmarks": len(results["per_benchmark"]),
        "per_benchmark_mean_auroc": float(np.mean(bench_aurocs)),
        "per_benchmark_std_auroc": float(np.std(bench_aurocs)),
        "per_benchmark_cv": float(np.std(bench_aurocs) / np.mean(bench_aurocs)) if np.mean(bench_aurocs) > 0 else 0,
        "per_benchmark_min": float(np.min(bench_aurocs)),
        "per_benchmark_max": float(np.max(bench_aurocs)),
    }
    print(f"\n  Per-benchmark mean: {results['summary']['per_benchmark_mean_auroc']:.3f} "
          f"± {results['summary']['per_benchmark_std_auroc']:.3f}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
