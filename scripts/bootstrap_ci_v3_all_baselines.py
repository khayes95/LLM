#!/usr/bin/env python3
"""Compute bootstrap confidence intervals for v3 test-only AUROC — ALL baselines.

Computes CIs for: calibrator, verbalized (raw), verbalized (isotonic),
response length, combined (verb+len), and random baseline.
Also computes pairwise significance tests (calibrator vs each baseline).

Usage:
    python scripts/bootstrap_ci_v3_all_baselines.py
    python scripts/bootstrap_ci_v3_all_baselines.py --scored_dir data/use_cases/scored_test_only_v3
    python scripts/bootstrap_ci_v3_all_baselines.py --smoke_test
"""
import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
from sklearn.metrics import roc_auc_score


# Methods: (field_name, display_name)
METHODS = [
    ("p_correct", "Calibrator (ours)"),
    ("verbalized_confidence", "Verbalized (raw)"),
    ("p_isotonic_verbalized", "Verbalized (Isotonic)"),
    ("p_length_baseline", "Response length"),
    ("p_combined_baseline", "Combined (verb+len)"),
    ("random", "Random"),
]


def load_scored(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_all_data(scored_dir):
    """Load scored JSONL files and return per-target and combined data."""
    per_target = {}
    combined = []
    for target in ["gpt5mini", "gpt52", "qwen35"]:
        fpath = Path(scored_dir) / f"{target}_scored.jsonl"
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        samples = load_scored(fpath)
        per_target[target] = samples
        combined.extend(samples)
        print(f"  {target}: {len(samples)} samples")
    print(f"  Combined: {len(combined)} samples")
    return per_target, combined


def bootstrap_auroc(y_true, y_score, n_bootstrap=10000, seed=42):
    """Compute AUROC with bootstrap 95% CI."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if n < 10 or len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "ci_lower": float("nan"),
                "ci_upper": float("nan"), "std": float("nan"),
                "n": n, "n_bootstrap": 0}

    base_auroc = roc_auc_score(y_true, y_score)
    boot_aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        s_b = y_score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_aurocs.append(roc_auc_score(y_b, s_b))

    boot_aurocs = np.array(boot_aurocs)
    return {
        "auroc": float(base_auroc),
        "ci_lower": float(np.percentile(boot_aurocs, 2.5)),
        "ci_upper": float(np.percentile(boot_aurocs, 97.5)),
        "std": float(np.std(boot_aurocs)),
        "n": n,
        "n_bootstrap": len(boot_aurocs),
    }


def permutation_test(labels, scores1, scores2, n_perm=10000, seed=42):
    """Permutation test for AUROC difference (scores1 > scores2)."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    scores1 = np.asarray(scores1)
    scores2 = np.asarray(scores2)

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

    return {
        "auroc_method": float(auroc1),
        "auroc_baseline": float(auroc2),
        "observed_diff": float(observed_diff),
        "p_value": float((count + 1) / (n_perm + 1)),
        "n_permutations": n_perm,
    }


def _worker_bootstrap(args_tuple):
    """Worker for parallel bootstrap."""
    method_name, y, scores, n_bootstrap = args_tuple
    result = bootstrap_auroc(y, scores, n_bootstrap)
    result["method"] = method_name
    return result


def _worker_permutation(args_tuple):
    """Worker for parallel permutation tests."""
    baseline_name, labels, cal_scores, base_scores, n_perm = args_tuple
    result = permutation_test(labels, cal_scores, base_scores, n_perm)
    result["baseline"] = baseline_name
    return result


def get_scores_for_method(data, field):
    """Extract (indices, labels, scores) for a method, filtering nulls.

    Returns indices into the original data list so we can align methods for
    pairwise significance tests.
    """
    if field == "random":
        rng = np.random.RandomState(999)
        indices = [i for i, d in enumerate(data) if d.get("is_correct") is not None]
        labels = [int(data[i]["is_correct"]) for i in indices]
        scores = rng.rand(len(labels)).tolist()
        return indices, labels, scores

    indices = []
    labels = []
    scores = []
    for i, d in enumerate(data):
        if d.get(field) is not None and d.get("is_correct") is not None:
            indices.append(i)
            labels.append(int(d["is_correct"]))
            scores.append(float(d[field]))
    return indices, labels, scores


def compute_all_baselines(data, n_bootstrap, n_perm, label=""):
    """Compute bootstrap CIs and significance tests for all methods on a dataset."""
    print(f"\n{'='*60}")
    print(f"  {label} ({len(data)} samples)")
    print(f"{'='*60}")

    # Prepare bootstrap tasks
    bootstrap_tasks = []
    method_data = {}  # store (indices, labels, scores) for significance tests
    for field, display_name in METHODS:
        indices, labels, scores = get_scores_for_method(data, field)
        if len(labels) < 20:
            print(f"  {display_name:30s} — skipped (only {len(labels)} valid samples)")
            continue
        if len(np.unique(labels)) < 2:
            print(f"  {display_name:30s} — skipped (single class)")
            continue
        bootstrap_tasks.append((display_name, np.array(labels), np.array(scores), n_bootstrap))
        method_data[display_name] = (set(indices), dict(zip(indices, labels)), dict(zip(indices, scores)))

    # Run bootstrap in parallel
    n_workers = min(len(bootstrap_tasks), multiprocessing.cpu_count())
    with ProcessPoolExecutor(max_workers=max(1, n_workers)) as executor:
        ci_results = list(executor.map(_worker_bootstrap, bootstrap_tasks))

    # Sort by AUROC descending
    ci_results.sort(key=lambda x: x.get("auroc", 0), reverse=True)

    # Print table
    print(f"\n  {'Method':<30s} {'AUROC':>8s} {'95% CI':>20s} {'N':>6s}")
    print(f"  {'—'*30} {'—'*8} {'—'*20} {'—'*6}")
    for r in ci_results:
        ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
        marker = " ***" if r["method"] == "Calibrator (ours)" else ""
        print(f"  {r['method']:<30s} {r['auroc']:>8.4f} {ci_str:>20s} {r['n']:>6d}{marker}")

    # Significance tests: calibrator vs each baseline
    sig_results = {}
    cal_key = "Calibrator (ours)"
    if cal_key in method_data:
        cal_idx_set, cal_label_map, cal_score_map = method_data[cal_key]
        perm_tasks = []
        for display_name in method_data:
            if display_name == cal_key:
                continue
            base_idx_set, base_label_map, base_score_map = method_data[display_name]
            # Intersect indices so both methods have scores for the same samples
            common = sorted(cal_idx_set & base_idx_set)
            if len(common) < 20:
                continue
            shared_labels = np.array([cal_label_map[i] for i in common])
            shared_cal = np.array([cal_score_map[i] for i in common])
            shared_base = np.array([base_score_map[i] for i in common])
            perm_tasks.append((display_name, shared_labels, shared_cal, shared_base, n_perm))

        if perm_tasks:
            with ProcessPoolExecutor(max_workers=max(1, min(len(perm_tasks), multiprocessing.cpu_count()))) as executor:
                perm_results = list(executor.map(_worker_permutation, perm_tasks))

            print(f"\n  Significance (Calibrator vs baseline, permutation test):")
            for pr in perm_results:
                sig_results[pr["baseline"]] = pr
                star = "***" if pr["p_value"] < 0.001 else "**" if pr["p_value"] < 0.01 else "*" if pr["p_value"] < 0.05 else "ns"
                print(f"    vs {pr['baseline']:<28s} diff={pr['observed_diff']:+.4f}  p={pr['p_value']:.6f} {star}")

    return ci_results, sig_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v3")
    parser.add_argument("--output", default="data/use_cases/results_test_only_v3/bootstrap_ci_v3_all_baselines.json")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_bootstrap = 200
        args.n_perm = 200

    print("=== Bootstrap CI for v3 test-only — ALL BASELINES ===\n")
    print(f"  n_bootstrap={args.n_bootstrap}, n_perm={args.n_perm}")

    # Load data
    print("\nLoading scored data...")
    per_target, combined = load_all_data(args.scored_dir)

    if not combined:
        print("ERROR: No data loaded")
        sys.exit(1)

    output = {}

    # Combined
    ci_results, sig_results = compute_all_baselines(
        combined, args.n_bootstrap, args.n_perm, "Combined (all targets)"
    )
    output["combined"] = {
        "baselines": {r["method"]: r for r in ci_results},
        "significance_vs_calibrator": sig_results,
    }

    # Per-target
    output["per_target"] = {}
    for target, samples in per_target.items():
        ci_results, sig_results = compute_all_baselines(
            samples, args.n_bootstrap, args.n_perm, f"Target: {target}"
        )
        output["per_target"][target] = {
            "baselines": {r["method"]: r for r in ci_results},
            "significance_vs_calibrator": sig_results,
        }

    # Per-benchmark (combined data, calibrator + all baselines)
    print(f"\n{'='*60}")
    print(f"  Per-benchmark breakdown (all baselines)")
    print(f"{'='*60}")
    bench_data = {}
    for s in combined:
        b = s.get("benchmark", "unknown")
        bench_data.setdefault(b, []).append(s)

    output["per_benchmark"] = {}
    for bench in sorted(bench_data.keys()):
        bdata = bench_data[bench]
        n = len(bdata)
        n_pos = sum(int(d["is_correct"]) for d in bdata if d.get("is_correct") is not None)
        if n_pos == 0 or n_pos == n or n < 20:
            print(f"  {bench}: N={n}, skipping (degenerate)")
            continue

        # Compute all baselines for this benchmark
        bench_results = {}
        for field, display_name in METHODS:
            _, labels, scores = get_scores_for_method(bdata, field)
            if len(labels) < 10 or len(np.unique(labels)) < 2:
                continue
            r = bootstrap_auroc(
                np.array(labels), np.array(scores),
                min(args.n_bootstrap, 5000)
            )
            r["method"] = display_name
            r["n_samples"] = len(labels)
            r["accuracy"] = n_pos / n
            bench_results[display_name] = r

        if bench_results:
            cal_auroc = bench_results.get("Calibrator (ours)", {}).get("auroc", float("nan"))
            print(f"  {bench}: Cal={cal_auroc:.3f}, N={n}, {len(bench_results)} methods")
            output["per_benchmark"][bench] = bench_results

    # Summary
    cal_aurocs = []
    for bench, methods_dict in output["per_benchmark"].items():
        if "Calibrator (ours)" in methods_dict:
            cal_aurocs.append(methods_dict["Calibrator (ours)"]["auroc"])

    output["summary"] = {
        "n_test_samples": len(combined),
        "n_benchmarks": len(output["per_benchmark"]),
        "methods_evaluated": [name for _, name in METHODS],
        "calibrator_per_benchmark_mean": float(np.mean(cal_aurocs)) if cal_aurocs else None,
        "calibrator_per_benchmark_std": float(np.std(cal_aurocs)) if cal_aurocs else None,
    }

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
