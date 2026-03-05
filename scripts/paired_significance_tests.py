#!/usr/bin/env python3
"""Run paired significance tests between calibrator and baselines.

Implements:
1. Paired permutation test for AUROC differences
2. McNemar's test for classification agreement
3. DeLong test for AUROC comparison (asymptotic)
4. Per-target-model significance

Usage:
    python scripts/paired_significance_tests.py --scored_dir data/use_cases/scored_test_only
"""
import argparse
import json
import os
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
from concurrent.futures import ProcessPoolExecutor


def load_scored(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def paired_permutation_auroc(labels, preds_a, preds_b, n_permutations=10000, seed=42):
    """Paired permutation test for AUROC difference.

    H0: AUROC(A) = AUROC(B)
    Test statistic: AUROC(A) - AUROC(B)
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    labels = np.array(labels)
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)

    observed_diff = roc_auc_score(labels, preds_a) - roc_auc_score(labels, preds_b)

    # Under H0, we can swap predictions for each sample
    perm_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        swap = rng.random(n) > 0.5
        preds_perm_a = np.where(swap, preds_b, preds_a)
        preds_perm_b = np.where(swap, preds_a, preds_b)
        try:
            perm_diffs[i] = roc_auc_score(labels, preds_perm_a) - roc_auc_score(labels, preds_perm_b)
        except ValueError:
            perm_diffs[i] = 0.0

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "significant_001": p_value < 0.001,
        "significant_005": p_value < 0.005,
        "significant_01": p_value < 0.01,
    }


def mcnemar_test(labels, preds_a, preds_b, threshold=0.5):
    """McNemar's test: are the two methods making different types of errors?

    Compare binary predictions (threshold at 0.5).
    """
    labels = np.array(labels)
    binary_a = np.array(preds_a) >= threshold
    binary_b = np.array(preds_b) >= threshold

    correct_a = (binary_a == labels)
    correct_b = (binary_b == labels)

    # Contingency: a correct & b wrong, a wrong & b correct
    b_only = np.sum(~correct_a & correct_b)  # b correct, a wrong
    a_only = np.sum(correct_a & ~correct_b)   # a correct, b wrong
    both_correct = np.sum(correct_a & correct_b)
    both_wrong = np.sum(~correct_a & ~correct_b)

    # McNemar's test statistic (with continuity correction)
    n_discordant = a_only + b_only
    if n_discordant == 0:
        return {"chi2": 0.0, "p_value": 1.0, "a_only": int(a_only), "b_only": int(b_only)}

    chi2 = (abs(a_only - b_only) - 1) ** 2 / (a_only + b_only)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "a_only_correct": int(a_only),
        "b_only_correct": int(b_only),
        "both_correct": int(both_correct),
        "both_wrong": int(both_wrong),
        "acc_a": float(np.mean(correct_a)),
        "acc_b": float(np.mean(correct_b)),
    }


def delong_auroc_variance(labels, preds):
    """Compute DeLong variance estimate for AUROC."""
    labels = np.array(labels)
    preds = np.array(preds)
    pos = preds[labels == 1]
    neg = preds[labels == 0]
    m = len(pos)
    n = len(neg)

    # Structural components
    v10 = np.zeros(m)
    v01 = np.zeros(n)
    for i in range(m):
        v10[i] = np.mean(pos[i] > neg) + 0.5 * np.mean(pos[i] == neg)
    for j in range(n):
        v01[j] = np.mean(pos > neg[j]) + 0.5 * np.mean(pos == neg[j])

    s10 = np.var(v10, ddof=1) if m > 1 else 0
    s01 = np.var(v01, ddof=1) if n > 1 else 0

    var = s10 / m + s01 / n
    return var


def delong_test(labels, preds_a, preds_b):
    """DeLong test for comparing two AUROCs on the same dataset."""
    labels = np.array(labels)
    auroc_a = roc_auc_score(labels, preds_a)
    auroc_b = roc_auc_score(labels, preds_b)
    diff = auroc_a - auroc_b

    var_a = delong_auroc_variance(labels, preds_a)
    var_b = delong_auroc_variance(labels, preds_b)

    # Approximate: assume independent (conservative)
    se_diff = np.sqrt(var_a + var_b)

    if se_diff < 1e-10:
        return {"z": float('inf'), "p_value": 0.0, "diff": float(diff), "se": float(se_diff)}

    z = diff / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "auroc_a": float(auroc_a),
        "auroc_b": float(auroc_b),
        "diff": float(diff),
        "se": float(se_diff),
        "z": float(z),
        "p_value": float(p_value),
        "significant_001": p_value < 0.001,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only")
    parser.add_argument("--output", default="data/use_cases/results_test_only_v2/significance_tests.json")
    parser.add_argument("--n_permutations", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    baseline_keys = {
        "verbalized_confidence": "Verbalized (raw)",
        "p_platt_verbalized": "Verbalized (Platt)",
        "p_isotonic_verbalized": "Verbalized (Isotonic)",
        "p_length_baseline": "Response length",
        "p_combined_baseline": "Combined (verb+len)",
        "p_correct_zeroshot": "Zero-shot base model",
    }

    targets = ["gpt5mini", "gpt52", "qwen35"]
    all_results = {}

    # Load all data
    combined_data = []
    for target in targets:
        path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if not path.exists():
            print(f"Skipping {target}")
            continue
        data = load_scored(path)
        combined_data.extend(data)

    print(f"Total samples: {len(combined_data)}")

    # Filter to valid calibrator scores
    valid = [d for d in combined_data
             if d.get("p_correct") is not None and d.get("is_correct") is not None]
    print(f"Valid calibrator samples: {len(valid)}")

    labels = [d["is_correct"] for d in valid]
    cal_preds = [d["p_correct"] for d in valid]

    print(f"\nCalibrator AUROC: {roc_auc_score(labels, cal_preds):.4f}")

    # Run tests for each baseline
    print(f"\n{'='*90}")
    print(f"{'Baseline':<25} {'AUROC':>7} {'Diff':>7} {'Perm p':>10} {'DeLong p':>10} {'McNemar p':>10}")
    print(f"{'='*90}")

    for key, name in baseline_keys.items():
        baseline_valid = [(d["is_correct"], d["p_correct"], d[key])
                          for d in valid if d.get(key) is not None]

        if len(baseline_valid) < 50:
            print(f"{name:<25} Too few samples ({len(baseline_valid)})")
            continue

        bl, cp, bp = zip(*baseline_valid)
        bl = list(bl)
        cp = list(cp)
        bp = list(bp)

        if len(set(bl)) < 2:
            continue

        baseline_auroc = roc_auc_score(bl, bp)

        # Paired permutation test
        perm = paired_permutation_auroc(bl, cp, bp, n_permutations=args.n_permutations)

        # DeLong test
        delong = delong_test(bl, cp, bp)

        # McNemar's test
        mcnemar = mcnemar_test(bl, cp, bp)

        print(f"{name:<25} {baseline_auroc:>7.4f} {perm['observed_diff']:>+7.4f} "
              f"{perm['p_value']:>10.6f} {delong['p_value']:>10.6f} {mcnemar['p_value']:>10.6f}")

        sig_star = "***" if perm['p_value'] < 0.001 else "**" if perm['p_value'] < 0.01 else "*" if perm['p_value'] < 0.05 else "ns"
        print(f"{'':>25} [{sig_star}]  Cal correct only: {mcnemar.get('a_only_correct', 0)}, "
              f"Baseline correct only: {mcnemar.get('b_only_correct', 0)}")

        all_results[key] = {
            "name": name,
            "n_samples": len(bl),
            "baseline_auroc": float(baseline_auroc),
            "calibrator_auroc": float(roc_auc_score(bl, cp)),
            "permutation_test": perm,
            "delong_test": delong,
            "mcnemar_test": mcnemar,
        }

    # Per-target significance
    print(f"\n{'='*90}")
    print(f"Per-target model significance (calibrator vs best baseline)")
    print(f"{'='*90}")

    per_target_results = {}
    for target in targets:
        target_data = [d for d in valid if d.get("target_model") == target]
        if len(target_data) < 50:
            continue

        tl = [d["is_correct"] for d in target_data]
        tp = [d["p_correct"] for d in target_data]
        if len(set(tl)) < 2:
            continue

        cal_auroc = roc_auc_score(tl, tp)
        print(f"\n  {target}: Calibrator AUROC = {cal_auroc:.4f} (n={len(target_data)})")

        best_baseline_auroc = 0
        best_baseline_name = ""
        target_tests = {}

        for key, name in baseline_keys.items():
            bl_data = [(d["is_correct"], d["p_correct"], d[key])
                       for d in target_data if d.get(key) is not None]
            if len(bl_data) < 30 or len(set([x[0] for x in bl_data])) < 2:
                continue

            bl, cp, bp = zip(*bl_data)
            bl, cp, bp = list(bl), list(cp), list(bp)
            bl_auroc = roc_auc_score(bl, bp)

            perm = paired_permutation_auroc(bl, cp, bp, n_permutations=args.n_permutations)
            sig = "***" if perm['p_value'] < 0.001 else "**" if perm['p_value'] < 0.01 else "*" if perm['p_value'] < 0.05 else "ns"
            print(f"    vs {name:<25}: diff={perm['observed_diff']:+.4f} p={perm['p_value']:.6f} [{sig}]")

            if bl_auroc > best_baseline_auroc:
                best_baseline_auroc = bl_auroc
                best_baseline_name = name

            target_tests[key] = {"name": name, "auroc": float(bl_auroc), "permutation_test": perm}

        per_target_results[target] = {
            "calibrator_auroc": float(cal_auroc),
            "n_samples": len(target_data),
            "best_baseline": best_baseline_name,
            "best_baseline_auroc": float(best_baseline_auroc),
            "tests": target_tests,
        }

    all_results["per_target"] = per_target_results

    # Save (convert numpy types for JSON)
    def numpy_to_python(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=numpy_to_python)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
