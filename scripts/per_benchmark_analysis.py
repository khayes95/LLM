#!/usr/bin/env python3
"""Per-benchmark AUROC analysis with significance tests for UQ calibrator paper.

Computes:
1. Per-benchmark AUROC breakdown (calibrator, verbalized, length baselines)
2. Paired significance tests (McNemar, DeLong approximation, permutation)
3. Summary statistics with bootstrap confidence intervals
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scored_data(scored_dir: str) -> list[dict]:
    """Load all scored JSONL files from the directory."""
    records = []
    for fname in sorted(Path(scored_dir).glob("*_scored.jsonl")):
        with open(fname) as f:
            for line in f:
                d = json.loads(line)
                records.append(d)
    return records


def parse_record(d: dict) -> dict:
    """Parse string fields into proper types, handling None gracefully."""
    # is_correct
    ic = d.get("is_correct")
    if isinstance(ic, str):
        ic = int(ic)
    # p_correct
    pc = d.get("p_correct")
    if isinstance(pc, str):
        pc = None if pc == "None" else float(pc)
    # verbalized_confidence
    vc = d.get("verbalized_confidence")
    if isinstance(vc, str):
        vc = None if vc == "None" else float(vc)
    # output_tokens
    ot = d.get("output_tokens")
    if isinstance(ot, str):
        ot = 0 if ot == "None" else int(ot)
    if ot is None:
        ot = 0

    return {
        "id": d["id"],
        "benchmark": d["benchmark"],
        "target_model": d["target_model"],
        "is_correct": ic,
        "p_correct": pc,
        "verbalized_confidence": vc,
        "output_tokens": ot,
    }


def safe_auroc(y_true, y_score):
    """Compute AUROC, returning None if undefined (all one class or constant score)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return None
    if len(np.unique(y_score)) < 2:
        return None
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# DeLong test (variance of AUROC difference)
# ---------------------------------------------------------------------------

def _compute_placement_values(y_true, y_score):
    """Compute structural components for DeLong's test.

    Returns V10 (placement values for positives) and V01 (for negatives).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)
    if m == 0 or n == 0:
        return None, None

    pos_scores = y_score[pos_idx]
    neg_scores = y_score[neg_idx]

    # V10[i] = fraction of negatives with score < pos_scores[i]
    #        + 0.5 * fraction of negatives with score == pos_scores[i]
    V10 = np.zeros(m)
    for i, ps in enumerate(pos_scores):
        V10[i] = (np.sum(neg_scores < ps) + 0.5 * np.sum(neg_scores == ps)) / n

    # V01[j] = fraction of positives with score > neg_scores[j]
    #        + 0.5 * fraction of positives with score == neg_scores[j]
    V01 = np.zeros(n)
    for j, ns in enumerate(neg_scores):
        V01[j] = (np.sum(pos_scores > ns) + 0.5 * np.sum(pos_scores == ns)) / m

    return V10, V01


def delong_test(y_true, scores_a, scores_b):
    """DeLong's test for comparing two AUROCs on the same dataset.

    Returns (z_stat, p_value).  Two-sided test.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    V10_a, V01_a = _compute_placement_values(y_true, scores_a)
    V10_b, V01_b = _compute_placement_values(y_true, scores_b)
    if V10_a is None or V10_b is None:
        return np.nan, np.nan

    m = len(V10_a)
    n = len(V01_a)

    auc_a = np.mean(V10_a)
    auc_b = np.mean(V10_b)

    # Covariance matrix of (AUC_a, AUC_b)
    S10 = np.cov(V10_a, V10_b)  # 2x2
    S01 = np.cov(V01_a, V01_b)  # 2x2

    # Variance of difference
    S = S10 / m + S01 / n  # 2x2
    # Var(AUC_a - AUC_b) = S[0,0] + S[1,1] - 2*S[0,1]
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]

    if var_diff <= 0:
        return np.nan, np.nan

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * stats.norm.sf(abs(z))
    return float(z), float(p)


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(y_true, scores_a, scores_b, threshold=0.5):
    """McNemar's test comparing two classifiers (thresholded at 0.5).

    Returns (chi2, p_value).
    """
    y_true = np.asarray(y_true, dtype=int)
    pred_a = (np.asarray(scores_a) >= threshold).astype(int)
    pred_b = (np.asarray(scores_b) >= threshold).astype(int)
    correct_a = (pred_a == y_true).astype(int)
    correct_b = (pred_b == y_true).astype(int)

    # b = A correct, B wrong; c = A wrong, B correct
    b = np.sum((correct_a == 1) & (correct_b == 0))
    c = np.sum((correct_a == 0) & (correct_b == 1))

    if b + c == 0:
        return 0.0, 1.0

    # With continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = stats.chi2.sf(chi2, df=1)
    return float(chi2), float(p)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test_auroc(y_true, scores_a, scores_b, n_perm=10000, rng_seed=42):
    """Paired permutation test for AUROC difference.

    Shuffles which method's score is used for each sample.
    Returns (observed_diff, p_value).
    """
    y_true = np.asarray(y_true, dtype=int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    auc_a = safe_auroc(y_true, scores_a)
    auc_b = safe_auroc(y_true, scores_b)
    if auc_a is None or auc_b is None:
        return np.nan, np.nan

    observed_diff = auc_a - auc_b
    rng = np.random.RandomState(rng_seed)
    n = len(y_true)
    count_extreme = 0

    for _ in range(n_perm):
        swap = rng.randint(0, 2, size=n).astype(bool)
        perm_a = np.where(swap, scores_b, scores_a)
        perm_b = np.where(swap, scores_a, scores_b)
        auc_pa = safe_auroc(y_true, perm_a)
        auc_pb = safe_auroc(y_true, perm_b)
        if auc_pa is None or auc_pb is None:
            continue
        if abs(auc_pa - auc_pb) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_perm + 1)  # +1 for observed
    return float(observed_diff), float(p_value)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_auroc_ci(y_true, y_score, n_boot=1000, alpha=0.05, rng_seed=42):
    """Bootstrap 95% CI for AUROC."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.RandomState(rng_seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        a = safe_auroc(y_true[idx], y_score[idx])
        if a is not None:
            aucs.append(a)
    if len(aucs) == 0:
        return np.nan, np.nan, np.nan
    aucs = np.array(aucs)
    lo = np.percentile(aucs, 100 * alpha / 2)
    hi = np.percentile(aucs, 100 * (1 - alpha / 2))
    return float(np.mean(aucs)), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(scored_dir: str, output_dir: str, fig_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Load data
    raw = load_scored_data(scored_dir)
    records = [parse_record(r) for r in raw]
    print(f"Loaded {len(records)} records from {scored_dir}")

    # -----------------------------------------------------------------------
    # 1. Per-benchmark AUROC table
    # -----------------------------------------------------------------------
    # Group by (benchmark, target_model)
    groups = defaultdict(list)
    for r in records:
        groups[(r["benchmark"], r["target_model"])].append(r)

    per_bm_rows = []
    for (bm, model), recs in sorted(groups.items()):
        n = len(recs)
        if n < 20:
            print(f"  Skipping ({bm}, {model}): only {n} samples")
            continue

        y_true = [r["is_correct"] for r in recs]
        p_correct = [r["p_correct"] for r in recs]
        accuracy = np.mean(y_true)

        # Skip if all one class
        if len(set(y_true)) < 2:
            print(f"  Skipping ({bm}, {model}): all {'correct' if y_true[0] else 'incorrect'}")
            continue

        # Calibrator AUROC
        auroc_cal = safe_auroc(y_true, p_correct)

        # Verbalized confidence AUROC
        verb_valid = [r for r in recs if r["verbalized_confidence"] is not None]
        if len(verb_valid) >= 20 and len(set(r["is_correct"] for r in verb_valid)) >= 2:
            auroc_verb = safe_auroc(
                [r["is_correct"] for r in verb_valid],
                [r["verbalized_confidence"] for r in verb_valid],
            )
            n_verb = len(verb_valid)
        else:
            auroc_verb = None
            n_verb = len(verb_valid)

        # Length baseline AUROC (longer = less confident → negate)
        tok_valid = [r for r in recs if r["output_tokens"] > 0]
        if len(tok_valid) >= 20 and len(set(r["is_correct"] for r in tok_valid)) >= 2:
            # Longer responses are typically less correct, so use negative length
            auroc_len = safe_auroc(
                [r["is_correct"] for r in tok_valid],
                [-r["output_tokens"] for r in tok_valid],
            )
            n_len = len(tok_valid)
        else:
            auroc_len = None
            n_len = len(tok_valid)

        row = {
            "benchmark": bm,
            "target_model": model,
            "n_samples": n,
            "accuracy": round(accuracy, 4),
            "auroc_calibrator": round(auroc_cal, 4) if auroc_cal is not None else None,
            "auroc_verbalized": round(auroc_verb, 4) if auroc_verb is not None else None,
            "auroc_length": round(auroc_len, 4) if auroc_len is not None else None,
            "n_verbalized": n_verb,
            "n_length": n_len,
        }
        per_bm_rows.append(row)

    # Sort by accuracy ascending (hardest first)
    per_bm_rows.sort(key=lambda r: r["accuracy"])

    # Aggregate across models: average AUROC per benchmark
    bm_agg = defaultdict(lambda: {"auroc_cal": [], "auroc_verb": [], "auroc_len": [],
                                   "acc": [], "n": []})
    for row in per_bm_rows:
        bm = row["benchmark"]
        if row["auroc_calibrator"] is not None:
            bm_agg[bm]["auroc_cal"].append(row["auroc_calibrator"])
        if row["auroc_verbalized"] is not None:
            bm_agg[bm]["auroc_verb"].append(row["auroc_verbalized"])
        if row["auroc_length"] is not None:
            bm_agg[bm]["auroc_len"].append(row["auroc_length"])
        bm_agg[bm]["acc"].append(row["accuracy"])
        bm_agg[bm]["n"].append(row["n_samples"])

    bm_summary = []
    for bm, vals in bm_agg.items():
        bm_summary.append({
            "benchmark": bm,
            "mean_accuracy": round(np.mean(vals["acc"]), 4),
            "total_n": sum(vals["n"]),
            "mean_auroc_calibrator": round(np.mean(vals["auroc_cal"]), 4) if vals["auroc_cal"] else None,
            "mean_auroc_verbalized": round(np.mean(vals["auroc_verb"]), 4) if vals["auroc_verb"] else None,
            "mean_auroc_length": round(np.mean(vals["auroc_len"]), 4) if vals["auroc_len"] else None,
            "n_models": len(vals["acc"]),
        })
    bm_summary.sort(key=lambda r: r["mean_accuracy"])

    # -----------------------------------------------------------------------
    # 2. Paired significance tests (pooled across all data)
    # -----------------------------------------------------------------------
    # Prepare pooled arrays
    all_y = []
    all_cal = []
    all_verb = []
    all_len = []
    all_rand = []

    rng_rand = np.random.RandomState(123)

    # We need matching indices for paired tests, so work with records that have all fields
    for r in records:
        if r["p_correct"] is None or r["is_correct"] is None:
            continue
        all_y.append(r["is_correct"])
        all_cal.append(r["p_correct"])
        all_rand.append(rng_rand.random())

        if r["verbalized_confidence"] is not None:
            all_verb.append(r["verbalized_confidence"])
        else:
            all_verb.append(0.5)  # Neutral imputation for paired tests

        if r["output_tokens"] > 0:
            all_len.append(-r["output_tokens"])
        else:
            all_len.append(0.0)  # Neutral for missing

    all_y = np.array(all_y)
    all_cal = np.array(all_cal)
    all_verb = np.array(all_verb)
    all_len = np.array(all_len)
    all_rand = np.array(all_rand)

    sig_results = {}

    # --- Calibrator vs Verbalized ---
    # Use only records with valid verbalized confidence for fair comparison
    verb_mask = np.array([r["verbalized_confidence"] is not None for r in records
                          if r["p_correct"] is not None and r["is_correct"] is not None])
    y_verb = all_y[verb_mask]
    cal_verb = all_cal[verb_mask]
    verb_verb = all_verb[verb_mask]

    print(f"\n=== Significance Tests ===")
    print(f"Calibrator vs Verbalized ({verb_mask.sum()} paired samples)")

    auc_cal_v = safe_auroc(y_verb, cal_verb)
    auc_verb_v = safe_auroc(y_verb, verb_verb)
    print(f"  Calibrator AUROC: {auc_cal_v:.4f}")
    print(f"  Verbalized AUROC: {auc_verb_v:.4f}")

    z_dl, p_dl = delong_test(y_verb, cal_verb, verb_verb)
    chi2_mc, p_mc = mcnemar_test(y_verb, cal_verb, verb_verb)
    diff_perm, p_perm = permutation_test_auroc(y_verb, cal_verb, verb_verb, n_perm=10000)

    sig_results["calibrator_vs_verbalized"] = {
        "n_samples": int(verb_mask.sum()),
        "auroc_calibrator": round(auc_cal_v, 4) if auc_cal_v else None,
        "auroc_baseline": round(auc_verb_v, 4) if auc_verb_v else None,
        "delong": {"z": round(z_dl, 4), "p": round(p_dl, 6), "sig_005": p_dl < 0.05},
        "mcnemar": {"chi2": round(chi2_mc, 4), "p": round(p_mc, 6), "sig_005": p_mc < 0.05},
        "permutation": {"diff": round(diff_perm, 4), "p": round(p_perm, 6), "sig_005": p_perm < 0.05},
    }
    print(f"  DeLong:     z={z_dl:.4f}, p={p_dl:.6f} {'*' if p_dl < 0.05 else ''}")
    print(f"  McNemar:    chi2={chi2_mc:.4f}, p={p_mc:.6f} {'*' if p_mc < 0.05 else ''}")
    print(f"  Permutation: diff={diff_perm:.4f}, p={p_perm:.6f} {'*' if p_perm < 0.05 else ''}")

    # --- Calibrator vs Length ---
    # Use only records with valid output_tokens
    len_mask = np.array([r["output_tokens"] > 0 for r in records
                         if r["p_correct"] is not None and r["is_correct"] is not None])
    y_len = all_y[len_mask]
    cal_len = all_cal[len_mask]
    len_len = all_len[len_mask]

    print(f"\nCalibrator vs Length ({len_mask.sum()} paired samples)")

    auc_cal_l = safe_auroc(y_len, cal_len)
    auc_len_l = safe_auroc(y_len, len_len)
    print(f"  Calibrator AUROC: {auc_cal_l:.4f}")
    print(f"  Length AUROC: {auc_len_l:.4f}")

    z_dl2, p_dl2 = delong_test(y_len, cal_len, len_len)
    chi2_mc2, p_mc2 = mcnemar_test(y_len, cal_len, len_len)
    diff_perm2, p_perm2 = permutation_test_auroc(y_len, cal_len, len_len, n_perm=10000)

    sig_results["calibrator_vs_length"] = {
        "n_samples": int(len_mask.sum()),
        "auroc_calibrator": round(auc_cal_l, 4) if auc_cal_l else None,
        "auroc_baseline": round(auc_len_l, 4) if auc_len_l else None,
        "delong": {"z": round(z_dl2, 4), "p": round(p_dl2, 6), "sig_005": p_dl2 < 0.05},
        "mcnemar": {"chi2": round(chi2_mc2, 4), "p": round(p_mc2, 6), "sig_005": p_mc2 < 0.05},
        "permutation": {"diff": round(diff_perm2, 4), "p": round(p_perm2, 6), "sig_005": p_perm2 < 0.05},
    }
    print(f"  DeLong:     z={z_dl2:.4f}, p={p_dl2:.6f} {'*' if p_dl2 < 0.05 else ''}")
    print(f"  McNemar:    chi2={chi2_mc2:.4f}, p={p_mc2:.6f} {'*' if p_mc2 < 0.05 else ''}")
    print(f"  Permutation: diff={diff_perm2:.4f}, p={p_perm2:.6f} {'*' if p_perm2 < 0.05 else ''}")

    # --- Calibrator vs Random ---
    print(f"\nCalibrator vs Random ({len(all_y)} paired samples)")

    auc_cal_r = safe_auroc(all_y, all_cal)
    auc_rand_r = safe_auroc(all_y, all_rand)
    print(f"  Calibrator AUROC: {auc_cal_r:.4f}")
    print(f"  Random AUROC: {auc_rand_r:.4f}")

    z_dl3, p_dl3 = delong_test(all_y, all_cal, all_rand)
    chi2_mc3, p_mc3 = mcnemar_test(all_y, all_cal, all_rand)
    diff_perm3, p_perm3 = permutation_test_auroc(all_y, all_cal, all_rand, n_perm=10000)

    sig_results["calibrator_vs_random"] = {
        "n_samples": len(all_y),
        "auroc_calibrator": round(auc_cal_r, 4) if auc_cal_r else None,
        "auroc_baseline": round(auc_rand_r, 4) if auc_rand_r else None,
        "delong": {"z": round(z_dl3, 4), "p": round(p_dl3, 6), "sig_005": p_dl3 < 0.05},
        "mcnemar": {"chi2": round(chi2_mc3, 4), "p": round(p_mc3, 6), "sig_005": p_mc3 < 0.05},
        "permutation": {"diff": round(diff_perm3, 4), "p": round(p_perm3, 6), "sig_005": p_perm3 < 0.05},
    }
    print(f"  DeLong:     z={z_dl3:.4f}, p={p_dl3:.6f} {'*' if p_dl3 < 0.05 else ''}")
    print(f"  McNemar:    chi2={chi2_mc3:.4f}, p={p_mc3:.6f} {'*' if p_mc3 < 0.05 else ''}")
    print(f"  Permutation: diff={diff_perm3:.4f}, p={p_perm3:.6f} {'*' if p_perm3 < 0.05 else ''}")

    # -----------------------------------------------------------------------
    # 3. Summary statistics for paper
    # -----------------------------------------------------------------------
    print("\n=== Summary Statistics ===")

    # Overall AUROC with bootstrap CI
    overall_auroc = safe_auroc(all_y, all_cal)
    mean_boot, lo_boot, hi_boot = bootstrap_auroc_ci(all_y, all_cal, n_boot=1000)
    print(f"Overall test AUROC: {overall_auroc:.4f}  (95% CI: [{lo_boot:.4f}, {hi_boot:.4f}])")

    # Per-benchmark AUROC stats
    cal_aurocs = [r["mean_auroc_calibrator"] for r in bm_summary if r["mean_auroc_calibrator"] is not None]
    mean_bm_auroc = np.mean(cal_aurocs) if cal_aurocs else None
    std_bm_auroc = np.std(cal_aurocs) if cal_aurocs else None
    print(f"Per-benchmark AUROC: mean={mean_bm_auroc:.4f}, std={std_bm_auroc:.4f}")

    # Calibrator > verbalized count
    n_cal_wins = 0
    n_compared = 0
    for row in bm_summary:
        if row["mean_auroc_calibrator"] is not None and row["mean_auroc_verbalized"] is not None:
            n_compared += 1
            if row["mean_auroc_calibrator"] > row["mean_auroc_verbalized"]:
                n_cal_wins += 1
    print(f"Calibrator > verbalized: {n_cal_wins}/{n_compared} benchmarks")

    # Benchmarks with AUROC > 0.7
    n_above_07 = sum(1 for a in cal_aurocs if a > 0.7)
    print(f"Benchmarks with AUROC > 0.7: {n_above_07}/{len(cal_aurocs)}")

    # Best/worst 3
    sorted_bm = sorted(
        [(r["benchmark"], r["mean_auroc_calibrator"]) for r in bm_summary
         if r["mean_auroc_calibrator"] is not None],
        key=lambda x: x[1], reverse=True,
    )
    best3 = sorted_bm[:3]
    worst3 = sorted_bm[-3:]
    print(f"Best 3:  {', '.join(f'{b}={a:.4f}' for b, a in best3)}")
    print(f"Worst 3: {', '.join(f'{b}={a:.4f}' for b, a in worst3)}")

    summary_stats = {
        "overall_auroc": round(overall_auroc, 4) if overall_auroc else None,
        "bootstrap_ci_95": [round(lo_boot, 4), round(hi_boot, 4)],
        "bootstrap_mean": round(mean_boot, 4),
        "per_benchmark_auroc_mean": round(mean_bm_auroc, 4) if mean_bm_auroc else None,
        "per_benchmark_auroc_std": round(std_bm_auroc, 4) if std_bm_auroc else None,
        "calibrator_wins_vs_verbalized": f"{n_cal_wins}/{n_compared}",
        "benchmarks_above_07": f"{n_above_07}/{len(cal_aurocs)}",
        "best_3": [{"benchmark": b, "auroc": a} for b, a in best3],
        "worst_3": [{"benchmark": b, "auroc": a} for b, a in worst3],
        "total_test_samples": len(records),
    }

    # -----------------------------------------------------------------------
    # Print LaTeX table
    # -----------------------------------------------------------------------
    print("\n=== LaTeX Table (per-benchmark, averaged across models) ===")
    print(r"\begin{tabular}{lrrrrr}")
    print(r"\toprule")
    print(r"Benchmark & N & Acc. & Calibrator & Verbalized & Length \\")
    print(r"\midrule")
    for row in bm_summary:
        cal_str = f"{row['mean_auroc_calibrator']:.3f}" if row["mean_auroc_calibrator"] is not None else "---"
        verb_str = f"{row['mean_auroc_verbalized']:.3f}" if row["mean_auroc_verbalized"] is not None else "---"
        len_str = f"{row['mean_auroc_length']:.3f}" if row["mean_auroc_length"] is not None else "---"
        # Bold the best
        vals = []
        for v, s in [(row["mean_auroc_calibrator"], cal_str),
                      (row["mean_auroc_verbalized"], verb_str),
                      (row["mean_auroc_length"], len_str)]:
            vals.append((v if v is not None else -1, s))
        best_val = max(v for v, _ in vals)
        formatted = []
        for v, s in vals:
            if s != "---" and v == best_val:
                formatted.append(r"\textbf{" + s + "}")
            else:
                formatted.append(s)

        print(f"  {row['benchmark']:20s} & {row['total_n']:4d} & {row['mean_accuracy']:.3f} "
              f"& {formatted[0]} & {formatted[1]} & {formatted[2]} \\\\")
    print(r"\midrule")
    # Overall row
    overall_verb = safe_auroc(y_verb, verb_verb) if len(y_verb) > 0 else None
    overall_len = safe_auroc(y_len, len_len) if len(y_len) > 0 else None
    verb_str_ov = f"{overall_verb:.3f}" if overall_verb else "---"
    len_str_ov = f"{overall_len:.3f}" if overall_len else "---"
    print(f"  {'Overall':20s} & {len(all_y):4d} & {np.mean(all_y):.3f} "
          f"& \\textbf{{{overall_auroc:.3f}}} "
          f"& {verb_str_ov} "
          f"& {len_str_ov} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # -----------------------------------------------------------------------
    # Print detailed per (benchmark, model) table
    # -----------------------------------------------------------------------
    print("\n=== Detailed Per-(Benchmark, Model) Table ===")
    header = f"{'Benchmark':20s} {'Model':10s} {'N':>5s} {'Acc':>6s} {'Cal':>7s} {'Verb':>7s} {'Len':>7s}"
    print(header)
    print("-" * len(header))
    for row in per_bm_rows:
        cal_str = f"{row['auroc_calibrator']:.4f}" if row["auroc_calibrator"] is not None else "  ---"
        verb_str = f"{row['auroc_verbalized']:.4f}" if row["auroc_verbalized"] is not None else "  ---"
        len_str = f"{row['auroc_length']:.4f}" if row["auroc_length"] is not None else "  ---"
        print(f"{row['benchmark']:20s} {row['target_model']:10s} {row['n_samples']:5d} "
              f"{row['accuracy']:6.3f} {cal_str:>7s} {verb_str:>7s} {len_str:>7s}")

    # -----------------------------------------------------------------------
    # Save JSON results
    # -----------------------------------------------------------------------
    results = {
        "per_benchmark_model": per_bm_rows,
        "per_benchmark_summary": bm_summary,
        "significance_tests": sig_results,
        "summary_statistics": summary_stats,
    }

    out_path = os.path.join(output_dir, "per_benchmark_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # -----------------------------------------------------------------------
    # Figure 1: Per-benchmark AUROC bar chart
    # -----------------------------------------------------------------------
    _plot_per_benchmark_auroc(per_bm_rows, bm_summary, fig_dir)

    # -----------------------------------------------------------------------
    # Figure 2: Significance tests heatmap
    # -----------------------------------------------------------------------
    _plot_significance(sig_results, fig_dir)

    print(f"\nFigures saved to {fig_dir}/")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

MODEL_LABELS = {"gpt5mini": "GPT-5-mini", "gpt52": "GPT-5.2", "qwen35": "Qwen3.5"}
MODEL_COLORS = {"gpt5mini": "#1f77b4", "gpt52": "#ff7f0e", "qwen35": "#2ca02c"}


def _plot_per_benchmark_auroc(per_bm_rows, bm_summary, fig_dir):
    """Horizontal bar chart: one group per benchmark, bars for each model."""
    # Group rows by benchmark, ordered by average accuracy (hardest first)
    bm_order = [r["benchmark"] for r in bm_summary]
    models = sorted(set(r["target_model"] for r in per_bm_rows))

    bm_data = defaultdict(dict)
    for row in per_bm_rows:
        bm_data[row["benchmark"]][row["target_model"]] = row

    n_bm = len(bm_order)
    n_models = len(models)
    bar_height = 0.25
    fig_height = max(6, n_bm * 0.6)

    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_positions = np.arange(n_bm)

    for i, model in enumerate(models):
        aurocs = []
        for bm in bm_order:
            if model in bm_data[bm] and bm_data[bm][model]["auroc_calibrator"] is not None:
                aurocs.append(bm_data[bm][model]["auroc_calibrator"])
            else:
                aurocs.append(0)
        offset = (i - n_models / 2 + 0.5) * bar_height
        color = MODEL_COLORS.get(model, "#999999")
        label = MODEL_LABELS.get(model, model)
        bars = ax.barh(y_positions + offset, aurocs, height=bar_height,
                       color=color, alpha=0.85, label=label, edgecolor="white", linewidth=0.5)
        # Add value labels
        for bar, val in zip(bars, aurocs):
            if val > 0:
                ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}", va="center", fontsize=7)

    # Reference lines
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Random (0.5)")
    ax.axvline(0.7, color="red", linestyle=":", linewidth=0.8, alpha=0.6, label="Threshold (0.7)")

    ax.set_yticks(y_positions)
    # Add accuracy to y-labels
    bm_labels = []
    for bm in bm_order:
        acc = next((r["mean_accuracy"] for r in bm_summary if r["benchmark"] == bm), None)
        bm_labels.append(f"{bm} (acc={acc:.2f})" if acc is not None else bm)
    ax.set_yticklabels(bm_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC", fontsize=11)
    ax.set_title("Per-Benchmark Calibrator AUROC (test-only, sorted by difficulty)", fontsize=12)
    ax.set_xlim(0, 1.08)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(fig_dir, "per_benchmark_auroc.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_significance(sig_results, fig_dir):
    """Table-style visualization of significance test results."""
    comparisons = list(sig_results.keys())
    tests = ["delong", "mcnemar", "permutation"]
    test_labels = ["DeLong", "McNemar", "Permutation"]
    comp_labels = [c.replace("calibrator_vs_", "Cal. vs ").replace("_", " ").title()
                   for c in comparisons]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")

    # Build table data
    col_labels = ["Comparison", "N", "AUROC_cal", "AUROC_base"]
    for tl in test_labels:
        col_labels.extend([f"{tl} stat", f"{tl} p"])

    table_data = []
    cell_colors = []
    for comp in comparisons:
        r = sig_results[comp]
        row = [
            comp.replace("calibrator_vs_", "vs ").replace("_", " "),
            str(r["n_samples"]),
            f"{r['auroc_calibrator']:.3f}" if r["auroc_calibrator"] else "---",
            f"{r['auroc_baseline']:.3f}" if r["auroc_baseline"] else "---",
        ]
        row_colors = ["white"] * 4
        for t in tests:
            tr = r[t]
            if t == "delong":
                stat_val = tr.get("z", np.nan)
            elif t == "mcnemar":
                stat_val = tr.get("chi2", np.nan)
            else:
                stat_val = tr.get("diff", np.nan)

            p_val = tr.get("p", np.nan)
            sig = tr.get("sig_005", False)

            row.append(f"{stat_val:.3f}" if not np.isnan(stat_val) else "---")
            p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "---"
            if sig:
                p_str += " *"
            row.append(p_str)
            row_colors.append("white")
            row_colors.append("#d4edda" if sig else "#f8d7da")

        table_data.append(row)
        cell_colors.append(row_colors)

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellColours=cell_colors,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Paired Significance Tests: Calibrator vs Baselines", fontsize=12, pad=20)
    plt.tight_layout()
    path = os.path.join(fig_dir, "significance_tests.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-benchmark AUROC analysis with significance tests")
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only",
                        help="Directory with scored JSONL files")
    parser.add_argument("--output_dir", default="data/use_cases/results_test_only",
                        help="Directory for output JSON")
    parser.add_argument("--fig_dir", default="figures/use_cases_test_only",
                        help="Directory for output figures")
    args = parser.parse_args()
    run_analysis(args.scored_dir, args.output_dir, args.fig_dir)


if __name__ == "__main__":
    main()
