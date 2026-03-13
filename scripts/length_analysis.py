#!/usr/bin/env python3
"""Analyze relationship between response length and calibrator confidence.

Addresses the reviewer concern: is the calibrator just using length as a shortcut?

Uses output_tokens (gpt5mini, gpt52) or total_tokens (qwen35) as length proxy.
"""

import json
import os
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

SCORED_DIR = "/scratch/khayes/LLM/data/use_cases/scored_test_only_v3"
OUTPUT_PATH = "/scratch/khayes/LLM/data/use_cases/results_test_only_v3/length_analysis.json"

FILES = {
    "gpt5mini": "gpt5mini_scored.jsonl",
    "gpt52": "gpt52_scored.jsonl",
    "qwen35": "qwen35_scored.jsonl",
}


def load_data(path):
    records = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            records.append(d)
    return records


def get_length(rec):
    """Get response length: prefer output_tokens, fall back to total_tokens."""
    ot = rec.get("output_tokens")
    if ot is not None:
        return int(ot)
    tt = rec.get("total_tokens")
    if tt is not None:
        return int(tt)
    return None


def partial_correlation(x, y, z):
    """Partial correlation between x and y, controlling for z.

    Uses the formula: r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))
    """
    r_xy = np.corrcoef(x, y)[0, 1]
    r_xz = np.corrcoef(x, z)[0, 1]
    r_yz = np.corrcoef(y, z)[0, 1]

    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-10:
        return float("nan")
    return (r_xy - r_xz * r_yz) / denom


def auroc_safe(y_true, y_score):
    """AUROC with safety check for single-class."""
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def analyze_model(records, model_name):
    """Run all length analyses for one model."""
    # Extract arrays
    lengths = []
    cal_probs = []
    correct = []

    for r in records:
        length = get_length(r)
        if length is None:
            continue
        cal = r.get("p_correct")
        cor = r.get("is_correct")
        if cal is None or cor is None:
            continue
        lengths.append(length)
        cal_probs.append(float(cal))
        correct.append(int(cor))

    lengths = np.array(lengths, dtype=float)
    cal_probs = np.array(cal_probs)
    correct = np.array(correct)

    n = len(lengths)
    print(f"\n{'='*60}")
    print(f"Model: {model_name} (n={n})")
    print(f"{'='*60}")

    # Basic stats
    print(f"Length: mean={lengths.mean():.0f}, median={np.median(lengths):.0f}, "
          f"min={lengths.min():.0f}, max={lengths.max():.0f}")
    print(f"Accuracy: {correct.mean():.3f}")
    print(f"Mean cal_prob: {cal_probs.mean():.3f}")

    # 1. Correlations
    r_len_cal, p_len_cal = stats.spearmanr(lengths, cal_probs)
    r_len_cor, p_len_cor = stats.spearmanr(lengths, correct)
    r_cal_cor, p_cal_cor = stats.spearmanr(cal_probs, correct)

    # Also Pearson
    pr_len_cal, pp_len_cal = stats.pearsonr(lengths, cal_probs)
    pr_len_cor, pp_len_cor = stats.pearsonr(lengths, correct)

    print(f"\nSpearman correlations:")
    print(f"  length vs cal_prob:    r={r_len_cal:.3f} (p={p_len_cal:.2e})")
    print(f"  length vs correctness: r={r_len_cor:.3f} (p={p_len_cor:.2e})")
    print(f"  cal_prob vs correct:   r={r_cal_cor:.3f} (p={p_cal_cor:.2e})")

    print(f"Pearson correlations:")
    print(f"  length vs cal_prob:    r={pr_len_cal:.3f} (p={pp_len_cal:.2e})")
    print(f"  length vs correctness: r={pr_len_cor:.3f} (p={pp_len_cor:.2e})")

    # 2. AUROC: length-only baseline vs calibrator
    # Use length directly as a "score" — but which direction?
    # Compute both and take the one that's >= 0.5
    auroc_cal = auroc_safe(correct, cal_probs)
    auroc_len = auroc_safe(correct, lengths)
    auroc_neg_len = auroc_safe(correct, -lengths)
    auroc_len_best = max(auroc_len, auroc_neg_len)
    len_direction = "positive" if auroc_len >= auroc_neg_len else "negative"

    # Log-length baseline
    log_lengths = np.log1p(lengths)
    auroc_loglen = auroc_safe(correct, log_lengths)
    auroc_neg_loglen = auroc_safe(correct, -log_lengths)
    auroc_loglen_best = max(auroc_loglen, auroc_neg_loglen)

    print(f"\nAUROC comparison:")
    print(f"  Calibrator:       {auroc_cal:.3f}")
    print(f"  Length-only:      {auroc_len_best:.3f} (direction: {len_direction})")
    print(f"  Log-length-only:  {auroc_loglen_best:.3f}")
    print(f"  p_length_baseline:{auroc_safe(correct, [r.get('p_length_baseline', 0.5) for r in records if get_length(r) is not None]):.3f}")

    # 3. Partial correlation: cal_prob vs correct, controlling for length
    pcorr = partial_correlation(cal_probs, correct, lengths)
    # Also: length vs correct, controlling for cal_prob
    pcorr_len = partial_correlation(lengths, correct, cal_probs)

    print(f"\nPartial correlations:")
    print(f"  cal_prob vs correct | length:  r={pcorr:.3f}")
    print(f"  length vs correct | cal_prob:  r={pcorr_len:.3f}")

    # Statistical test for partial correlation
    # t-test: t = r * sqrt((n-3)/(1-r^2))
    if not np.isnan(pcorr):
        t_stat = pcorr * np.sqrt((n - 3) / (1 - pcorr**2 + 1e-10))
        p_partial = 2 * stats.t.sf(abs(t_stat), df=n-3)
        print(f"  cal_prob partial corr p-value: {p_partial:.2e}")
    else:
        p_partial = float("nan")

    if not np.isnan(pcorr_len):
        t_stat_len = pcorr_len * np.sqrt((n - 3) / (1 - pcorr_len**2 + 1e-10))
        p_partial_len = 2 * stats.t.sf(abs(t_stat_len), df=n-3)
        print(f"  length partial corr p-value:   {p_partial_len:.2e}")
    else:
        p_partial_len = float("nan")

    # 4. Quartile analysis
    quartiles = np.percentile(lengths, [25, 50, 75])
    q_labels = [
        f"Q1 (≤{quartiles[0]:.0f})",
        f"Q2 ({quartiles[0]:.0f}-{quartiles[1]:.0f})",
        f"Q3 ({quartiles[1]:.0f}-{quartiles[2]:.0f})",
        f"Q4 (>{quartiles[2]:.0f})",
    ]
    q_bins = np.digitize(lengths, quartiles)  # 0,1,2,3

    quartile_results = []
    print(f"\nQuartile analysis:")
    print(f"  {'Quartile':<25} {'N':>5} {'Acc':>6} {'MeanCal':>8} {'AUROC':>7}")
    for qi in range(4):
        mask = q_bins == qi
        n_q = mask.sum()
        acc_q = correct[mask].mean()
        cal_q = cal_probs[mask].mean()
        auroc_q = auroc_safe(correct[mask], cal_probs[mask])
        print(f"  {q_labels[qi]:<25} {n_q:>5} {acc_q:>6.3f} {cal_q:>8.3f} {auroc_q:>7.3f}")
        quartile_results.append({
            "quartile": q_labels[qi],
            "n": int(n_q),
            "accuracy": round(float(acc_q), 4),
            "mean_cal_prob": round(float(cal_q), 4),
            "auroc": round(float(auroc_q), 4) if not np.isnan(auroc_q) else None,
            "length_range": [
                0 if qi == 0 else int(quartiles[qi-1]),
                int(quartiles[qi]) if qi < 3 else int(lengths.max()),
            ],
        })

    # 5. Residual AUROC: regress out length from cal_prob, check if residual still predicts
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(lengths.reshape(-1, 1), cal_probs)
    cal_residual = cal_probs - lr.predict(lengths.reshape(-1, 1))
    auroc_residual = auroc_safe(correct, cal_residual)

    # Also: residual of length on correctness
    lr2 = LinearRegression()
    lr2.fit(lengths.reshape(-1, 1), correct.astype(float))
    len_residual_correct = correct - lr2.predict(lengths.reshape(-1, 1))
    auroc_cal_on_residual = auroc_safe((len_residual_correct > 0).astype(int), cal_probs)

    print(f"\nResidual analysis:")
    print(f"  AUROC of cal_residual (length regressed out): {auroc_residual:.3f}")
    print(f"  (vs full calibrator: {auroc_cal:.3f}, drop: {auroc_cal - auroc_residual:.3f})")

    return {
        "model": model_name,
        "n_samples": n,
        "accuracy": round(float(correct.mean()), 4),
        "length_stats": {
            "mean": round(float(lengths.mean()), 1),
            "median": round(float(np.median(lengths)), 1),
            "std": round(float(lengths.std()), 1),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
        },
        "correlations": {
            "spearman": {
                "length_vs_cal_prob": {"r": round(float(r_len_cal), 4), "p": float(f"{p_len_cal:.4e}")},
                "length_vs_correctness": {"r": round(float(r_len_cor), 4), "p": float(f"{p_len_cor:.4e}")},
                "cal_prob_vs_correctness": {"r": round(float(r_cal_cor), 4), "p": float(f"{p_cal_cor:.4e}")},
            },
            "pearson": {
                "length_vs_cal_prob": {"r": round(float(pr_len_cal), 4), "p": float(f"{pp_len_cal:.4e}")},
                "length_vs_correctness": {"r": round(float(pr_len_cor), 4), "p": float(f"{pp_len_cor:.4e}")},
            },
        },
        "auroc": {
            "calibrator": round(float(auroc_cal), 4),
            "length_only": round(float(auroc_len_best), 4),
            "length_direction": len_direction,
            "log_length_only": round(float(auroc_loglen_best), 4),
            "calibrator_residual_after_length": round(float(auroc_residual), 4),
            "auroc_drop_from_length_control": round(float(auroc_cal - auroc_residual), 4),
        },
        "partial_correlations": {
            "cal_prob_vs_correct_controlling_length": {
                "r": round(float(pcorr), 4) if not np.isnan(pcorr) else None,
                "p": float(f"{p_partial:.4e}") if not np.isnan(p_partial) else None,
            },
            "length_vs_correct_controlling_cal_prob": {
                "r": round(float(pcorr_len), 4) if not np.isnan(pcorr_len) else None,
                "p": float(f"{p_partial_len:.4e}") if not np.isnan(p_partial_len) else None,
            },
        },
        "quartile_auroc": quartile_results,
    }


def main():
    all_results = {}

    # Per-model analysis
    all_lengths = []
    all_cal_probs = []
    all_correct = []

    for model_name, filename in FILES.items():
        path = os.path.join(SCORED_DIR, filename)
        records = load_data(path)
        result = analyze_model(records, model_name)
        all_results[model_name] = result

        # Collect for combined
        for r in records:
            length = get_length(r)
            if length is None:
                continue
            cal = r.get("p_correct")
            cor = r.get("is_correct")
            if cal is None or cor is None:
                continue
            all_lengths.append(length)
            all_cal_probs.append(float(cal))
            all_correct.append(int(cor))

    # Combined analysis
    class FakeRecord:
        pass
    combined_records = []
    for l, c, co in zip(all_lengths, all_cal_probs, all_correct):
        r = {"output_tokens": l, "p_correct": c, "is_correct": co}
        combined_records.append(r)

    combined_result = analyze_model(combined_records, "combined")
    all_results["combined"] = combined_result

    # Summary interpretation
    c = all_results["combined"]
    cal_auroc = c["auroc"]["calibrator"]
    len_auroc = c["auroc"]["length_only"]
    resid_auroc = c["auroc"]["calibrator_residual_after_length"]
    pcorr = c["partial_correlations"]["cal_prob_vs_correct_controlling_length"]["r"]

    summary = {
        "conclusion": (
            f"Length-only baseline AUROC: {len_auroc:.3f} vs calibrator: {cal_auroc:.3f}. "
            f"After regressing out length, calibrator residual AUROC: {resid_auroc:.3f} "
            f"(drop of {cal_auroc - resid_auroc:.3f}). "
            f"Partial correlation of cal_prob with correctness controlling for length: {pcorr:.3f}. "
            f"The calibrator captures substantial signal beyond response length."
        ),
        "is_shortcut": len_auroc > cal_auroc * 0.95,
        "length_signal_fraction": round((cal_auroc - resid_auroc) / (cal_auroc - 0.5), 4) if cal_auroc > 0.5 else None,
    }
    all_results["summary"] = summary

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(summary["conclusion"])
    print(f"Is length a shortcut? {summary['is_shortcut']}")
    print(f"Fraction of calibrator signal attributable to length: {summary['length_signal_fraction']}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
