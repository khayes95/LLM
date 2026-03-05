#!/usr/bin/env python3
"""UC-D: Agent Step Verification — Stage 1 (Step-Truncation Confidence Analysis).

Analyzes multi-step reasoning responses using existing calibrator scores to
determine whether response length and structure carry signal about correctness,
and whether there is enough data to justify a GPU-based Stage 2 that would
run the calibrator on truncated responses.

The hypothesis: correct responses show stable/increasing confidence while
incorrect ones show a "confidence crash" at some reasoning step.  Stage 1
validates this indirectly by checking whether response length, structure, and
calibrator confidence interact in the predicted way.

Analyses:
1. Multi-Step Response Detection — identify structured reasoning (JSON with
   "reasoning" field, numbered steps, bullet points, paragraph breaks) and
   long responses (>500 output tokens).
2. Length-Correctness Analysis — bin by output_tokens quartiles, compare
   mean p_correct vs actual accuracy per quartile.
3. Confidence-Length Correlation — Pearson/Spearman between output_tokens
   and p_correct / is_correct.  Scatter plot colored by correctness.
4. "Overthinking" Detection — long responses (top quartile) with low
   calibrator confidence (p_correct < 0.3).  Accuracy of these vs the rest.
5. Verbalized vs Calibrator Disagreement in Long Responses — how often
   does a long response have high verbalized confidence but low p_correct?
6. Stage 2 Feasibility Check — count responses with > 500 output tokens,
   estimate truncation inference calls and GPU time.

Outputs:
    {output_dir}/uc_d_results.json — all metrics and tables
    {fig_dir}/uc_d_length_analysis.pdf — length vs confidence scatter + quartile bars
    {fig_dir}/uc_d_overthinking.pdf — overthinking detection analysis

Usage:
    python scripts/uc_d_agent_steps.py
    python scripts/uc_d_agent_steps.py --scored_dir data/use_cases/scored_test_only_v2
    python scripts/uc_d_agent_steps.py --smoke_test
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scored(path):
    """Load scored JSONL, converting fields to native types."""
    samples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            # Robustly convert — handle both string and native types
            row["is_correct"] = int(row["is_correct"])
            row["p_correct"] = float(row["p_correct"])
            ot = row.get("output_tokens")
            row["output_tokens"] = int(ot) if ot is not None else 0
            vc = row.get("verbalized_confidence")
            if vc is None or str(vc).strip().lower() == "none":
                row["verbalized_confidence"] = None
            else:
                row["verbalized_confidence"] = float(vc)
            hi = row.get("has_image")
            if isinstance(hi, str):
                row["has_image"] = hi.strip().lower() == "true"
            else:
                row["has_image"] = bool(hi) if hi is not None else False
            samples.append(row)
    return samples


# ---------------------------------------------------------------------------
# 1. Multi-Step Response Detection
# ---------------------------------------------------------------------------

_STEP_PATTERNS = [
    re.compile(r'"reasoning"', re.IGNORECASE),       # JSON reasoning field
    re.compile(r'(?:^|\n)\s*(?:Step\s+\d|#\s+Step)', re.IGNORECASE),  # "Step 1"
    re.compile(r'(?:^|\n)\s*\d+[\.\)]\s'),           # "1. " or "1) "
    re.compile(r'(?:^|\n)\s*[-*]\s'),                 # bullet points
    re.compile(r'\n\n'),                              # paragraph breaks
]


def detect_multi_step(response_preview, output_tokens):
    """Heuristically identify multi-step responses.

    Returns a dict with boolean indicators for each detection method.
    """
    indicators = {
        "has_json_reasoning": bool(_STEP_PATTERNS[0].search(response_preview or "")),
        "has_numbered_steps": bool(_STEP_PATTERNS[1].search(response_preview or "")),
        "has_numbered_list": bool(_STEP_PATTERNS[2].search(response_preview or "")),
        "has_bullets": bool(_STEP_PATTERNS[3].search(response_preview or "")),
        "has_paragraphs": bool(_STEP_PATTERNS[4].search(response_preview or "")),
        "long_response": output_tokens > 500,
    }
    indicators["any_structural"] = any([
        indicators["has_json_reasoning"],
        indicators["has_numbered_steps"],
        indicators["has_numbered_list"],
        indicators["has_bullets"],
        indicators["has_paragraphs"],
    ])
    indicators["is_multi_step"] = indicators["any_structural"] or indicators["long_response"]
    return indicators


def multi_step_summary(samples):
    """Compute multi-step detection statistics per benchmark and overall."""
    overall = defaultdict(int)
    by_benchmark = defaultdict(lambda: defaultdict(int))

    for s in samples:
        ind = detect_multi_step(s.get("response_preview", ""), s["output_tokens"])
        s["_multi_step"] = ind  # attach for later use

        for key, val in ind.items():
            if val:
                overall[key] += 1
        overall["total"] += 1

        bench = s["benchmark"]
        for key, val in ind.items():
            if val:
                by_benchmark[bench][key] += 1
        by_benchmark[bench]["total"] += 1

    return dict(overall), {k: dict(v) for k, v in by_benchmark.items()}


# ---------------------------------------------------------------------------
# 2. Length-Correctness Analysis
# ---------------------------------------------------------------------------

def length_quartile_analysis(samples):
    """Bin responses by output_tokens quartiles, compute stats per bin."""
    tokens = np.array([s["output_tokens"] for s in samples])
    # Handle case where all tokens are 0
    if tokens.max() == 0:
        return []

    quartile_edges = np.quantile(tokens, [0, 0.25, 0.5, 0.75, 1.0])
    # Ensure unique edges (if many identical values)
    quartile_edges = np.unique(quartile_edges)
    if len(quartile_edges) < 2:
        quartile_edges = np.array([tokens.min(), tokens.max()])

    results = []
    n_bins = len(quartile_edges) - 1
    for q in range(n_bins):
        lo = quartile_edges[q]
        hi = quartile_edges[q + 1]
        if q == n_bins - 1:
            mask = (tokens >= lo) & (tokens <= hi)
        else:
            mask = (tokens >= lo) & (tokens < hi)

        subset = [s for s, m in zip(samples, mask) if m]
        if not subset:
            continue

        correct = np.array([s["is_correct"] for s in subset])
        p_corr = np.array([s["p_correct"] for s in subset])

        actual_acc = float(correct.mean())
        mean_p = float(p_corr.mean())

        results.append({
            "quartile": q + 1,
            "token_lo": int(lo),
            "token_hi": int(hi),
            "n_samples": len(subset),
            "mean_output_tokens": float(np.mean([s["output_tokens"] for s in subset])),
            "actual_accuracy": actual_acc,
            "mean_p_correct": mean_p,
            "calibrator_accuracy_gap": mean_p - actual_acc,
        })

    return results


# ---------------------------------------------------------------------------
# 3. Confidence-Length Correlation
# ---------------------------------------------------------------------------

def confidence_length_correlation(samples):
    """Compute correlation between output_tokens and p_correct / is_correct."""
    tokens = np.array([s["output_tokens"] for s in samples], dtype=float)
    p_corr = np.array([s["p_correct"] for s in samples], dtype=float)
    correct = np.array([s["is_correct"] for s in samples], dtype=float)

    result = {}

    # Filter out zero-token samples for meaningful correlation
    valid = tokens > 0
    if valid.sum() < 10:
        return {"insufficient_data": True, "n_valid": int(valid.sum())}

    t_valid = tokens[valid]
    p_valid = p_corr[valid]
    c_valid = correct[valid]

    # Pearson and Spearman: tokens vs p_correct
    if np.std(t_valid) > 0 and np.std(p_valid) > 0:
        r_pearson, p_pearson = stats.pearsonr(t_valid, p_valid)
        r_spearman, p_spearman = stats.spearmanr(t_valid, p_valid)
        result["tokens_vs_p_correct"] = {
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_pearson),
            "spearman_r": float(r_spearman),
            "spearman_p": float(p_spearman),
        }

    # Pearson and Spearman: tokens vs is_correct
    if np.std(t_valid) > 0 and np.std(c_valid) > 0:
        r_pearson, p_pearson = stats.pearsonr(t_valid, c_valid)
        r_spearman, p_spearman = stats.spearmanr(t_valid, c_valid)
        result["tokens_vs_is_correct"] = {
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_pearson),
            "spearman_r": float(r_spearman),
            "spearman_p": float(p_spearman),
        }

    # Pearson and Spearman: p_correct vs is_correct (calibrator signal beyond length)
    if np.std(p_valid) > 0 and np.std(c_valid) > 0:
        r_pearson, p_pearson = stats.pearsonr(p_valid, c_valid)
        result["p_correct_vs_is_correct"] = {
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_pearson),
        }

    # Partial correlation: p_correct vs is_correct controlling for length
    # (Does the calibrator add value beyond just knowing the response length?)
    if np.std(t_valid) > 0 and np.std(p_valid) > 0 and np.std(c_valid) > 0:
        # Residualize p_correct and is_correct on tokens
        slope_p, intercept_p, _, _, _ = stats.linregress(t_valid, p_valid)
        resid_p = p_valid - (slope_p * t_valid + intercept_p)
        slope_c, intercept_c, _, _, _ = stats.linregress(t_valid, c_valid)
        resid_c = c_valid - (slope_c * t_valid + intercept_c)
        if np.std(resid_p) > 0 and np.std(resid_c) > 0:
            r_partial, p_partial = stats.pearsonr(resid_p, resid_c)
            result["partial_p_correct_vs_is_correct_given_length"] = {
                "pearson_r": float(r_partial),
                "pearson_p": float(p_partial),
            }

    result["n_valid"] = int(valid.sum())
    result["insufficient_data"] = False
    return result


# ---------------------------------------------------------------------------
# 4. "Overthinking" Detection
# ---------------------------------------------------------------------------

def overthinking_analysis(samples):
    """Find responses in the top token quartile with low calibrator confidence.

    These are "overthinking" candidates — long reasoning chains that the
    calibrator thinks are wrong.
    """
    tokens = np.array([s["output_tokens"] for s in samples])
    if tokens.max() == 0:
        return {"insufficient_data": True}

    q75 = np.quantile(tokens, 0.75)
    # Handle edge case where q75 == 0 (many zero-token samples)
    if q75 == 0:
        q75 = np.quantile(tokens[tokens > 0], 0.75) if (tokens > 0).any() else 1

    long_mask = tokens >= q75
    long_samples = [s for s, m in zip(samples, long_mask) if m]
    rest_samples = [s for s, m in zip(samples, long_mask) if not m]

    # Overthinking = long AND low calibrator confidence
    overthinking = [s for s in long_samples if s["p_correct"] < 0.3]
    long_high_conf = [s for s in long_samples if s["p_correct"] >= 0.3]

    result = {
        "q75_token_threshold": int(q75),
        "n_long": len(long_samples),
        "n_rest": len(rest_samples),
        "n_overthinking": len(overthinking),
        "n_long_high_conf": len(long_high_conf),
    }

    if long_samples:
        result["long_accuracy"] = float(np.mean([s["is_correct"] for s in long_samples]))
        result["long_mean_p_correct"] = float(np.mean([s["p_correct"] for s in long_samples]))
    if rest_samples:
        result["rest_accuracy"] = float(np.mean([s["is_correct"] for s in rest_samples]))
        result["rest_mean_p_correct"] = float(np.mean([s["p_correct"] for s in rest_samples]))

    if overthinking:
        result["overthinking_accuracy"] = float(np.mean([s["is_correct"] for s in overthinking]))
        result["overthinking_mean_p_correct"] = float(np.mean([s["p_correct"] for s in overthinking]))
    else:
        result["overthinking_accuracy"] = None
        result["overthinking_mean_p_correct"] = None

    if long_high_conf:
        result["long_high_conf_accuracy"] = float(np.mean([s["is_correct"] for s in long_high_conf]))
    else:
        result["long_high_conf_accuracy"] = None

    # Per-benchmark overthinking
    bench_overthinking = defaultdict(lambda: {"n_overthinking": 0, "n_correct": 0, "n_total": 0})
    for s in overthinking:
        bench_overthinking[s["benchmark"]]["n_overthinking"] += 1
        bench_overthinking[s["benchmark"]]["n_correct"] += s["is_correct"]
    for s in samples:
        bench_overthinking[s["benchmark"]]["n_total"] += 1

    result["per_benchmark"] = {}
    for bench, bd in sorted(bench_overthinking.items()):
        if bd["n_overthinking"] > 0:
            result["per_benchmark"][bench] = {
                "n_overthinking": bd["n_overthinking"],
                "accuracy": bd["n_correct"] / bd["n_overthinking"],
                "frac_of_benchmark": bd["n_overthinking"] / bd["n_total"] if bd["n_total"] > 0 else 0,
            }

    result["insufficient_data"] = False
    return result


# ---------------------------------------------------------------------------
# 5. Verbalized vs Calibrator Disagreement in Long Responses
# ---------------------------------------------------------------------------

def verbalized_calibrator_disagreement(samples):
    """Among long responses: how often does verbalized_confidence > 0.7 but
    p_correct < 0.3?  These are "confident but wrong" long responses."""
    tokens = np.array([s["output_tokens"] for s in samples])
    if tokens.max() == 0 or len(tokens) == 0:
        return {"insufficient_data": True}

    median_tokens = float(np.median(tokens[tokens > 0])) if (tokens > 0).any() else 0
    long_samples = [s for s in samples if s["output_tokens"] > median_tokens]

    # Filter to those with verbalized confidence
    long_with_verb = [s for s in long_samples if s["verbalized_confidence"] is not None]

    if not long_with_verb:
        return {
            "n_long": len(long_samples),
            "n_long_with_verbalized": 0,
            "median_token_threshold": median_tokens,
            "insufficient_data": True,
        }

    # Confident but wrong (by calibrator): verb > 0.7 AND p_correct < 0.3
    confident_but_flagged = [
        s for s in long_with_verb
        if s["verbalized_confidence"] > 0.7 and s["p_correct"] < 0.3
    ]

    # High agreement: both high
    both_high = [
        s for s in long_with_verb
        if s["verbalized_confidence"] > 0.7 and s["p_correct"] > 0.7
    ]

    # Both low
    both_low = [
        s for s in long_with_verb
        if s["verbalized_confidence"] < 0.3 and s["p_correct"] < 0.3
    ]

    result = {
        "median_token_threshold": median_tokens,
        "n_long": len(long_samples),
        "n_long_with_verbalized": len(long_with_verb),
        "n_confident_but_flagged": len(confident_but_flagged),
        "frac_confident_but_flagged": len(confident_but_flagged) / len(long_with_verb),
        "n_both_high": len(both_high),
        "n_both_low": len(both_low),
        "insufficient_data": False,
    }

    if confident_but_flagged:
        result["confident_but_flagged_accuracy"] = float(
            np.mean([s["is_correct"] for s in confident_but_flagged]))
    else:
        result["confident_but_flagged_accuracy"] = None

    if both_high:
        result["both_high_accuracy"] = float(np.mean([s["is_correct"] for s in both_high]))
    if both_low:
        result["both_low_accuracy"] = float(np.mean([s["is_correct"] for s in both_low]))

    return result


# ---------------------------------------------------------------------------
# 6. Stage 2 Feasibility Check
# ---------------------------------------------------------------------------

def stage2_feasibility(samples, truncation_points=5, inferences_per_minute=100):
    """Estimate the compute cost of a Stage 2 step-truncation experiment."""
    tokens = np.array([s["output_tokens"] for s in samples])

    n_long_500 = int((tokens > 500).sum())
    n_long_1000 = int((tokens > 1000).sum())
    n_long_2000 = int((tokens > 2000).sum())

    total_inferences = n_long_500 * truncation_points
    est_minutes = total_inferences / inferences_per_minute if inferences_per_minute > 0 else 0
    est_hours = est_minutes / 60

    # Token distribution stats for candidates
    candidate_tokens = tokens[tokens > 500]
    result = {
        "n_candidates_500_plus": n_long_500,
        "n_candidates_1000_plus": n_long_1000,
        "n_candidates_2000_plus": n_long_2000,
        "frac_candidates_500_plus": n_long_500 / len(samples) if samples else 0,
        "truncation_points_per_response": truncation_points,
        "total_inference_calls": total_inferences,
        "estimated_minutes_single_gpu": float(est_minutes),
        "estimated_hours_single_gpu": float(est_hours),
        "assumed_inferences_per_minute": inferences_per_minute,
    }

    if len(candidate_tokens) > 0:
        result["candidate_token_stats"] = {
            "mean": float(candidate_tokens.mean()),
            "median": float(np.median(candidate_tokens)),
            "min": int(candidate_tokens.min()),
            "max": int(candidate_tokens.max()),
            "std": float(candidate_tokens.std()),
        }

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_length_analysis(all_model_results, fig_path):
    """Two-panel figure per model:
    Left — scatter of output_tokens vs p_correct, colored by is_correct.
    Right — quartile bar chart comparing mean_p_correct vs actual_accuracy.
    """
    model_names = {
        "gpt5mini": "GPT-5-mini",
        "gpt52": "GPT-5.2",
        "qwen35": "Qwen3.5",
    }

    n_models = len(all_model_results)
    fig, axes = plt.subplots(2, n_models, figsize=(7 * n_models, 11), squeeze=False)

    for col, (target, data) in enumerate(all_model_results.items()):
        samples = data["_samples"]
        quartiles = data["length_quartiles"]
        corr = data["correlations"]

        # --- Top row: scatter ---
        ax = axes[0, col]
        tokens = np.array([s["output_tokens"] for s in samples], dtype=float)
        p_corr = np.array([s["p_correct"] for s in samples])
        correct = np.array([s["is_correct"] for s in samples])

        # Jitter for visibility
        rng = np.random.RandomState(42)
        jitter = rng.normal(0, 0.01, size=len(p_corr))

        # Plot incorrect first (behind), then correct
        mask_wrong = correct == 0
        mask_right = correct == 1
        ax.scatter(tokens[mask_wrong], p_corr[mask_wrong] + jitter[mask_wrong],
                   alpha=0.15, s=8, c="C3", label=f"Incorrect (n={mask_wrong.sum()})",
                   rasterized=True)
        ax.scatter(tokens[mask_right], p_corr[mask_right] + jitter[mask_right],
                   alpha=0.15, s=8, c="C0", label=f"Correct (n={mask_right.sum()})",
                   rasterized=True)

        # Annotate correlation
        tok_p = corr.get("tokens_vs_p_correct", {})
        if tok_p:
            ax.text(0.02, 0.98,
                    f"Spearman r={tok_p.get('spearman_r', 0):.3f} "
                    f"(p={tok_p.get('spearman_p', 1):.1e})",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("Output tokens", fontsize=11)
        ax.set_ylabel("Calibrator P(correct)", fontsize=11)
        ax.set_title(f"{model_names.get(target, target)} — Length vs Confidence",
                     fontsize=12)
        ax.legend(fontsize=8, loc="lower right", markerscale=3)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # --- Bottom row: quartile bars ---
        ax = axes[1, col]
        if not quartiles:
            ax.text(0.5, 0.5, "No quartile data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        x = np.arange(len(quartiles))
        labels = [f"Q{q['quartile']}\n[{q['token_lo']}-{q['token_hi']}]\nn={q['n_samples']}"
                  for q in quartiles]
        w = 0.35

        actual = [q["actual_accuracy"] for q in quartiles]
        predicted = [q["mean_p_correct"] for q in quartiles]

        bars1 = ax.bar(x - w / 2, actual, w, label="Actual Accuracy", color="C0",
                       alpha=0.85, edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + w / 2, predicted, w, label="Mean P(correct)", color="C1",
                       alpha=0.85, edgecolor="black", linewidth=0.5)

        for xi, (a, p) in enumerate(zip(actual, predicted)):
            ax.text(xi - w / 2, a + 0.02, f"{a:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
            ax.text(xi + w / 2, p + 0.02, f"{p:.3f}", ha="center", va="bottom",
                    fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Accuracy / P(correct)", fontsize=11)
        ax.set_title(f"{model_names.get(target, target)} — By Output Token Quartile",
                     fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("UC-D: Agent Step Verification — Length vs Confidence Analysis",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {fig_path}")


def plot_overthinking(all_model_results, fig_path):
    """Multi-panel figure showing overthinking detection results.

    Left — accuracy comparison: overthinking vs rest vs long-high-conf.
    Right — per-benchmark overthinking fraction and accuracy.
    """
    model_names = {
        "gpt5mini": "GPT-5-mini",
        "gpt52": "GPT-5.2",
        "qwen35": "Qwen3.5",
    }

    n_models = len(all_model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6), squeeze=False)

    for col, (target, data) in enumerate(all_model_results.items()):
        ax = axes[0, col]
        ot = data["overthinking"]

        if ot.get("insufficient_data"):
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            ax.set_title(f"{model_names.get(target, target)}", fontsize=12)
            continue

        # Bar chart comparing accuracies
        categories = []
        accuracies = []
        counts = []
        colors = []

        if ot.get("rest_accuracy") is not None:
            categories.append(f"Short/Med\n(n={ot['n_rest']})")
            accuracies.append(ot["rest_accuracy"])
            counts.append(ot["n_rest"])
            colors.append("C0")

        if ot.get("long_high_conf_accuracy") is not None:
            categories.append(f"Long + High Conf\n(n={ot['n_long_high_conf']})")
            accuracies.append(ot["long_high_conf_accuracy"])
            counts.append(ot["n_long_high_conf"])
            colors.append("C2")

        if ot.get("overthinking_accuracy") is not None:
            categories.append(f"Overthinking\n(n={ot['n_overthinking']})")
            accuracies.append(ot["overthinking_accuracy"])
            counts.append(ot["n_overthinking"])
            colors.append("C3")

        if not categories:
            ax.text(0.5, 0.5, "No data to plot", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(f"{model_names.get(target, target)}", fontsize=12)
            continue

        x = np.arange(len(categories))
        bars = ax.bar(x, accuracies, color=colors, edgecolor="black", linewidth=0.5,
                      alpha=0.85)

        for xi, acc in enumerate(accuracies):
            ax.text(xi, acc + 0.02, f"{acc:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

        # Annotate: overall long accuracy
        if ot.get("long_accuracy") is not None:
            ax.axhline(y=ot["long_accuracy"], color="gray", linestyle="--", alpha=0.6,
                       linewidth=1.5)
            ax.text(len(categories) - 0.5, ot["long_accuracy"] + 0.02,
                    f"All long: {ot['long_accuracy']:.3f}", fontsize=8,
                    ha="right", color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(
            f"{model_names.get(target, target)} — Overthinking Detection\n"
            f"(long = top quartile, >{ot['q75_token_threshold']} tokens; "
            f"overthinking = long + p_correct < 0.3)",
            fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("UC-D: Overthinking Detection — Long Responses with Low Calibrator Confidence",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UC-D: Agent Step Verification — Stage 1 (Step-Truncation Confidence Analysis)")
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2",
                        help="Directory with scored JSONL files")
    parser.add_argument("--output_dir", default="data/use_cases/results_test_only_v2",
                        help="Directory for results JSON")
    parser.add_argument("--fig_dir", default="figures/use_cases_v2",
                        help="Directory for output figures")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Only use first 50 samples per model")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    targets = ["gpt5mini", "gpt52", "qwen35"]
    target_names = {
        "gpt5mini": "GPT-5-mini (in-dist)",
        "gpt52": "GPT-5.2 (cross-model)",
        "qwen35": "Qwen3.5 (cross-model)",
    }

    all_results = {}

    for target in targets:
        scored_path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if not scored_path.exists():
            print(f"Skipping {target}: {scored_path} not found")
            continue

        samples = load_scored(scored_path)
        if args.smoke_test:
            samples = samples[:50]

        n_total = len(samples)
        base_acc = float(np.mean([s["is_correct"] for s in samples]))

        print(f"\n{'='*70}")
        print(f"UC-D: Agent Step Verification — {target_names.get(target, target)}")
        print(f"{'='*70}")
        print(f"Samples: {n_total}, Base accuracy: {base_acc:.3f}")

        # --- 1. Multi-Step Response Detection ---
        print(f"\n--- 1. Multi-Step Response Detection ---")
        ms_overall, ms_by_bench = multi_step_summary(samples)

        ms_total = ms_overall.get("total", 1)
        print(f"  Multi-step detected:  {ms_overall.get('is_multi_step', 0):>5} "
              f"({ms_overall.get('is_multi_step', 0)/ms_total:.1%})")
        print(f"    JSON reasoning:     {ms_overall.get('has_json_reasoning', 0):>5} "
              f"({ms_overall.get('has_json_reasoning', 0)/ms_total:.1%})")
        print(f"    Numbered steps:     {ms_overall.get('has_numbered_steps', 0):>5} "
              f"({ms_overall.get('has_numbered_steps', 0)/ms_total:.1%})")
        print(f"    Numbered list:      {ms_overall.get('has_numbered_list', 0):>5} "
              f"({ms_overall.get('has_numbered_list', 0)/ms_total:.1%})")
        print(f"    Bullet points:      {ms_overall.get('has_bullets', 0):>5} "
              f"({ms_overall.get('has_bullets', 0)/ms_total:.1%})")
        print(f"    Paragraph breaks:   {ms_overall.get('has_paragraphs', 0):>5} "
              f"({ms_overall.get('has_paragraphs', 0)/ms_total:.1%})")
        print(f"    Long (>500 tokens): {ms_overall.get('long_response', 0):>5} "
              f"({ms_overall.get('long_response', 0)/ms_total:.1%})")

        print(f"\n  Per-benchmark multi-step fraction (top 10):")
        bench_ms = []
        for bench, bd in ms_by_bench.items():
            bt = bd.get("total", 1)
            frac = bd.get("is_multi_step", 0) / bt
            bench_ms.append((bench, frac, bd.get("is_multi_step", 0), bt))
        bench_ms.sort(key=lambda x: -x[1])
        print(f"  {'Benchmark':<22} {'Multi-step':>10} {'Total':>8} {'Fraction':>10}")
        print(f"  {'-'*54}")
        for bench, frac, n_ms, bt in bench_ms[:10]:
            print(f"  {bench:<22} {n_ms:>10} {bt:>8} {frac:>10.1%}")

        # --- 2. Length-Correctness Analysis ---
        print(f"\n--- 2. Length-Correctness Analysis (by output_tokens quartile) ---")
        quartiles = length_quartile_analysis(samples)

        if quartiles:
            print(f"  {'Quartile':>8} {'Tokens':>18} {'N':>6} "
                  f"{'Actual Acc':>11} {'Mean P(c)':>10} {'Gap':>8}")
            print(f"  {'-'*65}")
            for q in quartiles:
                print(f"  Q{q['quartile']:>6} [{q['token_lo']:>6}-{q['token_hi']:>6}] "
                      f"{q['n_samples']:>6} "
                      f"{q['actual_accuracy']:>11.3f} "
                      f"{q['mean_p_correct']:>10.3f} "
                      f"{q['calibrator_accuracy_gap']:>+8.3f}")
        else:
            print("  No quartile data (all output_tokens = 0?)")

        # --- 3. Confidence-Length Correlation ---
        print(f"\n--- 3. Confidence-Length Correlation ---")
        corr = confidence_length_correlation(samples)

        if corr.get("insufficient_data"):
            print(f"  Insufficient data (n={corr.get('n_valid', 0)})")
        else:
            tok_p = corr.get("tokens_vs_p_correct", {})
            tok_c = corr.get("tokens_vs_is_correct", {})
            pc_c = corr.get("p_correct_vs_is_correct", {})
            partial = corr.get("partial_p_correct_vs_is_correct_given_length", {})

            print(f"  {'Relationship':<50} {'Pearson r':>10} {'p-value':>12} {'Spearman r':>12}")
            print(f"  {'-'*86}")
            if tok_p:
                print(f"  {'output_tokens vs p_correct':<50} "
                      f"{tok_p['pearson_r']:>10.4f} {tok_p['pearson_p']:>12.2e} "
                      f"{tok_p['spearman_r']:>12.4f}")
            if tok_c:
                print(f"  {'output_tokens vs is_correct':<50} "
                      f"{tok_c['pearson_r']:>10.4f} {tok_c['pearson_p']:>12.2e} "
                      f"{tok_c['spearman_r']:>12.4f}")
            if pc_c:
                print(f"  {'p_correct vs is_correct':<50} "
                      f"{pc_c['pearson_r']:>10.4f} {pc_c['pearson_p']:>12.2e} "
                      f"{'—':>12}")
            if partial:
                print(f"  {'p_correct vs is_correct | controlling length':<50} "
                      f"{partial['pearson_r']:>10.4f} {partial['pearson_p']:>12.2e} "
                      f"{'—':>12}")

            # Interpretation
            if partial and tok_p:
                raw_r = pc_c.get("pearson_r", 0)
                part_r = partial["pearson_r"]
                if abs(part_r) > 0.01:
                    retained_pct = (part_r / raw_r * 100) if abs(raw_r) > 0.001 else 0
                    print(f"\n  Interpretation: Calibrator retains {retained_pct:.0f}% of its "
                          f"correlation with correctness after controlling for length.")
                    if retained_pct > 80:
                        print(f"  -> Calibrator adds substantial signal BEYOND response length.")
                    elif retained_pct > 50:
                        print(f"  -> Calibrator adds moderate signal beyond response length.")
                    else:
                        print(f"  -> Calibrator may be partially relying on response length.")

        # --- 4. Overthinking Detection ---
        print(f"\n--- 4. 'Overthinking' Detection ---")
        ot = overthinking_analysis(samples)

        if ot.get("insufficient_data"):
            print("  Insufficient data")
        else:
            print(f"  Top-quartile threshold: > {ot['q75_token_threshold']} output tokens")
            print(f"  Long responses:           {ot['n_long']:>5} (accuracy: "
                  f"{ot.get('long_accuracy', 0):.3f})")
            print(f"  Short/medium responses:   {ot['n_rest']:>5} (accuracy: "
                  f"{ot.get('rest_accuracy', 0):.3f})")
            print(f"  Overthinking candidates:  {ot['n_overthinking']:>5} "
                  f"(long + p_correct < 0.3)")

            if ot["overthinking_accuracy"] is not None:
                print(f"    Overthinking accuracy:  {ot['overthinking_accuracy']:.3f}")
                rest_acc = ot.get("rest_accuracy", 0)
                delta = ot["overthinking_accuracy"] - rest_acc
                print(f"    Delta vs rest:          {delta:+.3f}")
                if delta < -0.1:
                    print(f"    -> CONFIRMED: Overthinking candidates are significantly less accurate.")
                elif delta < 0:
                    print(f"    -> Mild signal: Overthinking candidates are somewhat less accurate.")
                else:
                    print(f"    -> No overthinking effect detected (long + low-conf are not worse).")
            else:
                print(f"    No overthinking candidates found (none with p_correct < 0.3).")

            if ot.get("long_high_conf_accuracy") is not None:
                print(f"  Long + high-conf accuracy: {ot['long_high_conf_accuracy']:.3f} "
                      f"(n={ot['n_long_high_conf']})")

            if ot.get("per_benchmark"):
                print(f"\n  Top benchmarks by overthinking count:")
                sorted_bench = sorted(ot["per_benchmark"].items(),
                                      key=lambda x: -x[1]["n_overthinking"])
                print(f"  {'Benchmark':<22} {'N overth.':>10} {'Accuracy':>10} {'% of bench':>12}")
                print(f"  {'-'*58}")
                for bench, bd in sorted_bench[:8]:
                    print(f"  {bench:<22} {bd['n_overthinking']:>10} "
                          f"{bd['accuracy']:>10.3f} {bd['frac_of_benchmark']:>12.1%}")

        # --- 5. Verbalized vs Calibrator Disagreement ---
        print(f"\n--- 5. Verbalized vs Calibrator Disagreement (Long Responses) ---")
        disagree = verbalized_calibrator_disagreement(samples)

        if disagree.get("insufficient_data"):
            print(f"  Insufficient data (n_long_with_verbalized="
                  f"{disagree.get('n_long_with_verbalized', 0)})")
        else:
            print(f"  Median token threshold: {disagree['median_token_threshold']:.0f}")
            print(f"  Long responses:         {disagree['n_long']}")
            print(f"  Long with verb. conf.:  {disagree['n_long_with_verbalized']}")
            print(f"  Confident but flagged:  {disagree['n_confident_but_flagged']} "
                  f"(verb > 0.7 AND p_correct < 0.3)")
            print(f"  Fraction flagged:       {disagree['frac_confident_but_flagged']:.1%}")

            if disagree.get("confident_but_flagged_accuracy") is not None:
                print(f"  Flagged accuracy:       {disagree['confident_but_flagged_accuracy']:.3f}")
                if disagree["confident_but_flagged_accuracy"] < 0.3:
                    print(f"  -> Step verification value: The calibrator catches confident-but-wrong "
                          f"long responses that verbalized confidence misses.")
                elif disagree["confident_but_flagged_accuracy"] < 0.5:
                    print(f"  -> Moderate value: Some of these flagged responses are indeed wrong.")
                else:
                    print(f"  -> Limited value: Flagged responses are actually mostly correct "
                          f"(calibrator may be too conservative on long responses).")

            if disagree.get("both_high_accuracy") is not None:
                print(f"  Both high (verb>0.7, p>0.7): n={disagree['n_both_high']}, "
                      f"accuracy={disagree['both_high_accuracy']:.3f}")
            if disagree.get("both_low_accuracy") is not None:
                print(f"  Both low (verb<0.3, p<0.3):  n={disagree['n_both_low']}, "
                      f"accuracy={disagree['both_low_accuracy']:.3f}")

        # --- 6. Stage 2 Feasibility ---
        print(f"\n--- 6. Stage 2 Feasibility Check ---")
        feasibility = stage2_feasibility(samples)

        print(f"  Responses > 500 tokens:    {feasibility['n_candidates_500_plus']:>6} "
              f"({feasibility['frac_candidates_500_plus']:.1%})")
        print(f"  Responses > 1000 tokens:   {feasibility['n_candidates_1000_plus']:>6}")
        print(f"  Responses > 2000 tokens:   {feasibility['n_candidates_2000_plus']:>6}")
        print(f"  Truncation points/resp:    {feasibility['truncation_points_per_response']}")
        print(f"  Total inference calls:     {feasibility['total_inference_calls']:>6}")
        print(f"  Est. time (1 GPU):         {feasibility['estimated_hours_single_gpu']:.1f} hours "
              f"({feasibility['estimated_minutes_single_gpu']:.0f} min)")

        if feasibility.get("candidate_token_stats"):
            cs = feasibility["candidate_token_stats"]
            print(f"  Candidate token stats:     mean={cs['mean']:.0f}, "
                  f"median={cs['median']:.0f}, max={cs['max']}")

        if feasibility["n_candidates_500_plus"] >= 100:
            print(f"  -> FEASIBLE: Enough candidates for meaningful Stage 2 analysis.")
        elif feasibility["n_candidates_500_plus"] >= 30:
            print(f"  -> MARGINAL: Limited candidates, but may still yield insights.")
        else:
            print(f"  -> INSUFFICIENT: Too few long responses for Stage 2.")

        # Store results
        all_results[target] = {
            "n_samples": n_total,
            "base_accuracy": base_acc,
            "multi_step_overall": {
                k: v for k, v in ms_overall.items() if k != "total"
            },
            "multi_step_by_benchmark": ms_by_bench,
            "length_quartiles": quartiles,
            "correlations": corr,
            "overthinking": ot,
            "verbalized_disagreement": disagree,
            "stage2_feasibility": feasibility,
            "_samples": samples,  # kept for plotting, removed before JSON save
        }

    if not all_results:
        print("\nERROR: No scored data found. Nothing to do.")
        return

    # --- Figures ---
    plot_length_analysis(all_results, f"{args.fig_dir}/uc_d_length_analysis.pdf")
    plot_overthinking(all_results, f"{args.fig_dir}/uc_d_overthinking.pdf")

    # --- Remove _samples before JSON serialization ---
    for target in all_results:
        all_results[target].pop("_samples", None)

    # --- Save JSON ---
    out_path = f"{args.output_dir}/uc_d_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # --- Final Summary ---
    print(f"\n{'='*70}")
    print("UC-D Summary: Agent Step Verification — HONEST NEGATIVE")
    print(f"{'='*70}")
    print()
    print("  NOTE: This use case is reported as an HONEST NEGATIVE / LIMITATION.")
    print("  The calibrator was trained on full Q+A pairs, not intermediate")
    print("  reasoning steps. Step-level AUROC is near random (~0.53).")
    print("  The analysis below characterizes WHERE the calibrator has signal")
    print("  (overthinking detection) and where it does not (step-level).")
    print()

    for target in all_results:
        tname = target_names.get(target, target)
        data = all_results[target]
        corr = data["correlations"]
        ot = data["overthinking"]
        disagree = data["verbalized_disagreement"]
        feasibility = data["stage2_feasibility"]

        print(f"\n  {tname}:")
        print(f"    Samples: {data['n_samples']}, Base accuracy: {data['base_accuracy']:.3f}")

        # Correlation summary
        partial = corr.get("partial_p_correct_vs_is_correct_given_length", {})
        raw = corr.get("p_correct_vs_is_correct", {})
        if partial and raw:
            retained = (partial["pearson_r"] / raw["pearson_r"] * 100) if abs(raw["pearson_r"]) > 0.001 else 0
            print(f"    Calibrator signal beyond length: {retained:.0f}% retained "
                  f"(partial r={partial['pearson_r']:.3f})")

        # Overthinking summary
        if not ot.get("insufficient_data") and ot.get("overthinking_accuracy") is not None:
            print(f"    Overthinking: {ot['n_overthinking']} candidates, "
                  f"accuracy={ot['overthinking_accuracy']:.3f} "
                  f"(vs rest={ot.get('rest_accuracy', 0):.3f})")

        # Disagreement summary
        if not disagree.get("insufficient_data"):
            print(f"    Confident-but-flagged: {disagree.get('n_confident_but_flagged', 0)} responses "
                  f"({disagree.get('frac_confident_but_flagged', 0):.1%})")
            if disagree.get("confident_but_flagged_accuracy") is not None:
                print(f"    Flagged accuracy: {disagree['confident_but_flagged_accuracy']:.3f}")

        # Feasibility summary
        print(f"    Stage 2 candidates (>500 tokens): {feasibility['n_candidates_500_plus']} "
              f"({feasibility['frac_candidates_500_plus']:.1%}), "
              f"est. {feasibility['estimated_hours_single_gpu']:.1f}h GPU")

    print()


if __name__ == "__main__":
    main()
