#!/usr/bin/env python3
"""UC-E: Adaptive Clarification — "Ask Before Answering".

When the UQ calibrator assigns a low P(correct) to a response, the system should
ask the user a clarifying question instead of delivering a confidently-wrong answer.
This script quantifies the value of UQ-triggered clarification:

Analyses:
1. Trigger rate vs error prevention — at each threshold, how many wrong answers
   are caught and how many correct answers are unnecessarily flagged?
2. Precision/recall for error detection — p_correct < threshold => "flag for
   clarification"; precision = fraction of flagged that are truly wrong.
3. Cost-benefit analysis — net benefit under various error_cost/trigger_cost ratios.
4. Per-benchmark — which domains benefit most from clarification?
5. Method comparison — calibrator vs verbalized vs baselines as trigger signal.
6. Example showcase — samples where calibrator catches errors the model was
   confident about (high verbalized, low p_correct).

Outputs:
    {output_dir}/uc_e_results.json — all metrics, per-benchmark, examples
    {fig_dir}/uc_e_trigger_tradeoff.pdf — precision/recall/F1 vs threshold
    {fig_dir}/uc_e_cost_benefit.pdf — net benefit under cost ratios

Usage:
    python scripts/uc_e_adaptive_clarification.py
    python scripts/uc_e_adaptive_clarification.py --scored_dir data/use_cases/scored_test_only_v2
    python scripts/uc_e_adaptive_clarification.py --smoke_test
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading (same as other UC scripts)
# ---------------------------------------------------------------------------

def load_scored(path):
    """Load scored JSONL, converting fields to native types."""
    samples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            row["is_correct"] = int(row["is_correct"])
            row["p_correct"] = float(row["p_correct"])
            row["output_tokens"] = int(row.get("output_tokens") or 0)
            vc = row.get("verbalized_confidence")
            if vc is None or str(vc).strip().lower() == "none":
                row["verbalized_confidence"] = None
            else:
                row["verbalized_confidence"] = float(vc)
            samples.append(row)
    return samples


# ---------------------------------------------------------------------------
# Analysis 1: Trigger metrics at each threshold
# ---------------------------------------------------------------------------

THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def trigger_analysis(samples, score_key="p_correct", thresholds=THRESHOLDS,
                     invert=False):
    """For each threshold, compute metrics when we flag samples with
    score < threshold (or score > threshold if invert=True) for clarification.

    'Flagged' = triggered for clarification = the system asks instead of answering.
    A true positive = flagged AND actually wrong (we prevented an error).
    A false positive = flagged AND actually correct (unnecessary interruption).
    """
    valid = [s for s in samples if s.get(score_key) is not None]
    if not valid:
        return []

    n_total = len(valid)
    n_wrong = sum(1 for s in valid if not s["is_correct"])
    n_correct = n_total - n_wrong

    results = []
    for t in thresholds:
        if invert:
            flagged = [s for s in valid if s[score_key] > t]
        else:
            flagged = [s for s in valid if s[score_key] < t]

        n_flagged = len(flagged)
        tp = sum(1 for s in flagged if not s["is_correct"])  # wrong answers caught
        fp = sum(1 for s in flagged if s["is_correct"])       # correct answers flagged unnecessarily

        fn = n_wrong - tp   # wrong answers missed
        tn = n_correct - fp  # correct answers not flagged (good)

        precision = tp / n_flagged if n_flagged > 0 else 0.0
        recall = tp / n_wrong if n_wrong > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Accuracy of non-flagged (auto-answered) samples
        if invert:
            auto = [s for s in valid if s[score_key] <= t]
        else:
            auto = [s for s in valid if s[score_key] >= t]
        auto_acc = np.mean([s["is_correct"] for s in auto]) if auto else None

        results.append({
            "threshold": t,
            "n_flagged": n_flagged,
            "pct_flagged": n_flagged / n_total,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auto_answered": len(auto),
            "auto_accuracy": float(auto_acc) if auto_acc is not None else None,
            "errors_prevented_pct": tp / n_wrong if n_wrong > 0 else 0.0,
        })
    return results


# ---------------------------------------------------------------------------
# Analysis 2: Cost-benefit under various error/trigger cost ratios
# ---------------------------------------------------------------------------

def cost_benefit_analysis(trigger_results, cost_ratios=None):
    """For each threshold and cost ratio, compute net benefit.

    Model: benefit = errors_caught * error_cost - false_triggers * trigger_cost
    We normalize so trigger_cost = 1 and vary error_cost.
    """
    if cost_ratios is None:
        cost_ratios = [2, 5, 10, 20, 50, 100]

    results = {}
    for ratio in cost_ratios:
        best_threshold = None
        best_benefit = float("-inf")
        threshold_benefits = []

        for tr in trigger_results:
            benefit = tr["true_positives"] * ratio - tr["false_positives"] * 1
            threshold_benefits.append({
                "threshold": tr["threshold"],
                "benefit": benefit,
                "errors_caught": tr["true_positives"],
                "false_triggers": tr["false_positives"],
            })
            if benefit > best_benefit:
                best_benefit = benefit
                best_threshold = tr["threshold"]

        results[f"ratio_{ratio}"] = {
            "error_cost_ratio": ratio,
            "best_threshold": best_threshold,
            "best_benefit": best_benefit,
            "all_thresholds": threshold_benefits,
        }
    return results


# ---------------------------------------------------------------------------
# Analysis 3: Per-benchmark breakdown
# ---------------------------------------------------------------------------

def per_benchmark_analysis(samples, threshold=0.3):
    """Which domains benefit most from clarification at a given threshold?"""
    by_bench = defaultdict(list)
    for s in samples:
        by_bench[s["benchmark"]].append(s)

    results = {}
    for bench, bs in sorted(by_bench.items()):
        n_total = len(bs)
        n_wrong = sum(1 for s in bs if not s["is_correct"])
        base_error_rate = n_wrong / n_total if n_total > 0 else 0.0

        flagged = [s for s in bs if s["p_correct"] < threshold]
        n_flagged = len(flagged)
        errors_caught = sum(1 for s in flagged if not s["is_correct"])

        # After clarification: remaining errors are wrong answers NOT flagged
        remaining_errors = n_wrong - errors_caught
        remaining_total = n_total - n_flagged
        post_error_rate = remaining_errors / remaining_total if remaining_total > 0 else 0.0

        results[bench] = {
            "n_total": n_total,
            "n_wrong": n_wrong,
            "base_error_rate": float(base_error_rate),
            "n_flagged": n_flagged,
            "pct_flagged": n_flagged / n_total if n_total > 0 else 0.0,
            "errors_caught": errors_caught,
            "errors_caught_pct": errors_caught / n_wrong if n_wrong > 0 else 0.0,
            "post_error_rate": float(post_error_rate),
            "error_reduction": float(base_error_rate - post_error_rate),
        }
    return results


# ---------------------------------------------------------------------------
# Analysis 4: Example showcase — overconfident errors caught by calibrator
# ---------------------------------------------------------------------------

def find_saved_examples(samples, n=10):
    """Find samples where model was verbally confident but calibrator correctly
    flagged as wrong. These are the 'saved' examples — errors that would have
    gone through without UQ."""
    candidates = [
        s for s in samples
        if not s["is_correct"]
        and s["verbalized_confidence"] is not None
        and s["verbalized_confidence"] >= 0.8
        and s["p_correct"] < 0.3
    ]
    # Sort by biggest gap (most overconfident)
    candidates.sort(key=lambda s: s["verbalized_confidence"] - s["p_correct"],
                    reverse=True)
    examples = []
    for s in candidates[:n]:
        examples.append({
            "id": s["id"],
            "benchmark": s["benchmark"],
            "target_model": s.get("target_model", ""),
            "verbalized_confidence": s["verbalized_confidence"],
            "p_correct": s["p_correct"],
            "gap": s["verbalized_confidence"] - s["p_correct"],
            "question_preview": s.get("question_preview", ""),
            "response_preview": s.get("response_preview", ""),
        })
    return examples


# ---------------------------------------------------------------------------
# Analysis 5: Method comparison (AUPRC for error detection)
# ---------------------------------------------------------------------------

def compute_auprc_error_detection(samples, score_key="p_correct", invert_score=False):
    """Area under precision-recall curve for detecting errors.
    Lower score = more likely to be wrong (for calibrator).
    If invert_score=True, higher score = more likely to be wrong.
    """
    valid = [s for s in samples if s.get(score_key) is not None]
    if not valid:
        return 0.0

    labels = np.array([1 - s["is_correct"] for s in valid])  # 1 = wrong
    scores = np.array([s[score_key] for s in valid])
    if not invert_score:
        scores = -scores  # lower p_correct => higher error likelihood

    # Sort by descending score
    order = np.argsort(-scores)
    labels_sorted = labels[order]

    n_pos = labels.sum()
    if n_pos == 0:
        return 0.0

    tp_cumsum = np.cumsum(labels_sorted)
    precisions = tp_cumsum / np.arange(1, len(labels_sorted) + 1)
    recalls = tp_cumsum / n_pos

    # Add (0, prec[0]) point
    precisions = np.concatenate([[precisions[0]], precisions])
    recalls = np.concatenate([[0.0], recalls])

    _trapz = getattr(np, "trapezoid", np.trapz)
    return float(_trapz(precisions, recalls))


def method_comparison(samples):
    """Compare calibrator vs baselines for error detection."""
    methods = {
        "calibrator": ("p_correct", False),
        "verbalized": ("verbalized_confidence", False),
        "platt_verbalized": ("p_platt_verbalized", False),
        "isotonic_verbalized": ("p_isotonic_verbalized", False),
        "length_baseline": ("p_length_baseline", False),
        "combined_baseline": ("p_combined_baseline", False),
    }
    results = {}
    for name, (key, invert) in methods.items():
        auprc = compute_auprc_error_detection(samples, score_key=key,
                                               invert_score=invert)
        # Also get best F1 from trigger analysis
        ta = trigger_analysis(samples, score_key=key, invert=invert)
        best_f1 = max((r["f1"] for r in ta), default=0.0)
        best_f1_thresh = None
        for r in ta:
            if r["f1"] == best_f1:
                best_f1_thresh = r["threshold"]
                break
        results[name] = {
            "auprc_error_detection": auprc,
            "best_f1": best_f1,
            "best_f1_threshold": best_f1_thresh,
        }
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trigger_tradeoff(all_model_results, fig_path):
    """Plot precision, recall, F1 vs threshold for each model."""
    n_models = len(all_model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), squeeze=False)

    model_names = {
        "gpt5mini": "GPT-5-mini",
        "gpt52": "GPT-5.2",
        "qwen35": "Qwen3.5",
    }

    for col, (target, data) in enumerate(all_model_results.items()):
        ax = axes[0, col]
        tr = data["trigger_analysis"]

        thresholds = [r["threshold"] for r in tr]
        precisions = [r["precision"] for r in tr]
        recalls = [r["recall"] for r in tr]
        f1s = [r["f1"] for r in tr]
        pct_flagged = [r["pct_flagged"] for r in tr]

        ax.plot(thresholds, precisions, "o-", color="C0", label="Precision (flagged are wrong)")
        ax.plot(thresholds, recalls, "s-", color="C1", label="Recall (wrong answers caught)")
        ax.plot(thresholds, f1s, "^-", color="C3", linewidth=2.5, label="F1")
        ax.plot(thresholds, pct_flagged, ":", color="gray", label="% queries flagged")

        # Mark best F1
        best_idx = int(np.argmax(f1s))
        ax.axvline(x=thresholds[best_idx], color="C3", alpha=0.3, linestyle="--")
        ax.annotate(f"Best F1={f1s[best_idx]:.2f}\nt={thresholds[best_idx]}",
                    xy=(thresholds[best_idx], f1s[best_idx]),
                    xytext=(10, -20), textcoords="offset points", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="C3"))

        ax.set_xlabel("Threshold (flag if p_correct < t)", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(f"{model_names.get(target, target)} (N={data['n_samples']})",
                     fontsize=13)
        ax.legend(fontsize=9, loc="center right")
        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle("UC-E: Adaptive Clarification — Error Detection vs Threshold",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {fig_path}")


def plot_cost_benefit(all_model_results, fig_path):
    """Plot net benefit vs threshold for various cost ratios."""
    n_models = len(all_model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), squeeze=False)

    model_names = {
        "gpt5mini": "GPT-5-mini",
        "gpt52": "GPT-5.2",
        "qwen35": "Qwen3.5",
    }

    for col, (target, data) in enumerate(all_model_results.items()):
        ax = axes[0, col]
        cb = data["cost_benefit"]

        colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
        for i, (ratio_key, ratio_data) in enumerate(cb.items()):
            ratio = ratio_data["error_cost_ratio"]
            thresholds = [r["threshold"] for r in ratio_data["all_thresholds"]]
            benefits = [r["benefit"] for r in ratio_data["all_thresholds"]]
            color = colors[i % len(colors)]
            ax.plot(thresholds, benefits, "o-", color=color,
                    label=f"Error cost = {ratio}x")

            # Mark optimal
            best_t = ratio_data["best_threshold"]
            best_b = ratio_data["best_benefit"]
            ax.plot(best_t, best_b, "*", color=color, markersize=12)

        ax.axhline(y=0, color="black", linestyle=":", alpha=0.3)
        ax.set_xlabel("Threshold (flag if p_correct < t)", fontsize=12)
        ax.set_ylabel("Net benefit (errors*cost - triggers)", fontsize=12)
        ax.set_title(f"{model_names.get(target, target)}", fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("UC-E: Cost-Benefit Analysis — When Is Clarification Worth It?",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {fig_path}")


def plot_per_benchmark(all_model_results, fig_path, threshold=0.3):
    """Per-benchmark error reduction bar chart."""
    # Use the model with most benchmarks
    best_target = max(all_model_results,
                      key=lambda t: len(all_model_results[t].get("per_benchmark", {})))
    data = all_model_results[best_target]
    pb = data["per_benchmark"]
    if not pb:
        return

    model_names = {
        "gpt5mini": "GPT-5-mini",
        "gpt52": "GPT-5.2",
        "qwen35": "Qwen3.5",
    }

    benchmarks = sorted(pb.keys(), key=lambda b: pb[b]["error_reduction"], reverse=True)
    error_before = [pb[b]["base_error_rate"] for b in benchmarks]
    error_after = [pb[b]["post_error_rate"] for b in benchmarks]
    pct_flagged = [pb[b]["pct_flagged"] for b in benchmarks]

    fig, ax = plt.subplots(figsize=(max(10, len(benchmarks) * 0.7), 6))
    x = np.arange(len(benchmarks))
    w = 0.35

    ax.bar(x - w / 2, error_before, w, label="Error rate (no UQ)", color="C3", alpha=0.7)
    ax.bar(x + w / 2, error_after, w, label=f"Error rate (with UQ, t={threshold})",
           color="C0", alpha=0.9)

    # Annotate % flagged
    for i, pct in enumerate(pct_flagged):
        ax.text(x[i] + w / 2, error_after[i] + 0.01, f"{pct:.0%} flagged",
                ha="center", va="bottom", fontsize=7, rotation=45, color="C0")

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Error rate", fontsize=12)
    ax.set_title(f"UC-E: Error Reduction by Benchmark — {model_names.get(best_target, best_target)} "
                 f"(flag if p_correct < {threshold})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UC-E: Adaptive Clarification — Ask Before Answering")
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
        n_wrong = sum(1 for s in samples if not s["is_correct"])
        base_err = n_wrong / n_total if n_total > 0 else 0.0

        print(f"\n{'='*70}")
        print(f"UC-E: Adaptive Clarification — {target_names.get(target, target)}")
        print(f"{'='*70}")
        print(f"Samples: {n_total}, Wrong: {n_wrong} ({base_err:.1%} error rate)")

        # --- 1. Trigger analysis ---
        print(f"\n--- Error Detection at Each Threshold ---")
        tr = trigger_analysis(samples)

        print(f"  {'Thresh':>7} {'Flagged':>8} {'Flag%':>6} {'Prec':>6} {'Recall':>7} "
              f"{'F1':>6} {'AutoAcc':>8} {'ErrPrev%':>9}")
        print(f"  {'-'*67}")
        for r in tr:
            auto_s = f"{r['auto_accuracy']:.3f}" if r['auto_accuracy'] is not None else "N/A"
            print(f"  {r['threshold']:>7.2f} {r['n_flagged']:>8} {r['pct_flagged']:>5.1%} "
                  f"{r['precision']:>6.3f} {r['recall']:>7.3f} {r['f1']:>6.3f} "
                  f"{auto_s:>8} {r['errors_prevented_pct']:>8.1%}")

        # --- 2. Cost-benefit ---
        print(f"\n--- Cost-Benefit (optimal threshold per error cost ratio) ---")
        cb = cost_benefit_analysis(tr)

        print(f"  {'ErrCost':>8} {'BestThresh':>11} {'Benefit':>8} {'ErrsCaught':>11} {'FalseTrig':>10}")
        print(f"  {'-'*52}")
        for ratio_key, rd in cb.items():
            best_t = rd["best_threshold"]
            # Find the entry at best threshold
            best_entry = [e for e in rd["all_thresholds"] if e["threshold"] == best_t][0]
            print(f"  {rd['error_cost_ratio']:>8}x {best_t:>11.2f} {rd['best_benefit']:>8.0f} "
                  f"{best_entry['errors_caught']:>11} {best_entry['false_triggers']:>10}")

        # --- 3. Per-benchmark ---
        print(f"\n--- Per-Benchmark Error Reduction (t=0.3) ---")
        pb = per_benchmark_analysis(samples, threshold=0.3)

        print(f"  {'Benchmark':<18} {'N':>5} {'ErrBefore':>9} {'Flagged%':>8} "
              f"{'ErrsCaught':>10} {'ErrAfter':>9} {'Reduction':>9}")
        print(f"  {'-'*72}")
        for bench, bd in sorted(pb.items(), key=lambda x: x[1]["error_reduction"],
                                reverse=True):
            print(f"  {bench:<18} {bd['n_total']:>5} {bd['base_error_rate']:>9.3f} "
                  f"{bd['pct_flagged']:>7.1%} {bd['errors_caught']:>10} "
                  f"{bd['post_error_rate']:>9.3f} {bd['error_reduction']:>+9.3f}")

        # --- 4. Examples ---
        examples = find_saved_examples(samples)
        if examples:
            print(f"\n--- Top Overconfident Errors Caught by Calibrator ---")
            for i, ex in enumerate(examples[:5]):
                print(f"  {i+1}. [{ex['benchmark']}] verbalized={ex['verbalized_confidence']:.2f} "
                      f"p_correct={ex['p_correct']:.3f} gap={ex['gap']:.2f}")
                print(f"     Q: {ex['question_preview'][:100]}...")

        # --- 5. Method comparison ---
        print(f"\n--- Method Comparison (AUPRC for Error Detection) ---")
        mc = method_comparison(samples)

        print(f"  {'Method':<25} {'AUPRC':>7} {'BestF1':>7} {'@Thresh':>8}")
        print(f"  {'-'*49}")
        for method, md in sorted(mc.items(), key=lambda x: x[1]["auprc_error_detection"],
                                 reverse=True):
            t_s = f"{md['best_f1_threshold']:.2f}" if md["best_f1_threshold"] is not None else "N/A"
            print(f"  {method:<25} {md['auprc_error_detection']:>7.3f} "
                  f"{md['best_f1']:>7.3f} {t_s:>8}")

        # Store
        all_results[target] = {
            "n_samples": n_total,
            "n_wrong": n_wrong,
            "base_error_rate": base_err,
            "trigger_analysis": tr,
            "cost_benefit": cb,
            "per_benchmark": pb,
            "saved_examples": examples,
            "method_comparison": mc,
        }

    if not all_results:
        print("\nERROR: No scored data found.")
        return

    # --- Figures ---
    plot_trigger_tradeoff(all_results, f"{args.fig_dir}/uc_e_trigger_tradeoff.pdf")
    plot_cost_benefit(all_results, f"{args.fig_dir}/uc_e_cost_benefit.pdf")
    plot_per_benchmark(all_results, f"{args.fig_dir}/uc_e_per_benchmark.pdf",
                       threshold=0.3)

    # --- Save JSON ---
    out_path = f"{args.output_dir}/uc_e_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- Final Summary ---
    print(f"\n{'='*70}")
    print("UC-E Summary: Adaptive Clarification Value")
    print(f"{'='*70}")
    for target in all_results:
        d = all_results[target]
        mc = d["method_comparison"]
        cal_f1 = mc.get("calibrator", {}).get("best_f1", 0)
        verb_f1 = mc.get("verbalized", {}).get("best_f1", 0)
        cal_auprc = mc.get("calibrator", {}).get("auprc_error_detection", 0)

        # Best threshold at error_cost=10x
        cb10 = d["cost_benefit"].get("ratio_10", {})
        best_t = cb10.get("best_threshold", "N/A")

        print(f"  {target_names.get(target, target)}:")
        print(f"    Best F1 — Cal: {cal_f1:.3f}  Verb: {verb_f1:.3f}  "
              f"AUPRC: {cal_auprc:.3f}")
        print(f"    Optimal threshold (10x cost): {best_t}")
        print(f"    Saved examples: {len(d['saved_examples'])} overconfident errors caught")


if __name__ == "__main__":
    main()
