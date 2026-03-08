#!/usr/bin/env python3
"""Demo: Legal & Factual Hallucination Detection via UQ Calibrator.

Simulates how a lawyer or pro-se litigant would use the UQ calibrator to detect
unreliable LLM outputs. Uses SimpleQA (factual accuracy) and GPQA (expert-level
knowledge) as proxies for legal research scenarios.

Key insight: When LLMs fabricate facts (wrong citations, incorrect dates,
made-up precedents), the calibrator assigns LOW confidence. This lets users
know which answers need manual verification.

Use cases demonstrated:
1. Hallucination flagging: What % of wrong answers get flagged at various thresholds?
2. Triage workflow: Sort answers by confidence, review only the bottom N%
3. Cost-benefit: How much review time is saved vs. reviewing everything?
4. Pro-se safety: At what confidence threshold is the AI "safe enough" for a non-expert?

Usage:
    python scripts/demo_legal_hallucination.py
    python scripts/demo_legal_hallucination.py --smoke_test
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score


def load_scored(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def filter_factual_benchmarks(samples):
    """Filter to benchmarks most relevant to factual/legal accuracy."""
    # SimpleQA = factual claims (closest to legal fact-checking)
    # GPQA = expert-level knowledge (closest to legal reasoning)
    # HLE = hard expert questions
    # livebench = current knowledge
    factual_benchmarks = {"simpleqa", "gpqa", "hle", "livebench", "bbeh"}
    return [s for s in samples if s["benchmark"] in factual_benchmarks]


def hallucination_detection_analysis(samples):
    """Analyze how well the calibrator detects wrong answers (hallucinations)."""
    labels = np.array([s["is_correct"] for s in samples])
    cal_scores = np.array([s.get("p_correct") or 0.5 for s in samples])
    verb_scores = np.array([s.get("verbalized_confidence") or 0.5 for s in samples])

    n_wrong = (labels == 0).sum()
    n_correct = (labels == 1).sum()
    n_total = len(labels)

    results = {
        "total_samples": n_total,
        "n_correct": int(n_correct),
        "n_wrong": int(n_wrong),
        "base_accuracy": float(labels.mean()),
    }

    # AUROC for both methods
    if len(set(labels)) >= 2:
        results["calibrator_auroc"] = float(roc_auc_score(labels, cal_scores))
        results["verbalized_auroc"] = float(roc_auc_score(labels, verb_scores))

    # Threshold analysis: at various confidence thresholds, what % of wrong answers
    # are correctly flagged (recalled)?
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_analysis = []
    for t in thresholds:
        flagged = cal_scores < t  # flag everything below threshold
        n_flagged = flagged.sum()
        # Of the wrong answers, how many did we catch?
        wrong_mask = labels == 0
        caught = (flagged & wrong_mask).sum()
        recall = caught / n_wrong if n_wrong > 0 else 0
        # Of what we flagged, how many were actually wrong?
        precision = caught / n_flagged if n_flagged > 0 else 0
        # What % of total work needs review?
        review_fraction = n_flagged / n_total

        # Same for verbalized
        v_flagged = verb_scores < t
        v_caught = (v_flagged & wrong_mask).sum()
        v_recall = v_caught / n_wrong if n_wrong > 0 else 0

        threshold_analysis.append({
            "threshold": t,
            "n_flagged": int(n_flagged),
            "review_fraction": float(review_fraction),
            "hallucination_recall": float(recall),
            "flag_precision": float(precision),
            "verbalized_recall": float(v_recall),
        })
    results["threshold_analysis"] = threshold_analysis

    # Triage simulation: if you only review the bottom K% by confidence,
    # what % of errors do you catch?
    triage = []
    order = np.argsort(cal_scores)  # ascending = lowest confidence first
    sorted_labels = labels[order]
    wrong_cumsum = np.cumsum(sorted_labels == 0)
    for review_pct in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        k = max(1, int(n_total * review_pct))
        errors_caught = int(wrong_cumsum[k - 1])
        triage.append({
            "review_pct": review_pct,
            "n_reviewed": k,
            "errors_caught": errors_caught,
            "error_recall": errors_caught / n_wrong if n_wrong > 0 else 0,
            "accuracy_of_unreviewed": float(sorted_labels[k:].mean()) if k < n_total else float("nan"),
        })
    results["triage_analysis"] = triage

    return results


def plot_hallucination_flagging(results, fig_dir):
    """Plot threshold vs recall/precision for hallucination detection."""
    ta = results["threshold_analysis"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Threshold vs hallucination recall (calibrator vs verbalized)
    ax = axes[0]
    thresholds = [t["threshold"] for t in ta]
    cal_recall = [t["hallucination_recall"] for t in ta]
    verb_recall = [t["verbalized_recall"] for t in ta]
    ax.plot(thresholds, cal_recall, "o-", color="#2196F3", linewidth=2, label="UQ Calibrator")
    ax.plot(thresholds, verb_recall, "s--", color="#FF9800", linewidth=2, label="Verbalized Conf.")
    ax.set_xlabel("Flag Threshold (flag if confidence < T)", fontsize=11)
    ax.set_ylabel("Hallucination Recall (% errors caught)", fontsize=11)
    ax.set_title("(a) Error Detection Rate", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # Panel 2: Review fraction vs error recall (triage curve)
    ax = axes[1]
    triage = results["triage_analysis"]
    rpcts = [t["review_pct"] for t in triage]
    recalls = [t["error_recall"] for t in triage]
    ax.plot(rpcts, recalls, "o-", color="#e74c3c", linewidth=2, markersize=8)
    ax.fill_between(rpcts, recalls, alpha=0.1, color="#e74c3c")
    ax.set_xlabel("Fraction of Responses Reviewed\n(lowest confidence first)", fontsize=11)
    ax.set_ylabel("Fraction of Errors Caught", fontsize=11)
    ax.set_title("(b) Triage Efficiency", fontsize=12)
    ax.set_xlim(0, 0.55)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    # Annotate key point
    for t in triage:
        if t["review_pct"] == 0.20:
            ax.annotate(f"Review 20% → catch {t['error_recall']:.0%} errors",
                       xy=(0.20, t["error_recall"]),
                       xytext=(0.30, t["error_recall"] - 0.15),
                       arrowprops=dict(arrowstyle="->", color="black"),
                       fontsize=10, color="#e74c3c")

    # Panel 3: Accuracy of unreviewed content at each triage level
    ax = axes[2]
    accs = [t["accuracy_of_unreviewed"] for t in triage]
    ax.plot(rpcts, accs, "o-", color="#2ecc71", linewidth=2, markersize=8)
    ax.axhline(y=results["base_accuracy"], color="gray", linestyle="--",
               alpha=0.5, label=f"Base accuracy ({results['base_accuracy']:.1%})")
    ax.axhline(y=0.90, color="#e74c3c", linestyle=":", alpha=0.5, label="90% target")
    ax.set_xlabel("Fraction Sent for Review", fontsize=11)
    ax.set_ylabel("Accuracy of Auto-Approved Content", fontsize=11)
    ax.set_title("(c) Safety of Auto-Approved Answers", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.55)
    ax.grid(alpha=0.3)

    plt.suptitle("Legal/Factual Hallucination Detection with UQ Calibrator",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(fig_dir, "legal_hallucination_detection.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_pro_se_safety(results, fig_dir):
    """For pro-se litigants: 'Is this answer safe to rely on?'"""
    ta = results["threshold_analysis"]

    fig, ax = plt.subplots(figsize=(8, 5))

    thresholds = [t["threshold"] for t in ta]
    precisions = [t["flag_precision"] for t in ta]
    review_fracs = [t["review_fraction"] for t in ta]

    # Color by safety level
    ax2 = ax.twinx()
    ax.bar(thresholds, review_fracs, width=0.08, color="#FFE0B2", edgecolor="#FF9800",
           label="% Answers Flagged for Review")
    ax2.plot(thresholds, precisions, "o-", color="#e74c3c", linewidth=2, markersize=8,
             label="Precision (% of flags that are actually wrong)")

    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Fraction of Answers Flagged", fontsize=12, color="#FF9800")
    ax2.set_ylabel("Flag Precision (% actually wrong)", fontsize=12, color="#e74c3c")
    ax.set_title("Pro-Se Safety Guide:\n'Flag anything below this confidence level'", fontsize=13)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(fig_dir, "pro_se_safety_guide.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2")
    parser.add_argument("--output_dir", default="data/use_cases/legal_demo")
    parser.add_argument("--fig_dir", default="figures/legal_demo")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # Load all scored data
    all_samples = []
    for fn in sorted(os.listdir(args.scored_dir)):
        if fn.endswith("_scored.jsonl"):
            path = os.path.join(args.scored_dir, fn)
            samples = load_scored(path)
            if args.smoke_test:
                samples = samples[:20]
            all_samples.extend(samples)

    print(f"Total samples loaded: {len(all_samples)}")

    # Filter to factual benchmarks (proxy for legal research)
    factual = filter_factual_benchmarks(all_samples)
    print(f"Factual/knowledge samples: {len(factual)}")

    benchmarks = defaultdict(int)
    for s in factual:
        benchmarks[s["benchmark"]] += 1
    for b, c in sorted(benchmarks.items()):
        print(f"  {b}: {c}")

    # Run analysis
    print("\n=== Hallucination Detection Analysis ===")
    results = hallucination_detection_analysis(factual)

    print(f"Base accuracy: {results['base_accuracy']:.1%}")
    print(f"Calibrator AUROC: {results.get('calibrator_auroc', 'N/A'):.3f}")
    print(f"Verbalized AUROC: {results.get('verbalized_auroc', 'N/A'):.3f}")

    print("\n--- Threshold Analysis ---")
    print(f"{'Threshold':>10} {'Flagged%':>10} {'Errors Caught':>15} {'Verb. Recall':>15}")
    for t in results["threshold_analysis"]:
        print(f"{t['threshold']:>10.1f} {t['review_fraction']:>10.1%} "
              f"{t['hallucination_recall']:>15.1%} {t['verbalized_recall']:>15.1%}")

    print("\n--- Triage Efficiency ---")
    print(f"{'Review%':>10} {'Errors Caught':>15} {'Auto-Approve Acc':>18}")
    for t in results["triage_analysis"]:
        acc_str = f"{t['accuracy_of_unreviewed']:.1%}" if not np.isnan(t["accuracy_of_unreviewed"]) else "N/A"
        print(f"{t['review_pct']:>10.0%} {t['error_recall']:>15.1%} {acc_str:>18}")

    # Generate figures
    print("\n=== Generating Figures ===")
    plot_hallucination_flagging(results, args.fig_dir)
    plot_pro_se_safety(results, args.fig_dir)

    # Also run on ALL data (not just factual) for comparison
    print("\n=== Full Dataset Analysis (all benchmarks) ===")
    full_results = hallucination_detection_analysis(all_samples)
    print(f"Full dataset AUROC: {full_results.get('calibrator_auroc', 'N/A'):.3f}")

    # Save
    output = {
        "factual_benchmarks": results,
        "full_dataset": full_results,
        "narrative": {
            "scenario": "Legal research assistant / Pro-se litigant tool",
            "problem": "LLMs fabricate case citations, misstate dates, and invent legal precedents",
            "solution": "UQ calibrator flags unreliable answers before they reach the user",
            "key_finding": (
                f"By reviewing only the bottom 20% of answers by calibrator confidence, "
                f"a user catches {results['triage_analysis'][3]['error_recall']:.0%} of all errors. "
                f"The remaining 80% of auto-approved answers have "
                f"{results['triage_analysis'][3]['accuracy_of_unreviewed']:.1%} accuracy."
            ),
        },
    }
    out_path = os.path.join(args.output_dir, "legal_hallucination_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Print narrative
    print("\n" + "=" * 70)
    print("USE CASE NARRATIVE: LEGAL HALLUCINATION DETECTION")
    print("=" * 70)
    print(f"""
SCENARIO: A solo practitioner uses GPT-5 to draft legal memos.

PROBLEM: GPT-5 sometimes fabricates case citations, misstates holdings,
or invents legal tests. The lawyer needs to verify everything, which
defeats the purpose of using AI.

SOLUTION: Run every GPT-5 output through the UQ calibrator. Flag
anything with confidence < 0.5 for manual verification.

RESULTS (on {len(factual)} factual/reasoning samples):
- Calibrator AUROC: {results.get('calibrator_auroc', 0):.3f} (vs {results.get('verbalized_auroc', 0):.3f} for self-reported confidence)
- Review bottom 20% by confidence → catch {results['triage_analysis'][3]['error_recall']:.0%} of errors
- Auto-approved content accuracy: {results['triage_analysis'][3]['accuracy_of_unreviewed']:.1%}
- Time saved: ~80% reduction in manual review workload

FOR PRO-SE LITIGANTS:
- Set threshold at 0.7: anything below should trigger "consult a lawyer"
- At this threshold, {results['threshold_analysis'][4]['hallucination_recall']:.0%} of wrong answers are caught
- Only {results['threshold_analysis'][4]['review_fraction']:.0%} of answers get the warning
""")


if __name__ == "__main__":
    main()
