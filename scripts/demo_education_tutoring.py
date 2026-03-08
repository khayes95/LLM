#!/usr/bin/env python3
"""Demo: AI Tutoring Quality Control via UQ Calibrator.

Shows how an education platform (Khan Academy, Chegg, university help desk)
would use the UQ calibrator to ensure AI tutoring quality. Focuses on math
benchmarks as proxy for STEM homework help.

Use cases:
1. Wrong-step detection: Catch incorrect math solutions before students see them
2. Adaptive difficulty: Route hard questions to human tutors
3. Quality dashboard: Show teachers which topic areas the AI struggles with
4. Student trust calibration: Show students "how sure" the AI tutor is

Usage:
    python scripts/demo_education_tutoring.py
    python scripts/demo_education_tutoring.py --smoke_test
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def load_scored(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def filter_education_benchmarks(samples):
    """Filter to benchmarks relevant to education/tutoring."""
    edu_benchmarks = {"mathverse", "mathvision", "mathvista", "omnimath",
                      "bbeh", "gpqa"}  # STEM reasoning
    return [s for s in samples if s["benchmark"] in edu_benchmarks]


def wrong_step_detection(samples):
    """Simulate detecting wrong math solutions.

    In a tutoring context, every incorrect answer is a 'wrong step' that
    could mislead a student. The calibrator should flag these.
    """
    labels = np.array([s["is_correct"] for s in samples])
    cal_scores = np.array([s.get("p_correct") or 0.5 for s in samples])
    verb_scores = np.array([s.get("verbalized_confidence") or 0.5 for s in samples])

    n_total = len(labels)
    n_wrong = (labels == 0).sum()

    # For each confidence bucket, compute accuracy
    buckets = np.linspace(0, 1, 11)
    bucket_data = []
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        mask = (cal_scores >= lo) & (cal_scores < hi)
        if mask.sum() == 0:
            continue
        bucket_data.append({
            "bucket": f"{lo:.1f}-{hi:.1f}",
            "lo": lo, "hi": hi,
            "n": int(mask.sum()),
            "accuracy": float(labels[mask].mean()),
            "cal_mean": float(cal_scores[mask].mean()),
        })

    # Detection at various thresholds
    detection = []
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        flagged = cal_scores < t
        wrong = labels == 0
        tp = (flagged & wrong).sum()
        fp = (flagged & ~wrong).sum()
        fn = (~flagged & wrong).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        detection.append({
            "threshold": t,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "n_flagged": int(flagged.sum()),
            "pct_flagged": float(flagged.mean()),
        })

    auroc = float(roc_auc_score(labels, cal_scores)) if len(set(labels)) >= 2 else 0
    verb_auroc = float(roc_auc_score(labels, verb_scores)) if len(set(labels)) >= 2 else 0

    return {
        "n_total": n_total,
        "n_wrong": int(n_wrong),
        "base_accuracy": float(labels.mean()),
        "calibrator_auroc": auroc,
        "verbalized_auroc": verb_auroc,
        "calibration_buckets": bucket_data,
        "detection": detection,
    }


def adaptive_routing(samples):
    """Simulate routing: high-confidence → auto-answer, low-confidence → human tutor.

    Shows how many questions can be safely auto-answered vs need human tutors.
    """
    labels = np.array([s["is_correct"] for s in samples])
    cal_scores = np.array([s.get("p_correct") or 0.5 for s in samples])

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    routing = []
    for t in thresholds:
        auto = cal_scores >= t
        human = ~auto
        auto_acc = labels[auto].mean() if auto.sum() > 0 else 0
        human_acc = labels[human].mean() if human.sum() > 0 else 0

        routing.append({
            "threshold": t,
            "auto_pct": float(auto.mean()),
            "auto_accuracy": float(auto_acc),
            "human_pct": float(human.mean()),
            "human_accuracy": float(human_acc),
            "overall_accuracy_if_human_perfect": float(
                auto_acc * auto.mean() + 1.0 * human.mean()
            ),
        })
    return routing


def per_benchmark_difficulty(samples):
    """Show which subject areas the AI struggles with — teacher dashboard."""
    bench_data = defaultdict(lambda: {"correct": 0, "total": 0, "cal_scores": []})
    for s in samples:
        b = s["benchmark"]
        bench_data[b]["total"] += 1
        bench_data[b]["correct"] += s["is_correct"]
        bench_data[b]["cal_scores"].append(s.get("p_correct") or 0.5)

    results = {}
    for b, d in bench_data.items():
        acc = d["correct"] / d["total"]
        mean_conf = np.mean(d["cal_scores"])
        results[b] = {
            "accuracy": float(acc),
            "mean_confidence": float(mean_conf),
            "n": int(d["total"]),
            "overconfident": bool(mean_conf > acc + 0.1),
        }
    return results


def plot_tutoring_dashboard(wrong_step, routing, per_bench, fig_dir):
    """Multi-panel tutoring quality dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Calibration reliability diagram
    ax = axes[0, 0]
    buckets = wrong_step["calibration_buckets"]
    predicted = [b["cal_mean"] for b in buckets]
    observed = [b["accuracy"] for b in buckets]
    sizes = [b["n"] for b in buckets]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    scatter = ax.scatter(predicted, observed, s=[s * 3 for s in sizes],
                        c="#2196F3", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Predicted Confidence", fontsize=11)
    ax.set_ylabel("Actual Accuracy", fontsize=11)
    ax.set_title("(a) Calibration: Does Confidence Match Reality?", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # Panel 2: Wrong-step detection precision/recall
    ax = axes[0, 1]
    det = wrong_step["detection"]
    thresholds = [d["threshold"] for d in det]
    precisions = [d["precision"] for d in det]
    recalls = [d["recall"] for d in det]
    f1s = [d["f1"] for d in det]
    ax.plot(thresholds, recalls, "o-", color="#e74c3c", linewidth=2, label="Recall (errors caught)")
    ax.plot(thresholds, precisions, "s-", color="#2196F3", linewidth=2, label="Precision (flags correct)")
    ax.plot(thresholds, f1s, "^-", color="#2ecc71", linewidth=2, label="F1")
    ax.set_xlabel("Flag Threshold (flag if conf < T)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("(b) Wrong-Step Detection Performance", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 3: Routing efficiency
    ax = axes[1, 0]
    auto_pcts = [r["auto_pct"] for r in routing]
    auto_accs = [r["auto_accuracy"] for r in routing]
    overall_accs = [r["overall_accuracy_if_human_perfect"] for r in routing]
    rthresholds = [r["threshold"] for r in routing]

    ax.plot(rthresholds, auto_pcts, "o-", color="#9b59b6", linewidth=2,
            label="% Auto-Answered")
    ax2 = ax.twinx()
    ax2.plot(rthresholds, auto_accs, "s-", color="#2ecc71", linewidth=2,
             label="Auto-Answer Accuracy")
    ax2.plot(rthresholds, overall_accs, "^--", color="#f39c12", linewidth=2,
             label="Overall (if human=100%)")

    ax.set_xlabel("Routing Threshold", fontsize=11)
    ax.set_ylabel("% Questions Auto-Answered", fontsize=11, color="#9b59b6")
    ax2.set_ylabel("Accuracy", fontsize=11, color="#2ecc71")
    ax.set_title("(c) Adaptive Routing: AI vs Human Tutor", fontsize=12)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center left")

    # Panel 4: Per-benchmark difficulty (teacher dashboard)
    ax = axes[1, 1]
    bench_names = sorted(per_bench.keys())
    accs = [per_bench[b]["accuracy"] for b in bench_names]
    confs = [per_bench[b]["mean_confidence"] for b in bench_names]
    ns = [per_bench[b]["n"] for b in bench_names]
    colors = ["#e74c3c" if per_bench[b]["overconfident"] else "#2ecc71" for b in bench_names]

    ax.scatter(confs, accs, s=[n * 2 for n in ns], c=colors, alpha=0.7, edgecolor="white")
    for i, b in enumerate(bench_names):
        ax.annotate(b, (confs[i], accs[i]), fontsize=8, ha="center", va="bottom")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("Mean Calibrator Confidence", fontsize=11)
    ax.set_ylabel("Actual Accuracy", fontsize=11)
    ax.set_title("(d) Teacher Dashboard: AI Difficulty by Subject", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    plt.suptitle("AI Tutoring Quality Dashboard — Powered by UQ Calibrator",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(fig_dir, "tutoring_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2")
    parser.add_argument("--output_dir", default="data/use_cases/education_demo")
    parser.add_argument("--fig_dir", default="figures/education_demo")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # Load
    all_samples = []
    for fn in sorted(os.listdir(args.scored_dir)):
        if fn.endswith("_scored.jsonl"):
            path = os.path.join(args.scored_dir, fn)
            samples = load_scored(path)
            if args.smoke_test:
                samples = samples[:20]
            all_samples.extend(samples)

    edu_samples = filter_education_benchmarks(all_samples)
    print(f"Total: {len(all_samples)} | Education-relevant: {len(edu_samples)}")

    bench_counts = defaultdict(int)
    for s in edu_samples:
        bench_counts[s["benchmark"]] += 1
    for b, c in sorted(bench_counts.items()):
        print(f"  {b}: {c}")

    # Analyses
    print("\n=== Wrong-Step Detection ===")
    wrong_step = wrong_step_detection(edu_samples)
    print(f"Base accuracy: {wrong_step['base_accuracy']:.1%}")
    print(f"Calibrator AUROC: {wrong_step['calibrator_auroc']:.3f}")
    print(f"Verbalized AUROC: {wrong_step['verbalized_auroc']:.3f}")
    print(f"\nDetection by threshold:")
    print(f"{'Threshold':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'%Flagged':>10}")
    for d in wrong_step["detection"]:
        print(f"{d['threshold']:>10.1f} {d['recall']:>10.1%} {d['precision']:>10.1%} "
              f"{d['f1']:>10.3f} {d['pct_flagged']:>10.1%}")

    print("\n=== Adaptive Routing ===")
    routing = adaptive_routing(edu_samples)
    print(f"{'Threshold':>10} {'Auto%':>10} {'Auto Acc':>10} {'Overall*':>10}")
    for r in routing:
        print(f"{r['threshold']:>10.1f} {r['auto_pct']:>10.1%} {r['auto_accuracy']:>10.1%} "
              f"{r['overall_accuracy_if_human_perfect']:>10.1%}")

    print("\n=== Per-Subject Difficulty ===")
    per_bench = per_benchmark_difficulty(edu_samples)
    for b, info in sorted(per_bench.items(), key=lambda x: x[1]["accuracy"]):
        flag = " ** OVERCONFIDENT" if info["overconfident"] else ""
        print(f"  {b:15s} | acc={info['accuracy']:.1%} | conf={info['mean_confidence']:.3f} "
              f"| n={info['n']}{flag}")

    # Figures
    print("\n=== Generating Figures ===")
    plot_tutoring_dashboard(wrong_step, routing, per_bench, args.fig_dir)

    # Save
    output = {
        "wrong_step_detection": wrong_step,
        "adaptive_routing": routing,
        "per_benchmark": per_bench,
        "narrative": {
            "scenario": "AI-powered math tutoring platform",
            "problem": "AI sometimes gives wrong solutions; students learn incorrect methods",
            "solution": "UQ calibrator catches wrong steps before students see them",
            "key_findings": [
                f"AUROC {wrong_step['calibrator_auroc']:.3f} for detecting wrong math solutions",
                f"At threshold 0.5: catches {wrong_step['detection'][3]['recall']:.0%} of errors "
                f"(precision {wrong_step['detection'][3]['precision']:.0%})",
                f"Routing at 0.8 threshold: {routing[3]['auto_pct']:.0%} auto-answered "
                f"at {routing[3]['auto_accuracy']:.1%} accuracy",
            ],
        },
    }
    out_path = os.path.join(args.output_dir, "education_tutoring_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Narrative
    print("\n" + "=" * 70)
    print("USE CASE NARRATIVE: AI TUTORING QUALITY CONTROL")
    print("=" * 70)
    print(f"""
SCENARIO: Khan Academy integrates GPT-5 as a math tutor.

PROBLEM: GPT-5 gets {1 - wrong_step['base_accuracy']:.0%} of math problems wrong.
Wrong solutions teach students incorrect methods — worse than no help at all.

SOLUTION: Every AI-generated solution passes through the UQ calibrator.
- High confidence (>0.8): Show to student immediately
- Medium (0.5-0.8): Show with "check your work" warning
- Low (<0.5): Route to human tutor

RESULTS ({len(edu_samples)} STEM/math samples):
- Calibrator AUROC: {wrong_step['calibrator_auroc']:.3f}
- At routing threshold 0.8:
  - {routing[3]['auto_pct']:.0%} of questions auto-answered at {routing[3]['auto_accuracy']:.1%} accuracy
  - Remaining {routing[3]['human_pct']:.0%} routed to human tutors
  - Overall accuracy (assuming human=100%): {routing[3]['overall_accuracy_if_human_perfect']:.1%}
- This reduces human tutor workload by {routing[3]['auto_pct']:.0%}
""")


if __name__ == "__main__":
    main()
