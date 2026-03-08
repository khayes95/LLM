#!/usr/bin/env python3
"""Demo: Realistic Education/Tutoring Deployment of UQ Calibrator.

Designed around what teachers, tutoring platforms, and students actually need.

Scenarios:
1. WRONG-STEP SHIELD: AI tutor explains a math problem. Before showing the
   student, check if the solution is correct. If not, route to human tutor.
   Metric: What % of wrong solutions reach the student?

2. TEACHER GRADING AID: AI helps grade 200 student papers. Teacher reviews
   the ones the AI is least confident about. Metric: How many grading errors
   does the teacher need to catch manually?

3. DIFFICULTY RADAR: Per-topic breakdown showing where the AI tutor is
   weakest. Teachers use this to know which topics need human coverage.

4. STUDENT TRUST METER: Show the student a simple indicator:
   "I'm confident about this" vs "You might want to double-check this"
   Metric: How often does "confident" = actually correct?

5. PARENT SAFETY REPORT: Weekly summary for parents showing how many
   answers were high/medium/low confidence and what that means.

Usage:
    conda activate uq_eval && python scripts/demo_education_realistic.py
    conda activate uq_eval && python scripts/demo_education_realistic.py --include_unseen
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


def load_all(scored_dir, unseen_dir=None, include_unseen=False):
    all_samples = []
    for fn in sorted(os.listdir(scored_dir)):
        if fn.endswith("_scored.jsonl"):
            all_samples.extend(load_scored(os.path.join(scored_dir, fn)))
    unseen_samples = []
    if include_unseen and unseen_dir and os.path.exists(unseen_dir):
        for fn in sorted(os.listdir(unseen_dir)):
            if fn.endswith("_scored.jsonl"):
                unseen_samples.extend(load_scored(os.path.join(unseen_dir, fn)))
    return all_samples, unseen_samples


# ============================================================
# SCENARIO 1: WRONG-STEP SHIELD
# ============================================================

def wrong_step_shield(samples):
    """Before showing AI solution to student, check with calibrator.

    If calibrator says low confidence → don't show, route to human.
    Key metric: What % of wrong solutions reach the student?
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])
    n = len(labels)
    n_wrong = (labels == 0).sum()
    n_correct = (labels == 1).sum()

    results = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        shown = scores >= threshold  # shown to student
        blocked = ~shown  # routed to human

        # Wrong solutions that reach the student (THE danger metric)
        wrong_shown = (shown & (labels == 0)).sum()
        wrong_shown_rate = wrong_shown / shown.sum() if shown.sum() > 0 else 0

        # Correct solutions blocked unnecessarily (wasted human tutor time)
        correct_blocked = (blocked & (labels == 1)).sum()
        waste_rate = correct_blocked / blocked.sum() if blocked.sum() > 0 else 0

        # Coverage: what % of questions does the AI handle?
        coverage = shown.mean()

        results.append({
            "threshold": threshold,
            "coverage": float(coverage),
            "n_shown": int(shown.sum()),
            "wrong_shown": int(wrong_shown),
            "wrong_shown_rate": float(wrong_shown_rate),
            "correct_blocked": int(correct_blocked),
            "waste_rate": float(waste_rate),
        })

    return {
        "n_total": n,
        "n_wrong": int(n_wrong),
        "base_error_rate": float(n_wrong / n),
        "shields": results,
    }


# ============================================================
# SCENARIO 2: TEACHER GRADING AID
# ============================================================

def grading_aid(samples):
    """AI grades papers. Teacher reviews least-confident grades.

    Without calibrator: teacher reviews all N papers (or random subset).
    With calibrator: teacher reviews bottom K% by confidence.
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])
    n = len(labels)
    n_errors = (labels == 0).sum()

    # Sort by confidence ascending
    order = np.argsort(scores)
    sorted_labels = labels[order]

    # Typical teacher: reviews 20% of AI-graded papers randomly
    rng = np.random.RandomState(42)
    random_review_20 = rng.choice(n, int(n * 0.2), replace=False)
    random_catch = (labels[random_review_20] == 0).sum()

    results = []
    for review_pct in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        k = max(1, int(n * review_pct))
        # Calibrator-guided: review lowest confidence
        errors_in_bottom_k = (sorted_labels[:k] == 0).sum()
        # Random review of same size
        random_catches = []
        for _ in range(100):
            idx = rng.choice(n, k, replace=False)
            random_catches.append((labels[idx] == 0).sum())
        random_mean = np.mean(random_catches)

        results.append({
            "review_pct": review_pct,
            "n_reviewed": k,
            "calibrator_catches": int(errors_in_bottom_k),
            "random_catches": float(random_mean),
            "advantage_ratio": float(errors_in_bottom_k / random_mean) if random_mean > 0 else float("inf"),
            "calibrator_catch_rate": float(errors_in_bottom_k / n_errors) if n_errors > 0 else 0,
        })

    return {
        "n_total": n,
        "n_grading_errors": int(n_errors),
        "base_error_rate": float(n_errors / n),
        "random_20pct_catches": int(random_catch),
        "grading": results,
    }


# ============================================================
# SCENARIO 3: DIFFICULTY RADAR
# ============================================================

def difficulty_radar(samples):
    """Per-topic breakdown: where does the AI tutor fail most?

    Teachers use this to know which topics need human tutors.
    """
    bench_data = defaultdict(lambda: {"correct": 0, "total": 0, "scores": []})
    for s in samples:
        b = s["benchmark"]
        bench_data[b]["total"] += 1
        bench_data[b]["correct"] += s["is_correct"]
        bench_data[b]["scores"].append(s.get("p_correct") or 0.5)

    # Map benchmarks to student-facing topic names
    topic_names = {
        "mathverse": "Geometry & Visual Math",
        "mathvision": "Math Problem Solving (Visual)",
        "mathvista": "Math Reasoning (Charts/Graphs)",
        "omnimath": "Competition Math",
        "gpqa": "Graduate-Level Science",
        "bbeh": "Logic & Reasoning",
        "hle": "Expert Knowledge",
        "simpleqa": "Factual Questions",
        "chembench": "Chemistry",
        "prbench": "Professional Knowledge",
        "livebench": "Current Events & Analysis",
        "healthbench": "Health & Medicine",
        "triviaqa": "Trivia & General Knowledge",
    }

    results = {}
    for b, d in bench_data.items():
        if d["total"] < 5:
            continue
        acc = d["correct"] / d["total"]
        mean_conf = np.mean(d["scores"])
        labels = np.array([1 if s["is_correct"] else 0 for s in samples if s["benchmark"] == b])
        sc = np.array([s.get("p_correct") or 0.5 for s in samples if s["benchmark"] == b])
        auroc = float(roc_auc_score(labels, sc)) if len(set(labels)) >= 2 else float("nan")

        results[b] = {
            "topic": topic_names.get(b, b),
            "accuracy": float(acc),
            "mean_confidence": float(mean_conf),
            "auroc": auroc,
            "n": int(d["total"]),
            "teacher_action": "AI handles" if acc > 0.7 else "Supplement with human" if acc > 0.4 else "Human tutor required",
        }

    return results


# ============================================================
# SCENARIO 4: STUDENT TRUST METER
# ============================================================

def student_trust_meter(samples):
    """Simple binary indicator for students: confident vs not confident.

    "I'm pretty sure about this!" vs "You might want to double-check"
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])

    threshold = 0.7  # confident if > 0.7
    confident = scores >= threshold
    unsure = ~confident

    confident_acc = labels[confident].mean() if confident.sum() > 0 else 0
    unsure_acc = labels[unsure].mean() if unsure.sum() > 0 else 0

    # How misleading is the trust meter?
    # "Confident but wrong" = bad for student
    confident_wrong = (confident & (labels == 0)).sum()
    confident_wrong_rate = confident_wrong / confident.sum() if confident.sum() > 0 else 0

    return {
        "threshold": threshold,
        "confident_pct": float(confident.mean()),
        "confident_accuracy": float(confident_acc),
        "confident_wrong": int(confident_wrong),
        "confident_wrong_rate": float(confident_wrong_rate),
        "unsure_pct": float(unsure.mean()),
        "unsure_accuracy": float(unsure_acc),
        "separation": float(confident_acc - unsure_acc),
    }


# ============================================================
# SCENARIO 5: PARENT SAFETY REPORT
# ============================================================

def parent_report(samples):
    """Weekly summary a parent would understand."""
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])
    n = len(labels)

    report = {
        "total_questions": n,
        "ai_answered_correctly": int(labels.sum()),
        "ai_answered_incorrectly": int((labels == 0).sum()),
        "overall_accuracy": float(labels.mean()),
    }

    # Break into safety categories
    high_conf = scores >= 0.8
    med_conf = (scores >= 0.5) & (scores < 0.8)
    low_conf = scores < 0.5

    report["high_confidence"] = {
        "count": int(high_conf.sum()),
        "accuracy": float(labels[high_conf].mean()) if high_conf.sum() > 0 else 0,
        "description": "AI was confident — these answers are almost always correct",
    }
    report["medium_confidence"] = {
        "count": int(med_conf.sum()),
        "accuracy": float(labels[med_conf].mean()) if med_conf.sum() > 0 else 0,
        "description": "AI was somewhat unsure — your child should double-check these",
    }
    report["low_confidence"] = {
        "count": int(low_conf.sum()),
        "accuracy": float(labels[low_conf].mean()) if low_conf.sum() > 0 else 0,
        "description": "AI flagged these as unreliable — a human tutor reviewed them",
    }

    return report


# ============================================================
# PLOTTING
# ============================================================

def plot_education_dashboard(shield, grading, radar, trust, parent, fig_dir, tag=""):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Panel 1: Wrong-step shield
    ax = fig.add_subplot(gs[0, 0])
    shields = shield["shields"]
    thresholds = [s["threshold"] for s in shields]
    wrong_rates = [s["wrong_shown_rate"] for s in shields]
    coverages = [s["coverage"] for s in shields]

    ax.plot(thresholds, wrong_rates, "o-", color="#e74c3c", linewidth=2.5, label="Error rate shown to student")
    ax2 = ax.twinx()
    ax2.plot(thresholds, coverages, "s--", color="#3498db", linewidth=2, label="AI coverage")
    ax.axhline(y=shield["base_error_rate"], color="gray", linestyle=":",
               label=f"No filter: {shield['base_error_rate']:.0%} error rate")
    ax.set_xlabel("Confidence Threshold", fontsize=10)
    ax.set_ylabel("Error Rate Reaching Student", fontsize=10, color="#e74c3c")
    ax2.set_ylabel("AI Coverage (% questions handled)", fontsize=10, color="#3498db")
    ax.set_title("(a) Wrong-Step Shield", fontsize=11)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    ax.set_ylim(0, max(wrong_rates) * 1.3)
    ax.grid(alpha=0.2)

    # Panel 2: Grading aid — calibrator vs random review
    ax = fig.add_subplot(gs[0, 1])
    rows = grading["grading"]
    rpcts = [r["review_pct"] for r in rows]
    cal_catches = [r["calibrator_catches"] for r in rows]
    rand_catches = [r["random_catches"] for r in rows]
    ax.bar([x - 0.015 for x in rpcts], cal_catches, width=0.025, color="#2ecc71",
           label="Calibrator-guided review")
    ax.bar([x + 0.015 for x in rpcts], rand_catches, width=0.025, color="#95a5a6",
           label="Random review (same effort)")
    ax.set_xlabel("Fraction of Papers Reviewed", fontsize=10)
    ax.set_ylabel("Grading Errors Caught", fontsize=10)
    ax.set_title(f"(b) Grading Aid: Calibrator vs Random\n"
                 f"(total errors: {grading['n_grading_errors']})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis="y")

    # Panel 3: Difficulty radar
    ax = fig.add_subplot(gs[1, 0])
    topics = sorted(radar.items(), key=lambda x: x[1]["accuracy"])
    names = [radar[t[0]]["topic"][:20] for t in topics]
    accs = [t[1]["accuracy"] for t in topics]
    colors_list = ["#e74c3c" if a < 0.4 else "#f39c12" if a < 0.7 else "#27ae60" for a in accs]
    bars = ax.barh(range(len(names)), accs, color=colors_list, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("AI Accuracy", fontsize=10)
    ax.set_title("(c) Difficulty Radar: Where AI Needs Help", fontsize=11)
    ax.axvline(x=0.7, color="black", linestyle="--", alpha=0.3)
    ax.axvline(x=0.4, color="black", linestyle="--", alpha=0.3)
    for i, (acc, t) in enumerate(zip(accs, topics)):
        action = t[1]["teacher_action"]
        ax.text(acc + 0.02, i, f"{acc:.0%} — {action}", va="center", fontsize=7)

    # Panel 4: Student trust meter
    ax = fig.add_subplot(gs[1, 1])
    categories = ["AI says:\n'I'm confident!'", "AI says:\n'Double-check this'"]
    accuracies = [trust["confident_accuracy"], trust["unsure_accuracy"]]
    pcts = [trust["confident_pct"], trust["unsure_pct"]]
    colors2 = ["#27ae60", "#f39c12"]
    bars = ax.bar(categories, accuracies, color=colors2, edgecolor="white", width=0.5)
    for i, (bar, acc, pct) in enumerate(zip(bars, accuracies, pcts)):
        ax.text(i, acc + 0.02,
                f"{acc:.0%} accurate\n({pct:.0%} of answers)",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Actual Accuracy", fontsize=10)
    ax.set_title(f"(d) Student Trust Meter\n"
                 f"({trust['confident_wrong']} 'confident' answers were wrong — "
                 f"{trust['confident_wrong_rate']:.1%} false confidence)", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.2, axis="y")

    # Panel 5: Parent report (text)
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    hc = parent["high_confidence"]
    mc = parent["medium_confidence"]
    lc = parent["low_confidence"]
    report_text = f"""WEEKLY PARENT REPORT — AI Tutoring Safety Summary
{'─' * 60}
This week your child asked {parent['total_questions']} questions to the AI tutor.

  ● {hc['count']} answers ({hc['count']/parent['total_questions']:.0%}) were HIGH confidence → {hc['accuracy']:.0%} correct
    {hc['description']}

  ● {mc['count']} answers ({mc['count']/parent['total_questions']:.0%}) were MEDIUM confidence → {mc['accuracy']:.0%} correct
    {mc['description']}

  ● {lc['count']} answers ({lc['count']/parent['total_questions']:.0%}) were LOW confidence → {lc['accuracy']:.0%} correct
    {lc['description']}

Overall accuracy: {parent['overall_accuracy']:.0%}  |  {parent['ai_answered_incorrectly']} answers were incorrect."""

    ax.text(0.05, 0.95, report_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.9))

    plt.suptitle(f"AI Tutoring Quality Dashboard{tag}", fontsize=14, fontweight="bold")
    suffix = tag.replace(" ", "_").replace("(", "").replace(")", "").lower()
    path = os.path.join(fig_dir, f"education_dashboard{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2")
    parser.add_argument("--unseen_dir", default="data/use_cases/scored_unseen")
    parser.add_argument("--output_dir", default="data/use_cases/education_realistic")
    parser.add_argument("--fig_dir", default="figures/education_realistic")
    parser.add_argument("--include_unseen", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    in_dist, unseen = load_all(args.scored_dir, args.unseen_dir, args.include_unseen)
    if args.smoke_test:
        in_dist = in_dist[:50]
        unseen = unseen[:50]

    # Filter to education-relevant benchmarks
    edu_benchmarks = {"mathverse", "mathvision", "mathvista", "omnimath",
                      "gpqa", "bbeh", "hle", "chembench"}
    in_dist_edu = [s for s in in_dist if s["benchmark"] in edu_benchmarks]

    datasets = [("In-Distribution", in_dist_edu)]
    if unseen:
        datasets.append(("Out-of-Distribution", unseen))

    all_output = {}

    for name, samples in datasets:
        print(f"\n{'='*60}")
        print(f"  {name}: {len(samples)} samples")
        print(f"{'='*60}")

        if len(samples) < 10:
            print("  Too few samples, skipping.")
            continue

        labels = np.array([s["is_correct"] for s in samples])
        scores = np.array([s.get("p_correct") or 0.5 for s in samples])
        base_acc = labels.mean()
        auroc = roc_auc_score(labels, scores) if len(set(labels)) >= 2 else float("nan")
        print(f"  Base accuracy: {base_acc:.1%} | AUROC: {auroc:.3f}")

        print("\n--- Scenario 1: Wrong-Step Shield ---")
        shield = wrong_step_shield(samples)
        print(f"  Without filter: {shield['base_error_rate']:.0%} of answers are wrong")
        print(f"  {'Threshold':>10} {'Coverage':>10} {'Error Rate':>12} {'Wrong Shown':>12}")
        for s in shield["shields"]:
            print(f"  {s['threshold']:>10.1f} {s['coverage']:>10.0%} "
                  f"{s['wrong_shown_rate']:>12.1%} {s['wrong_shown']:>12}")

        print("\n--- Scenario 2: Grading Aid ---")
        grading = grading_aid(samples)
        print(f"  Total grading errors: {grading['n_grading_errors']}")
        print(f"  {'Review%':>8} {'Calibrator':>12} {'Random':>10} {'Advantage':>10}")
        for r in grading["grading"]:
            print(f"  {r['review_pct']:>8.0%} {r['calibrator_catches']:>12} "
                  f"{r['random_catches']:>10.0f} {r['advantage_ratio']:>10.1f}x")

        print("\n--- Scenario 3: Difficulty Radar ---")
        radar = difficulty_radar(samples)
        for b in sorted(radar.keys(), key=lambda x: radar[x]["accuracy"]):
            info = radar[b]
            print(f"  {info['topic']:25s} | acc={info['accuracy']:.0%} | "
                  f"auroc={info['auroc']:.3f} | → {info['teacher_action']}")

        print("\n--- Scenario 4: Student Trust Meter ---")
        trust = student_trust_meter(samples)
        print(f"  'Confident': {trust['confident_pct']:.0%} of answers, "
              f"{trust['confident_accuracy']:.0%} accurate "
              f"({trust['confident_wrong']} wrong — {trust['confident_wrong_rate']:.1%} false confidence)")
        print(f"  'Check this': {trust['unsure_pct']:.0%} of answers, "
              f"{trust['unsure_accuracy']:.0%} accurate")
        print(f"  Separation: {trust['separation']:.1%} accuracy gap")

        print("\n--- Scenario 5: Parent Report ---")
        parent = parent_report(samples)
        print(f"  High conf: {parent['high_confidence']['count']} answers, "
              f"{parent['high_confidence']['accuracy']:.0%} correct")
        print(f"  Med conf:  {parent['medium_confidence']['count']} answers, "
              f"{parent['medium_confidence']['accuracy']:.0%} correct")
        print(f"  Low conf:  {parent['low_confidence']['count']} answers, "
              f"{parent['low_confidence']['accuracy']:.0%} correct")

        # Plot
        tag = " (in-distribution)" if "In-Dist" in name else " (out-of-distribution)"
        plot_education_dashboard(shield, grading, radar, trust, parent, args.fig_dir, tag)

        key = "in_dist" if "In-Dist" in name else "ood"
        all_output[key] = {
            "n": len(samples),
            "shield": shield,
            "grading": grading,
            "radar": {k: {kk: vv for kk, vv in v.items()} for k, v in radar.items()},
            "trust": trust,
            "parent": parent,
        }

    out_path = os.path.join(args.output_dir, "education_realistic_results.json")
    with open(out_path, "w") as f:
        json.dump(all_output, f, indent=2,
                  default=lambda x: int(x) if isinstance(x, (np.integer,))
                  else float(x) if isinstance(x, (np.floating,))
                  else bool(x) if isinstance(x, (np.bool_,)) else x)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
