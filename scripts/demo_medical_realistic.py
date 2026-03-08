#!/usr/bin/env python3
"""Demo: Realistic Medical/Clinical Deployment of UQ Calibrator.

Designed around what clinicians, telehealth platforms, and patients need.

Scenarios:
1. CLINICAL TRIAGE: AI answers patient questions. Flag uncertain ones for
   physician review. Metric: What % of wrong medical advice reaches patients?

2. HARM PREVENTION: Wrong medical advice can cause real harm. What's the
   "number needed to review" — how many do you review to prevent one bad answer?

3. TELEHEALTH ROUTING: Route high-confidence questions to AI, medium to nurse,
   low to physician. Cost analysis per routing strategy.

4. PATIENT-FACING SAFETY: Plain language for patients:
   "Based on general information" vs "Please consult your doctor about this"

Usage:
    conda activate uq_eval && python scripts/demo_medical_realistic.py
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


def load_all(scored_dir, unseen_dir=None):
    in_dist, unseen = [], []
    for fn in sorted(os.listdir(scored_dir)):
        if fn.endswith("_scored.jsonl"):
            in_dist.extend(load_scored(os.path.join(scored_dir, fn)))
    if unseen_dir and os.path.exists(unseen_dir):
        for fn in sorted(os.listdir(unseen_dir)):
            if fn.endswith("_scored.jsonl"):
                unseen.extend(load_scored(os.path.join(unseen_dir, fn)))
    return in_dist, unseen


def clinical_triage(samples):
    """Flag uncertain medical answers for physician review."""
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])
    n = len(labels)
    n_wrong = (labels == 0).sum()

    results = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        auto = scores >= threshold
        wrong_auto = (auto & (labels == 0)).sum()
        wrong_auto_rate = wrong_auto / auto.sum() if auto.sum() > 0 else 0
        coverage = auto.mean()

        results.append({
            "threshold": threshold,
            "coverage": float(coverage),
            "wrong_reaching_patient": int(wrong_auto),
            "wrong_rate_in_auto": float(wrong_auto_rate),
            "flagged_for_physician": int((~auto).sum()),
        })

    return {"n": n, "n_wrong": int(n_wrong), "base_error_rate": float(n_wrong / n),
            "triage": results}


def harm_prevention(samples):
    """Number needed to review (NNR) to prevent one wrong medical answer."""
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])

    order = np.argsort(scores)
    sorted_labels = labels[order]

    results = []
    errors_found = 0
    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 0:
            errors_found += 1
        reviews = i + 1
        if errors_found > 0:
            nnr = reviews / errors_found
        else:
            nnr = float("inf")

        if reviews in [10, 25, 50, 100, 150, 200, 250, 300]:
            results.append({
                "n_reviewed": reviews,
                "errors_found": errors_found,
                "nnr": float(nnr),
                "review_pct": float(reviews / len(labels)),
            })

    return results


def telehealth_routing(samples):
    """Route questions: AI → nurse → physician based on confidence.

    Cost model:
    - AI auto-response: $0.10 per question
    - Nurse review: $5 per question
    - Physician review: $25 per question
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])
    n = len(labels)

    # All-physician baseline
    all_physician_cost = n * 25

    strategies = [
        {"name": "All physician", "ai_thresh": 1.01, "nurse_thresh": 1.01},
        {"name": "AI>0.9, else physician", "ai_thresh": 0.9, "nurse_thresh": 1.01},
        {"name": "AI>0.8, nurse>0.5, else physician", "ai_thresh": 0.8, "nurse_thresh": 0.5},
        {"name": "AI>0.7, nurse>0.4, else physician", "ai_thresh": 0.7, "nurse_thresh": 0.4},
        {"name": "AI>0.6, nurse>0.3, else physician", "ai_thresh": 0.6, "nurse_thresh": 0.3},
    ]

    results = []
    for strat in strategies:
        ai_mask = scores >= strat["ai_thresh"]
        nurse_mask = (~ai_mask) & (scores >= strat["nurse_thresh"])
        physician_mask = ~ai_mask & ~nurse_mask

        ai_cost = ai_mask.sum() * 0.10
        nurse_cost = nurse_mask.sum() * 5
        physician_cost = physician_mask.sum() * 25
        total_cost = ai_cost + nurse_cost + physician_cost

        # Errors in AI auto-responses (worst case)
        ai_errors = (ai_mask & (labels == 0)).sum()
        ai_error_rate = ai_errors / ai_mask.sum() if ai_mask.sum() > 0 else 0

        results.append({
            "strategy": strat["name"],
            "ai_pct": float(ai_mask.mean()),
            "nurse_pct": float(nurse_mask.mean()),
            "physician_pct": float(physician_mask.mean()),
            "total_cost": float(total_cost),
            "cost_savings": float(1 - total_cost / all_physician_cost),
            "ai_errors": int(ai_errors),
            "ai_error_rate": float(ai_error_rate),
        })

    return {"all_physician_cost": float(all_physician_cost), "strategies": results}


def patient_safety_labels(samples):
    """Patient-facing labels."""
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])

    tiers = [
        {"name": "General information",
         "message": "This is based on general medical information. "
                    "It is not a substitute for professional medical advice.",
         "lo": 0.8, "hi": 1.01},
        {"name": "Consult your doctor",
         "message": "This information may not be fully accurate for your situation. "
                    "Please discuss with your healthcare provider.",
         "lo": 0.5, "hi": 0.8},
        {"name": "Seek professional help",
         "message": "This response may contain inaccuracies. "
                    "Please consult a qualified healthcare professional.",
         "lo": 0.0, "hi": 0.5},
    ]

    results = []
    for t in tiers:
        mask = (scores >= t["lo"]) & (scores < t["hi"])
        acc = labels[mask].mean() if mask.sum() > 0 else 0
        results.append({
            **t, "n": int(mask.sum()), "pct": float(mask.mean()),
            "accuracy": float(acc),
        })
    return results


def plot_medical_dashboard(triage, nnr, routing, safety, fig_dir, tag=""):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Clinical triage
    ax = axes[0, 0]
    rows = triage["triage"]
    thresholds = [r["threshold"] for r in rows]
    wrong_rates = [r["wrong_rate_in_auto"] for r in rows]
    coverages = [r["coverage"] for r in rows]
    ax.plot(thresholds, wrong_rates, "o-", color="#e74c3c", linewidth=2.5,
            label="Error rate in auto-responses")
    ax2 = ax.twinx()
    ax2.plot(thresholds, coverages, "s--", color="#3498db", linewidth=2,
             label="AI coverage")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Error Rate (patient-facing)", color="#e74c3c")
    ax2.set_ylabel("AI Coverage", color="#3498db")
    ax.set_title("(a) Clinical Triage: Error Rate vs Coverage")
    lines1, l1 = ax.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, l1 + l2, fontsize=8)
    ax.grid(alpha=0.2)

    # Panel 2: NNR curve
    ax = axes[0, 1]
    if nnr:
        reviews = [r["n_reviewed"] for r in nnr]
        nnrs = [r["nnr"] for r in nnr]
        ax.plot(reviews, nnrs, "o-", color="#9b59b6", linewidth=2.5)
        ax.set_xlabel("Number of Answers Reviewed")
        ax.set_ylabel("Number Needed to Review (NNR)\nper error found")
        ax.set_title("(b) Harm Prevention: Review Efficiency")
        ax.grid(alpha=0.2)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_title("(b) Harm Prevention")

    # Panel 3: Telehealth routing cost
    ax = axes[1, 0]
    strats = routing["strategies"]
    names = [s["strategy"][:25] for s in strats]
    costs = [s["total_cost"] for s in strats]
    errors = [s["ai_errors"] for s in strats]
    colors = ["#e74c3c" if e > 0 else "#27ae60" for e in errors]
    bars = ax.barh(range(len(names)), costs, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Total Cost ($)")
    ax.set_title("(c) Telehealth Routing Cost\n(red = has AI errors)")
    for i, (cost, err) in enumerate(zip(costs, errors)):
        ax.text(cost + 50, i, f"${cost:,.0f} ({err} AI errors)", va="center", fontsize=8)

    # Panel 4: Patient safety labels
    ax = axes[1, 1]
    for i, t in enumerate(safety):
        color = "#27ae60" if t["accuracy"] > 0.9 else "#f39c12" if t["accuracy"] > 0.6 else "#e74c3c"
        ax.barh(i, t["pct"], color=color, edgecolor="white", height=0.5)
        ax.text(t["pct"] + 0.02, i,
                f'{t["accuracy"]:.0%} accurate (n={t["n"]})',
                va="center", fontsize=9)
    ax.set_yticks(range(len(safety)))
    ax.set_yticklabels([t["name"] for t in safety], fontsize=9)
    ax.set_xlabel("Fraction of Responses")
    ax.set_title("(d) Patient-Facing Safety Labels")

    plt.suptitle(f"Medical AI Safety Dashboard{tag}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    suffix = tag.replace(" ", "_").replace("(", "").replace(")", "").lower()
    path = os.path.join(fig_dir, f"medical_dashboard{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2")
    parser.add_argument("--unseen_dir", default="data/use_cases/scored_unseen")
    parser.add_argument("--output_dir", default="data/use_cases/medical_realistic")
    parser.add_argument("--fig_dir", default="figures/medical_realistic")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    in_dist, unseen = load_all(args.scored_dir, args.unseen_dir)

    # In-dist: use chembench + mmmu (medical subjects)
    medical_benchmarks = {"chembench", "mmmu", "hle"}
    in_dist_med = [s for s in in_dist if s["benchmark"] in medical_benchmarks]

    # OOD: healthbench is genuinely unseen
    unseen_health = [s for s in unseen if s["benchmark"] == "healthbench"]

    if args.smoke_test:
        in_dist_med = in_dist_med[:50]
        unseen_health = unseen_health[:50]

    datasets = [("In-Distribution (chembench, mmmu, hle)", in_dist_med)]
    if unseen_health:
        datasets.append(("Out-of-Distribution (healthbench — never in training)", unseen_health))

    for name, samples in datasets:
        print(f"\n{'='*60}")
        print(f"  {name}: {len(samples)} samples")
        print(f"{'='*60}")
        if len(samples) < 10:
            print("  Too few samples.")
            continue

        labels = np.array([s["is_correct"] for s in samples])
        scores = np.array([s.get("p_correct") or 0.5 for s in samples])
        auroc = roc_auc_score(labels, scores) if len(set(labels)) >= 2 else float("nan")
        print(f"  Base accuracy: {labels.mean():.1%} | AUROC: {auroc:.3f}")

        triage = clinical_triage(samples)
        print("\n  Clinical Triage:")
        print(f"  {'Threshold':>10} {'Coverage':>10} {'Wrong Rate':>12} {'Flagged':>8}")
        for r in triage["triage"]:
            print(f"  {r['threshold']:>10.1f} {r['coverage']:>10.0%} "
                  f"{r['wrong_rate_in_auto']:>12.1%} {r['flagged_for_physician']:>8}")

        nnr = harm_prevention(samples)
        if nnr:
            print("\n  Harm Prevention (NNR):")
            for r in nnr:
                print(f"    Review {r['n_reviewed']}: found {r['errors_found']} errors "
                      f"(NNR={r['nnr']:.1f})")

        routing = telehealth_routing(samples)
        print("\n  Telehealth Routing:")
        for s in routing["strategies"]:
            print(f"    {s['strategy']:40s} | ${s['total_cost']:>8,.0f} | "
                  f"savings={s['cost_savings']:.0%} | AI errors={s['ai_errors']}")

        safety = patient_safety_labels(samples)
        print("\n  Patient Safety Labels:")
        for t in safety:
            print(f"    {t['name']:25s} | {t['pct']:.0%} of answers | "
                  f"accuracy={t['accuracy']:.0%}")

        tag = " (in-distribution)" if "In-Dist" in name else " (healthbench OOD)"
        plot_medical_dashboard(triage, nnr, routing, safety, args.fig_dir, tag)

    out_path = os.path.join(args.output_dir, "medical_realistic_results.json")
    with open(out_path, "w") as f:
        json.dump({"status": "complete"}, f)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
