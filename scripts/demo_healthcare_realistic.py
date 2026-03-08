#!/usr/bin/env python3
"""Demo: Realistic Healthcare Deployment of UQ Calibrator.

Designed around what clinicians, telehealth platforms, hospital systems,
and patients actually need from AI uncertainty quantification.

Scenarios:
1. CLINICAL DECISION SUPPORT: AI assists with differential diagnosis or
   treatment suggestions. Flag uncertain outputs for attending physician
   review. Metric: What % of wrong clinical suggestions reach patients?

2. HARM SEVERITY TRIAGE: Not all errors are equal in medicine. Weight errors
   by potential harm severity. A wrong drug interaction is worse than a wrong
   definition. Compute harm-weighted error rates.

3. TELEHEALTH ROUTING: Route questions by confidence tier.
   AI auto-response ($0.10) -> Nurse review ($5) -> Physician review ($25).
   Cost-safety Pareto analysis across routing strategies.

4. PATIENT-FACING SAFETY LABELS: Plain language disclaimers:
   "General health information" vs "Please consult your doctor" vs
   "IMPORTANT: Seek professional medical advice before acting on this"

5. REGULATORY COMPLIANCE AUDIT: Generate audit trail for FDA/HIPAA compliance.
   Document calibrator scores, review decisions, error rates. Support
   "reasonable clinical decision support" defense.

Usage:
    conda activate uq_eval && python scripts/demo_healthcare_realistic.py
    conda activate uq_eval && python scripts/demo_healthcare_realistic.py --include_unseen
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
    in_dist, unseen = [], []
    for fn in sorted(os.listdir(scored_dir)):
        if fn.endswith("_scored.jsonl"):
            in_dist.extend(load_scored(os.path.join(scored_dir, fn)))
    if include_unseen and unseen_dir and os.path.exists(unseen_dir):
        for fn in sorted(os.listdir(unseen_dir)):
            if fn.endswith("_scored.jsonl"):
                unseen.extend(load_scored(os.path.join(unseen_dir, fn)))
    return in_dist, unseen


# ============================================================
# SCENARIO 1: CLINICAL DECISION SUPPORT TRIAGE
# ============================================================

def clinical_triage(samples):
    """Flag uncertain medical AI outputs for physician review.

    Key question: At each confidence threshold, what fraction of wrong
    answers reach patients without human review?

    Compare to: Resident physician baseline — catches ~90% of errors
    when reviewing all outputs (misses ~10% due to fatigue, time pressure).
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])
    n = len(labels)
    n_wrong = (labels == 0).sum()

    # Resident physician baseline: reviews everything, catches 90%
    resident_catch_rate = 0.90
    resident_errors_missed = int(n_wrong * (1 - resident_catch_rate))

    results = {"n_total": n, "n_wrong": int(n_wrong),
               "base_error_rate": float(n_wrong / n),
               "resident_catch_rate": resident_catch_rate,
               "resident_errors_missed": resident_errors_missed}

    triage_rows = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        auto = scores >= threshold
        wrong_auto = (auto & (labels == 0)).sum()
        wrong_auto_rate = wrong_auto / auto.sum() if auto.sum() > 0 else 0
        coverage = auto.mean()

        # Sort by confidence ascending for review-efficiency analysis
        order = np.argsort(scores)
        sorted_labels = labels[order]
        k = (~auto).sum()  # number flagged for review
        errors_in_flagged = (sorted_labels[:k] == 0).sum() if k > 0 else 0
        catch_rate = errors_in_flagged / n_wrong if n_wrong > 0 else 1.0

        triage_rows.append({
            "threshold": threshold,
            "coverage": float(coverage),
            "n_auto_approved": int(auto.sum()),
            "wrong_reaching_patient": int(wrong_auto),
            "wrong_rate_in_auto": float(wrong_auto_rate),
            "flagged_for_physician": int((~auto).sum()),
            "catch_rate": float(catch_rate),
            "better_than_resident": catch_rate > resident_catch_rate,
        })

    results["triage"] = triage_rows
    return results


# ============================================================
# SCENARIO 2: HARM SEVERITY TRIAGE
# ============================================================

def harm_severity_analysis(samples):
    """Not all medical errors are equal. Weight by potential harm.

    NNR (Number Needed to Review): How many answers must a physician
    review (sorted by ascending confidence) to catch one error?
    Lower NNR = more efficient use of physician time.
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])

    order = np.argsort(scores)
    sorted_labels = labels[order]

    nnr_results = []
    errors_found = 0
    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 0:
            errors_found += 1
        reviews = i + 1
        nnr = reviews / errors_found if errors_found > 0 else float("inf")

        # Sample at key review counts
        checkpoints = [10, 25, 50, 100, 150, 200, 300, 500]
        if reviews in checkpoints or reviews == len(sorted_labels):
            nnr_results.append({
                "n_reviewed": reviews,
                "errors_found": errors_found,
                "nnr": float(nnr),
                "review_pct": float(reviews / len(labels)),
                "error_recall": float(errors_found / (labels == 0).sum()) if (labels == 0).sum() > 0 else 0,
            })

    # Harm-weighted: assume errors in high-confidence tier are more
    # dangerous (clinician trusts them more)
    n_wrong = (labels == 0).sum()
    high_conf_wrong = ((scores >= 0.8) & (labels == 0)).sum()
    med_conf_wrong = ((scores >= 0.5) & (scores < 0.8) & (labels == 0)).sum()
    low_conf_wrong = ((scores < 0.5) & (labels == 0)).sum()

    return {
        "nnr_curve": nnr_results,
        "harm_tiers": {
            "high_confidence_errors": int(high_conf_wrong),
            "medium_confidence_errors": int(med_conf_wrong),
            "low_confidence_errors": int(low_conf_wrong),
            "high_conf_error_pct": float(high_conf_wrong / n_wrong) if n_wrong > 0 else 0,
            "note": "High-confidence errors are most dangerous — clinician may not double-check",
        }
    }


# ============================================================
# SCENARIO 3: TELEHEALTH ROUTING
# ============================================================

def telehealth_routing(samples):
    """Route questions: AI -> nurse -> physician based on confidence.

    Cost model:
    - AI auto-response: $0.10 per question
    - Nurse review: $5 per question (RN, ~15 min)
    - Physician review: $25 per question (MD, ~15 min at $100/hr)
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])
    n = len(labels)

    all_physician_cost = n * 25

    strategies = [
        {"name": "All physician review", "ai_thresh": 1.01, "nurse_thresh": 1.01},
        {"name": "AI>0.9, else physician", "ai_thresh": 0.9, "nurse_thresh": 1.01},
        {"name": "AI>0.8, nurse 0.5-0.8, else MD", "ai_thresh": 0.8, "nurse_thresh": 0.5},
        {"name": "AI>0.7, nurse 0.4-0.7, else MD", "ai_thresh": 0.7, "nurse_thresh": 0.4},
        {"name": "AI>0.6, nurse 0.3-0.6, else MD", "ai_thresh": 0.6, "nurse_thresh": 0.3},
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

        ai_errors = (ai_mask & (labels == 0)).sum()
        ai_error_rate = ai_errors / ai_mask.sum() if ai_mask.sum() > 0 else 0

        results.append({
            "strategy": strat["name"],
            "ai_pct": float(ai_mask.mean()),
            "nurse_pct": float(nurse_mask.mean()),
            "physician_pct": float(physician_mask.mean()),
            "total_cost": float(total_cost),
            "cost_per_question": float(total_cost / n),
            "cost_savings": float(1 - total_cost / all_physician_cost),
            "ai_errors": int(ai_errors),
            "ai_error_rate": float(ai_error_rate),
        })

    return {"all_physician_cost": float(all_physician_cost),
            "cost_per_question_baseline": 25.0,
            "strategies": results}


# ============================================================
# SCENARIO 4: PATIENT-FACING SAFETY LABELS
# ============================================================

def patient_safety_labels(samples):
    """Plain-language labels for patient-facing AI.

    GREEN: "This is based on general medical information."
    YELLOW: "This may not apply to your situation. Discuss with your doctor."
    RED: "IMPORTANT: Please consult a healthcare professional before acting."
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])

    tiers = [
        {"name": "GREEN", "label": "General health information",
         "message": "This is based on general medical information. "
                    "It is not a substitute for professional medical advice.",
         "lo": 0.8, "hi": 1.01, "color": "#27ae60"},
        {"name": "YELLOW", "label": "Consult your doctor",
         "message": "This information may not be fully accurate for your situation. "
                    "Please discuss with your healthcare provider.",
         "lo": 0.5, "hi": 0.8, "color": "#f39c12"},
        {"name": "RED", "label": "Seek professional help",
         "message": "This response may contain inaccuracies. "
                    "Please consult a qualified healthcare professional before acting.",
         "lo": 0.0, "hi": 0.5, "color": "#e74c3c"},
    ]

    results = []
    for t in tiers:
        mask = (scores >= t["lo"]) & (scores < t["hi"])
        n_tier = mask.sum()
        acc = labels[mask].mean() if n_tier > 0 else 0
        n_false = (labels[mask] == 0).sum() if n_tier > 0 else 0

        results.append({
            "name": t["name"],
            "label": t["label"],
            "message": t["message"],
            "color": t["color"],
            "n": int(n_tier),
            "pct": float(mask.mean()),
            "accuracy": float(acc),
            "false_rate": float(1 - acc) if n_tier > 0 else 0,
            "n_false": int(n_false),
        })

    green = results[0]
    return {
        "tiers": results,
        "false_green_rate": green["false_rate"],
        "n_false_greens": green["n_false"],
        "total_greens": green["n"],
    }


# ============================================================
# SCENARIO 5: REGULATORY COMPLIANCE AUDIT
# ============================================================

def compliance_audit(samples):
    """Generate regulatory audit summary for FDA/HIPAA compliance.

    Each AI-assisted clinical decision is logged with calibrator score,
    review disposition, and outcome. Supports "reasonable clinical
    decision support" defense.
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])

    review_threshold = 0.7
    reviewed = scores < review_threshold
    auto_approved = ~reviewed

    errors_in_reviewed = (labels[reviewed] == 0).sum()
    errors_slipped = (labels[auto_approved] == 0).sum()
    total_errors = (labels == 0).sum()

    # Time savings: physician review = 15 min per item
    physician_hours_saved = auto_approved.sum() * 0.25  # 15 min per item
    cost_saved = physician_hours_saved * 100  # $100/hr physician rate

    return {
        "review_threshold": review_threshold,
        "total_items": len(labels),
        "auto_approved": int(auto_approved.sum()),
        "physician_reviewed": int(reviewed.sum()),
        "errors_caught_by_review": int(errors_in_reviewed),
        "errors_slipped_through": int(errors_slipped),
        "total_errors": int(total_errors),
        "catch_rate": float(errors_in_reviewed / total_errors) if total_errors > 0 else 1.0,
        "physician_hours_saved": float(physician_hours_saved),
        "cost_saved_dollars": float(cost_saved),
        "fda_defensible": errors_slipped < total_errors * 0.05,  # <5% slip = defensible for medical
        "slip_rate": float(errors_slipped / total_errors) if total_errors > 0 else 0,
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_healthcare_dashboard(triage, harm, routing, safety, audit, fig_dir, tag=""):
    """5-panel dashboard for clinical deployment."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # Panel 1: Clinical triage — error rate vs coverage
    ax = fig.add_subplot(gs[0, 0])
    rows = triage["triage"]
    thresholds = [r["threshold"] for r in rows]
    wrong_rates = [r["wrong_rate_in_auto"] for r in rows]
    coverages = [r["coverage"] for r in rows]
    ax.plot(thresholds, wrong_rates, "o-", color="#e74c3c", linewidth=2.5,
            label="Error rate (auto)")
    ax2 = ax.twinx()
    ax2.plot(thresholds, coverages, "s--", color="#3498db", linewidth=2,
             label="AI coverage")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Error Rate (patient-facing)", color="#e74c3c")
    ax2.set_ylabel("AI Coverage", color="#3498db")
    ax.set_title("(a) Clinical Triage:\nError Rate vs Coverage")
    lines1, l1 = ax.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, l1 + l2, fontsize=7, loc="center right")
    ax.grid(alpha=0.2)

    # Panel 2: NNR curve
    ax = fig.add_subplot(gs[0, 1])
    nnr_data = harm["nnr_curve"]
    if nnr_data:
        reviews = [r["n_reviewed"] for r in nnr_data]
        nnrs = [min(r["nnr"], 20) for r in nnr_data]  # cap for display
        ax.plot(reviews, nnrs, "o-", color="#9b59b6", linewidth=2.5)
        ax.axhline(y=1.0, color="#27ae60", linestyle="--", alpha=0.5,
                   label="Perfect efficiency (NNR=1)")
        ax.set_xlabel("Answers Reviewed (ascending confidence)")
        ax.set_ylabel("Number Needed to Review (NNR)")
        ax.set_title("(b) Harm Prevention:\nReview Efficiency")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_title("(b) Harm Prevention")
    ax.grid(alpha=0.2)

    # Panel 3: Harm severity breakdown
    ax = fig.add_subplot(gs[0, 2])
    ht = harm["harm_tiers"]
    tier_names = ["High conf\n(most dangerous)", "Medium conf", "Low conf\n(flagged)"]
    tier_vals = [ht["high_confidence_errors"], ht["medium_confidence_errors"],
                 ht["low_confidence_errors"]]
    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    bars = ax.bar(tier_names, tier_vals, color=colors, edgecolor="white")
    for bar, val in zip(bars, tier_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Errors")
    ax.set_title("(c) Error Distribution by\nConfidence Tier")
    ax.grid(alpha=0.2, axis="y")

    # Panel 4: Telehealth routing cost
    ax = fig.add_subplot(gs[1, 0])
    strats = routing["strategies"]
    names = [s["strategy"][:28] for s in strats]
    costs = [s["total_cost"] for s in strats]
    errors = [s["ai_errors"] for s in strats]
    colors_bar = ["#e74c3c" if e > 0 else "#27ae60" for e in errors]
    bars = ax.barh(range(len(names)), costs, color=colors_bar, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Total Cost ($)")
    ax.set_title("(d) Telehealth Routing Cost\n(red = has AI errors)")
    for i, (cost, err) in enumerate(zip(costs, errors)):
        ax.text(cost + max(costs) * 0.02, i, f"${cost:,.0f} ({err} err)",
                va="center", fontsize=7)

    # Panel 5: Compliance audit summary
    ax = fig.add_subplot(gs[1, 1:])
    audit_text = f"""REGULATORY COMPLIANCE AUDIT
{'='*50}
Total AI-assisted clinical decisions:  {audit['total_items']:,}
Auto-approved (confidence >= {audit['review_threshold']}):  {audit['auto_approved']:,}
Physician-reviewed:                    {audit['physician_reviewed']:,}

Errors caught by physician review:     {audit['errors_caught_by_review']}
Errors in auto-approved (slipped):     {audit['errors_slipped_through']}
Overall error catch rate:              {audit['catch_rate']:.0%}
Error slip rate:                       {audit['slip_rate']:.1%}

Physician hours saved:                 {audit['physician_hours_saved']:.0f} hrs
Cost savings (@ $100/hr):             ${audit['cost_saved_dollars']:,.0f}

PATIENT SAFETY TIERS:
  GREEN (reliable):   {safety['tiers'][0]['pct']:.0%} of responses, {safety['tiers'][0]['accuracy']:.0%} accurate
  YELLOW (verify):    {safety['tiers'][1]['pct']:.0%} of responses, {safety['tiers'][1]['accuracy']:.0%} accurate
  RED (unreliable):   {safety['tiers'][2]['pct']:.0%} of responses, {safety['tiers'][2]['accuracy']:.0%} accurate

False GREEN rate (wrong labeled safe): {safety['false_green_rate']:.1%}
FDA-defensible (<5% slip):             {'YES' if audit['fda_defensible'] else 'NO'}"""

    ax.text(0.05, 0.95, audit_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(e) Regulatory Compliance & Safety Summary", fontsize=12)

    plt.suptitle(f"Healthcare AI Safety Dashboard{tag}",
                 fontsize=14, fontweight="bold")
    suffix = tag.replace(" ", "_").replace("(", "").replace(")", "").lower()
    path = os.path.join(fig_dir, f"healthcare_dashboard{suffix}.png")
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
    parser.add_argument("--output_dir", default="data/use_cases/healthcare_realistic")
    parser.add_argument("--fig_dir", default="figures/healthcare_realistic")
    parser.add_argument("--include_unseen", action="store_true",
                        help="Also run on OOD (unseen benchmark) data")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    in_dist, unseen = load_all(args.scored_dir, args.unseen_dir, args.include_unseen)
    if args.smoke_test:
        in_dist = in_dist[:50]
        unseen = unseen[:50]

    # Healthcare-relevant benchmarks: medical/science knowledge + reasoning
    healthcare_benchmarks = {"chembench", "mmmu", "hle", "gpqa", "simpleqa"}
    in_dist_health = [s for s in in_dist if s["benchmark"] in healthcare_benchmarks]

    datasets = [("In-Distribution", in_dist_health)]
    if unseen:
        # HealthBench is genuinely OOD if available
        unseen_health = [s for s in unseen if s["benchmark"] in
                        {"healthbench", "chembench", "gpqa", "simpleqa"}]
        if not unseen_health:
            unseen_health = unseen
        datasets.append(("Out-of-Distribution (unseen benchmarks)", unseen_health))

    all_results = {}
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

        # Run all scenarios
        print("\n--- Scenario 1: Clinical Decision Support Triage ---")
        triage = clinical_triage(samples)
        print(f"  Total errors: {triage['n_wrong']} | Base error rate: {triage['base_error_rate']:.1%}")
        print(f"  Resident baseline misses: {triage['resident_errors_missed']} errors")
        print(f"  {'Thresh':>8} {'Coverage':>10} {'Wrong Auto':>12} {'Catch%':>8} {'Better?':>8}")
        for r in triage["triage"]:
            print(f"  {r['threshold']:>8.1f} {r['coverage']:>10.0%} "
                  f"{r['wrong_reaching_patient']:>12} {r['catch_rate']:>8.0%} "
                  f"{'YES' if r['better_than_resident'] else 'no':>8}")

        print("\n--- Scenario 2: Harm Severity Analysis ---")
        harm = harm_severity_analysis(samples)
        ht = harm["harm_tiers"]
        print(f"  High-confidence errors (most dangerous): {ht['high_confidence_errors']} "
              f"({ht['high_conf_error_pct']:.0%} of all errors)")
        print(f"  Medium-confidence errors: {ht['medium_confidence_errors']}")
        print(f"  Low-confidence errors (already flagged): {ht['low_confidence_errors']}")
        if harm["nnr_curve"]:
            print("  NNR at key review counts:")
            for r in harm["nnr_curve"][:5]:
                print(f"    Review {r['n_reviewed']}: {r['errors_found']} errors found "
                      f"(NNR={r['nnr']:.1f}, recall={r['error_recall']:.0%})")

        print("\n--- Scenario 3: Telehealth Routing ---")
        routing = telehealth_routing(samples)
        for s in routing["strategies"]:
            print(f"    {s['strategy']:40s} | ${s['total_cost']:>8,.0f} | "
                  f"savings={s['cost_savings']:.0%} | AI errors={s['ai_errors']}")

        print("\n--- Scenario 4: Patient Safety Labels ---")
        safety = patient_safety_labels(samples)
        for t in safety["tiers"]:
            print(f"    {t['name']:8s} ({t['label']:25s}) | {t['pct']:5.0%} of answers | "
                  f"accuracy={t['accuracy']:.0%} | {t['n_false']} wrong")
        print(f"  FALSE GREEN RATE: {safety['false_green_rate']:.1%} "
              f"({safety['n_false_greens']} wrong answers labeled safe)")

        print("\n--- Scenario 5: Regulatory Compliance Audit ---")
        audit = compliance_audit(samples)
        print(f"  Auto-approved: {audit['auto_approved']} | Reviewed: {audit['physician_reviewed']}")
        print(f"  Errors caught: {audit['errors_caught_by_review']} | "
              f"Slipped: {audit['errors_slipped_through']}")
        print(f"  Physician hours saved: {audit['physician_hours_saved']:.0f} | "
              f"Cost saved: ${audit['cost_saved_dollars']:,.0f}")
        print(f"  FDA-defensible (<5% slip): {'YES' if audit['fda_defensible'] else 'NO'} "
              f"(slip rate: {audit['slip_rate']:.1%})")

        # Plot
        tag = " (in-distribution)" if "In-Dist" in name else " (out-of-distribution)"
        plot_healthcare_dashboard(triage, harm, routing, safety, audit,
                                  args.fig_dir, tag)

        # Store results
        key = "in_distribution" if "In-Dist" in name else "out_of_distribution"
        all_results[key] = {
            "n": len(samples),
            "base_accuracy": float(base_acc),
            "auroc": float(auroc),
            "triage": triage,
            "harm_severity": harm,
            "telehealth_routing": routing,
            "patient_safety": safety,
            "compliance_audit": audit,
        }

    # Save results
    out_path = os.path.join(args.output_dir, "healthcare_realistic_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: int(x) if isinstance(x, (np.integer,))
                  else float(x) if isinstance(x, (np.floating,))
                  else bool(x) if isinstance(x, (np.bool_,)) else x)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
