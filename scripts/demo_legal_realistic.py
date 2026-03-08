#!/usr/bin/env python3
"""Demo: Realistic Legal Deployment of UQ Calibrator.

Designed around what senior lawyers actually need, not generic ML metrics.

Scenarios:
1. ASSOCIATE TRIAGE: Partner gets 50 AI-drafted research memos. Which ones
   need deep review? Sort by calibrator confidence, review bottom N%.
   Metric: How many malpractice-risk errors slip through?

2. HALLUCINATION AUDIT: Before filing, run every factual claim through the
   calibrator. Flag anything the model is uncertain about. Show the lawyer
   a red/yellow/green annotation per response.

3. PRO-SE PLAIN LANGUAGE: Replace confidence scores with plain English
   warnings that a non-lawyer can understand.

4. BILLING JUSTIFICATION: Generate an audit log showing calibrator scores
   for each piece of AI-assisted work product, supporting the firm's
   "reasonable reliance" defense if challenged.

5. COMPARATIVE VALUE: Is the calibrator better than a first-year associate
   at catching errors? (Benchmark: associate error rate ~15-20% on review tasks)

Usage:
    conda activate uq_eval && python scripts/demo_legal_realistic.py
    conda activate uq_eval && python scripts/demo_legal_realistic.py --include_unseen
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
    """Load scored data. Optionally include unseen (OOD) benchmarks."""
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
# SCENARIO 1: ASSOCIATE TRIAGE
# ============================================================

def associate_triage(samples):
    """Partner has N memos from AI. Sort by confidence. Review bottom K%.

    Key question: If I only review the least confident 20%, 30%, 40%,
    how many errors slip through in the auto-approved pile?

    Compare to: A first-year associate who catches ~80-85% of errors
    (i.e., misses 15-20%) when reviewing ALL memos.
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])
    n = len(labels)
    n_errors = (labels == 0).sum()

    # Sort by confidence ascending (review least confident first)
    order = np.argsort(scores)
    sorted_labels = labels[order]

    # First-year associate baseline: reviews everything, catches 82% of errors
    associate_catch_rate = 0.82  # literature: junior associates miss ~18% of issues
    associate_errors_missed = int(n_errors * (1 - associate_catch_rate))

    results = {"n_total": n, "n_errors": int(n_errors),
               "associate_catch_rate": associate_catch_rate,
               "associate_errors_missed": associate_errors_missed}

    triage_rows = []
    for review_pct in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        k = max(1, int(n * review_pct))
        # Errors in the reviewed pile (caught by human)
        errors_in_reviewed = (sorted_labels[:k] == 0).sum()
        # Errors in auto-approved pile (slip through)
        errors_slipped = n_errors - errors_in_reviewed
        # Error rate in auto-approved
        auto_pile = sorted_labels[k:]
        auto_error_rate = (auto_pile == 0).mean() if len(auto_pile) > 0 else 0
        # Malpractice risk: how does this compare to associate review?
        better_than_associate = errors_slipped < associate_errors_missed

        triage_rows.append({
            "review_pct": review_pct,
            "n_reviewed": k,
            "errors_caught": int(errors_in_reviewed),
            "errors_slipped": int(errors_slipped),
            "catch_rate": float(errors_in_reviewed / n_errors) if n_errors > 0 else 1.0,
            "auto_error_rate": float(auto_error_rate),
            "better_than_associate": better_than_associate,
            "memos_saved_from_review": n - k,
        })

    results["triage"] = triage_rows
    return results


# ============================================================
# SCENARIO 2: RED/YELLOW/GREEN ANNOTATION
# ============================================================

def annotation_system(samples):
    """Annotate each response with a plain-language reliability label.

    GREEN: "This response appears reliable based on our analysis."
    YELLOW: "This response may contain inaccuracies. Verify key claims."
    RED: "This response is likely unreliable. Do not rely on it without
          independent verification."

    Key metric: What's the FALSE GREEN rate? (Green-labeled but actually wrong)
    This is the malpractice risk metric.
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])

    tiers = [
        {"name": "GREEN", "label": "Appears reliable",
         "lo": 0.8, "hi": 1.01, "color": "#27ae60"},
        {"name": "YELLOW", "label": "Verify key claims",
         "lo": 0.5, "hi": 0.8, "color": "#f39c12"},
        {"name": "RED", "label": "Likely unreliable",
         "lo": 0.0, "hi": 0.5, "color": "#e74c3c"},
    ]

    tier_results = []
    for tier in tiers:
        mask = (scores >= tier["lo"]) & (scores < tier["hi"])
        n_tier = mask.sum()
        if n_tier == 0:
            tier_results.append({**tier, "n": 0, "pct": 0, "accuracy": 0,
                                "false_rate": 0, "n_false": 0})
            continue

        acc = labels[mask].mean()
        n_false = (labels[mask] == 0).sum()
        false_rate = 1 - acc  # rate of wrong answers in this tier

        tier_results.append({
            "name": tier["name"],
            "label": tier["label"],
            "color": tier["color"],
            "n": int(n_tier),
            "pct": float(mask.mean()),
            "accuracy": float(acc),
            "false_rate": float(false_rate),
            "n_false": int(n_false),
        })

    # Key malpractice metric: false green rate
    green = tier_results[0]
    return {
        "tiers": tier_results,
        "false_green_rate": green["false_rate"],
        "n_false_greens": green["n_false"],
        "total_greens": green["n"],
    }


# ============================================================
# SCENARIO 3: PRO-SE PLAIN LANGUAGE
# ============================================================

def prose_warnings(samples):
    """Generate plain-language warnings for non-lawyers.

    Instead of confidence scores, produce messages like:
    - "You can likely rely on this information."
    - "This answer may not be complete. Consider consulting a lawyer."
    - "WARNING: This answer may be wrong. Please consult a licensed attorney
       before taking any action based on this information."
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])

    warnings = [
        {"threshold": 0.8,
         "message": "You can likely rely on this information for general understanding.",
         "caveat": "This is not legal advice. For important decisions, consult a licensed attorney."},
        {"threshold": 0.5,
         "message": "This answer may not be fully accurate. Key claims should be verified.",
         "caveat": "Consider consulting a lawyer before acting on this information."},
        {"threshold": 0.0,
         "message": "WARNING: This answer may contain significant errors.",
         "caveat": "Do NOT rely on this without consulting a licensed attorney."},
    ]

    results = []
    prev_threshold = 1.01
    for w in warnings:
        mask = (scores >= w["threshold"]) & (scores < prev_threshold)
        n_tier = mask.sum()
        acc = labels[mask].mean() if n_tier > 0 else 0

        results.append({
            **w,
            "n": int(n_tier),
            "pct": float(mask.mean()),
            "actual_accuracy": float(acc),
            "user_risk": "LOW" if acc > 0.9 else "MEDIUM" if acc > 0.6 else "HIGH",
        })
        prev_threshold = w["threshold"]

    return results


# ============================================================
# SCENARIO 4: BILLING AUDIT LOG
# ============================================================

def billing_audit(samples):
    """Generate audit log format for AI-assisted work product.

    Each entry shows: task ID, calibrator score, reliability tier,
    whether human review was performed, and the outcome.

    This supports a firm's "reasonable reliance" defense: we had a
    systematic process for vetting AI outputs.
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])

    # Simulate billing: review anything below 0.7 threshold
    review_threshold = 0.7
    reviewed = scores < review_threshold
    auto_approved = ~reviewed

    # Assume human review catches all errors
    errors_in_reviewed = (labels[reviewed] == 0).sum()
    errors_slipped = (labels[auto_approved] == 0).sum()
    total_errors = (labels == 0).sum()

    billable_hours_saved = auto_approved.sum() * 0.1  # 6 min per item
    cost_saved = billable_hours_saved * 200  # $200/hr associate rate

    return {
        "review_threshold": review_threshold,
        "total_items": len(labels),
        "auto_approved": int(auto_approved.sum()),
        "human_reviewed": int(reviewed.sum()),
        "errors_caught_by_review": int(errors_in_reviewed),
        "errors_slipped_through": int(errors_slipped),
        "total_errors": int(total_errors),
        "catch_rate": float(errors_in_reviewed / total_errors) if total_errors > 0 else 1.0,
        "billable_hours_saved": float(billable_hours_saved),
        "cost_saved_dollars": float(cost_saved),
        "defensible": errors_slipped < total_errors * 0.10,  # <10% slip = defensible
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_legal_dashboard(triage, annotation, prose, audit, fig_dir, tag=""):
    """4-panel dashboard designed for a law firm presentation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Associate triage — errors caught vs review effort
    ax = axes[0, 0]
    rows = triage["triage"]
    review_pcts = [r["review_pct"] for r in rows]
    catch_rates = [r["catch_rate"] for r in rows]
    ax.plot(review_pcts, catch_rates, "o-", color="#2c3e50", linewidth=2.5, markersize=8)
    ax.axhline(y=triage["associate_catch_rate"], color="#e74c3c", linestyle="--",
               linewidth=2, label=f"1st-year associate ({triage['associate_catch_rate']:.0%} catch rate)")
    ax.fill_between(review_pcts, catch_rates, alpha=0.1, color="#2c3e50")
    ax.set_xlabel("Fraction of Memos Reviewed (lowest confidence first)", fontsize=11)
    ax.set_ylabel("Error Catch Rate", fontsize=11)
    ax.set_title("(a) Triage: Review Effort vs Errors Caught", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    # Find crossover point
    for r in rows:
        if r["catch_rate"] >= triage["associate_catch_rate"]:
            ax.annotate(f"Match associate at {r['review_pct']:.0%} review",
                       xy=(r["review_pct"], r["catch_rate"]),
                       xytext=(r["review_pct"] + 0.08, r["catch_rate"] - 0.12),
                       arrowprops=dict(arrowstyle="->", color="#2c3e50"),
                       fontsize=9, color="#2c3e50")
            break

    # Panel 2: Annotation tiers with false-rate emphasis
    ax = axes[0, 1]
    tiers = annotation["tiers"]
    names = [t["name"] for t in tiers]
    pcts = [t["pct"] for t in tiers]
    false_rates = [t["false_rate"] for t in tiers]
    colors = [t["color"] for t in tiers]

    x = np.arange(len(names))
    bars = ax.bar(x - 0.2, pcts, 0.35, color=colors, alpha=0.7, label="% of responses")
    bars2 = ax.bar(x + 0.2, false_rates, 0.35, color=colors, edgecolor="black",
                   linewidth=1.5, label="Error rate in tier")

    for i, (p, fr, n) in enumerate(zip(pcts, false_rates, [t["n"] for t in tiers])):
        ax.text(i - 0.2, p + 0.02, f"{p:.0%}\n(n={n})", ha="center", fontsize=8)
        ax.text(i + 0.2, fr + 0.02, f"{fr:.0%}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({t['label']})" for n, t in zip(names, tiers)], fontsize=9)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title(f"(b) Reliability Tiers — False GREEN rate: {annotation['false_green_rate']:.1%}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis="y")

    # Panel 3: Pro-se plain language warnings
    ax = axes[1, 0]
    for i, w in enumerate(prose):
        color = "#27ae60" if w["user_risk"] == "LOW" else "#f39c12" if w["user_risk"] == "MEDIUM" else "#e74c3c"
        ax.barh(i, w["pct"], color=color, edgecolor="white", height=0.6)
        ax.text(w["pct"] + 0.02, i,
                f'{w["actual_accuracy"]:.0%} accurate  ({w["n"]} answers)',
                va="center", fontsize=10)

    ax.set_yticks(range(len(prose)))
    labels = []
    for w in prose:
        msg = w["message"]
        if len(msg) > 50:
            msg = msg[:47] + "..."
        labels.append(msg)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Fraction of Responses", fontsize=11)
    ax.set_title("(c) Plain-Language Warnings for Self-Help Users", fontsize=12)
    ax.set_xlim(0, 1.0)

    # Panel 4: Billing audit summary
    ax = axes[1, 1]
    audit_text = f"""AUDIT SUMMARY
{'='*40}
Total AI work products:     {audit['total_items']:,}
Auto-approved (conf > {audit['review_threshold']}): {audit['auto_approved']:,}
Human-reviewed:             {audit['human_reviewed']:,}

Errors caught by review:    {audit['errors_caught_by_review']}
Errors slipped through:     {audit['errors_slipped_through']}
Overall catch rate:         {audit['catch_rate']:.0%}

Billable hours saved:       {audit['billable_hours_saved']:.0f} hrs
Cost savings:               ${audit['cost_saved_dollars']:,.0f}

Defensible process:         {'YES' if audit['defensible'] else 'NO'}
({'<10% error slip rate' if audit['defensible'] else '>10% slip rate — increase review'})"""

    ax.text(0.05, 0.95, audit_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(d) Billing & Compliance Audit", fontsize=12)

    plt.suptitle(f"Legal AI Quality Assurance Dashboard{tag}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    suffix = tag.replace(" ", "_").replace("(", "").replace(")", "").lower()
    path = os.path.join(fig_dir, f"legal_dashboard{suffix}.png")
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
    parser.add_argument("--output_dir", default="data/use_cases/legal_realistic")
    parser.add_argument("--fig_dir", default="figures/legal_realistic")
    parser.add_argument("--include_unseen", action="store_true",
                        help="Also run on OOD (unseen benchmark) data for honest comparison")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    in_dist, unseen = load_all(args.scored_dir, args.unseen_dir, args.include_unseen)
    if args.smoke_test:
        in_dist = in_dist[:50]
        unseen = unseen[:50]

    # Filter to knowledge/reasoning benchmarks (most relevant to legal)
    legal_relevant = {"simpleqa", "gpqa", "hle", "livebench", "bbeh",
                      "chembench", "mmmu", "prbench"}
    in_dist_legal = [s for s in in_dist if s["benchmark"] in legal_relevant]

    datasets = [("In-Distribution", in_dist_legal)]
    if unseen:
        datasets.append(("Out-of-Distribution (unseen benchmarks)", unseen))

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
        print("\n--- Scenario 1: Associate Triage ---")
        triage = associate_triage(samples)
        print(f"  Total errors: {triage['n_errors']}")
        print(f"  1st-year associate would miss: {triage['associate_errors_missed']} errors")
        print(f"  {'Review%':>8} {'Caught':>8} {'Slipped':>8} {'Catch%':>8} {'Better?':>8}")
        for r in triage["triage"]:
            print(f"  {r['review_pct']:>8.0%} {r['errors_caught']:>8} {r['errors_slipped']:>8} "
                  f"{r['catch_rate']:>8.0%} {'YES' if r['better_than_associate'] else 'no':>8}")

        print("\n--- Scenario 2: Reliability Annotation ---")
        annotation = annotation_system(samples)
        for t in annotation["tiers"]:
            print(f"  {t['name']:8s} | {t['pct']:5.0%} of responses | "
                  f"accuracy {t['accuracy']:.1%} | {t['n_false']} false labels")
        print(f"  MALPRACTICE RISK: {annotation['n_false_greens']} wrong answers labeled GREEN "
              f"({annotation['false_green_rate']:.1%} false green rate)")

        print("\n--- Scenario 3: Pro-Se Warnings ---")
        prose = prose_warnings(samples)
        for w in prose:
            print(f"  [{w['user_risk']:6s}] {w['pct']:5.0%} | acc={w['actual_accuracy']:.0%} | "
                  f'"{w["message"][:60]}"')

        print("\n--- Scenario 4: Billing Audit ---")
        audit = billing_audit(samples)
        print(f"  Auto-approved: {audit['auto_approved']} | Reviewed: {audit['human_reviewed']}")
        print(f"  Errors caught: {audit['errors_caught_by_review']} | "
              f"Slipped: {audit['errors_slipped_through']}")
        print(f"  Hours saved: {audit['billable_hours_saved']:.0f} | "
              f"Cost saved: ${audit['cost_saved_dollars']:,.0f}")
        print(f"  Defensible: {'YES' if audit['defensible'] else 'NO'}")

        # Plot
        tag = " (in-distribution)" if "In-Dist" in name else " (out-of-distribution)"
        plot_legal_dashboard(triage, annotation, prose, audit, args.fig_dir, tag)

    # Save results
    output = {
        "in_distribution": {
            "n": len(in_dist_legal),
            "triage": associate_triage(in_dist_legal) if len(in_dist_legal) >= 10 else None,
            "annotation": annotation_system(in_dist_legal) if len(in_dist_legal) >= 10 else None,
            "audit": billing_audit(in_dist_legal) if len(in_dist_legal) >= 10 else None,
        },
    }
    if unseen and len(unseen) >= 10:
        output["out_of_distribution"] = {
            "n": len(unseen),
            "triage": associate_triage(unseen),
            "annotation": annotation_system(unseen),
            "audit": billing_audit(unseen),
        }

    out_path = os.path.join(args.output_dir, "legal_realistic_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: int(x) if isinstance(x, (np.integer,)) else float(x) if isinstance(x, (np.floating,)) else bool(x) if isinstance(x, (np.bool_,)) else x)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
