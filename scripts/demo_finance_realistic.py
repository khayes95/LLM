#!/usr/bin/env python3
"""Demo: Realistic Finance Deployment of UQ Calibrator.

Designed around what financial analysts, compliance officers, robo-advisors,
and retail investors need from AI uncertainty quantification.

Scenarios:
1. ANALYST TRIAGE: AI generates research notes or market summaries. Sort by
   calibrator confidence, review bottom N%. Metric: How many factual errors
   reach clients in published research?

2. RISK TIERING: Green/Yellow/Red reliability labels for AI-generated
   financial content. Key metric: FALSE GREEN rate — wrong information
   labeled as reliable → regulatory risk (SEC, FINRA).

3. ROBO-ADVISOR ROUTING: Route client questions by confidence tier.
   AI auto-response ($0.05) -> Junior analyst ($15) -> Senior analyst ($50).
   Cost-safety tradeoff for wealth management platforms.

4. RETAIL INVESTOR WARNINGS: Plain-language disclaimers for non-professional
   investors using AI for financial decisions:
   "General information" vs "Consult a financial advisor" vs
   "WARNING: Do not make investment decisions based on this"

5. COMPLIANCE AUDIT LOG: Generate audit trail for SEC/FINRA compliance.
   Document calibrator scores, review decisions, error rates. Support
   "reasonable basis" and "suitability" defense under Reg BI.

Usage:
    conda activate uq_eval && python scripts/demo_finance_realistic.py
    conda activate uq_eval && python scripts/demo_finance_realistic.py --include_unseen
"""
import argparse
import json
import os

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
# SCENARIO 1: ANALYST TRIAGE
# ============================================================

def analyst_triage(samples):
    """Senior analyst has N AI-generated research notes. Sort by confidence.
    Review bottom K%. How many factual errors reach clients?

    Compare to: Junior analyst baseline — catches ~75% of factual errors
    when reviewing all notes (misses 25% due to domain gaps, time pressure).
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])
    n = len(labels)
    n_errors = (labels == 0).sum()

    junior_catch_rate = 0.75  # junior analysts miss ~25% of errors
    junior_errors_missed = int(n_errors * (1 - junior_catch_rate))

    order = np.argsort(scores)
    sorted_labels = labels[order]

    results = {"n_total": n, "n_errors": int(n_errors),
               "junior_catch_rate": junior_catch_rate,
               "junior_errors_missed": junior_errors_missed}

    triage_rows = []
    for review_pct in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        k = max(1, int(n * review_pct))
        errors_in_reviewed = (sorted_labels[:k] == 0).sum()
        errors_slipped = n_errors - errors_in_reviewed
        auto_pile = sorted_labels[k:]
        auto_error_rate = (auto_pile == 0).mean() if len(auto_pile) > 0 else 0
        better_than_junior = errors_slipped < junior_errors_missed

        triage_rows.append({
            "review_pct": review_pct,
            "n_reviewed": k,
            "errors_caught": int(errors_in_reviewed),
            "errors_slipped": int(errors_slipped),
            "catch_rate": float(errors_in_reviewed / n_errors) if n_errors > 0 else 1.0,
            "auto_error_rate": float(auto_error_rate),
            "better_than_junior": better_than_junior,
            "notes_saved_from_review": n - k,
        })

    results["triage"] = triage_rows
    return results


# ============================================================
# SCENARIO 2: RISK TIERING
# ============================================================

def risk_tiering(samples):
    """Annotate each AI output with reliability tier for compliance.

    GREEN: "This analysis appears reliable based on our quality checks."
    YELLOW: "This analysis should be independently verified before use."
    RED: "This analysis may contain material errors. Do not distribute."

    Key metric: FALSE GREEN rate = wrong analyses labeled reliable
    → SEC/FINRA regulatory risk if distributed to clients.
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])

    tiers = [
        {"name": "GREEN", "label": "Appears reliable",
         "lo": 0.8, "hi": 1.01, "color": "#27ae60"},
        {"name": "YELLOW", "label": "Verify independently",
         "lo": 0.5, "hi": 0.8, "color": "#f39c12"},
        {"name": "RED", "label": "May contain material errors",
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
        tier_results.append({
            "name": tier["name"],
            "label": tier["label"],
            "color": tier["color"],
            "n": int(n_tier),
            "pct": float(mask.mean()),
            "accuracy": float(acc),
            "false_rate": float(1 - acc),
            "n_false": int(n_false),
        })

    green = tier_results[0]
    return {
        "tiers": tier_results,
        "false_green_rate": green["false_rate"],
        "n_false_greens": green["n_false"],
        "total_greens": green["n"],
    }


# ============================================================
# SCENARIO 3: ROBO-ADVISOR ROUTING
# ============================================================

def roboadvisor_routing(samples):
    """Route client questions by confidence tier.

    Cost model:
    - AI auto-response: $0.05 per question (API cost)
    - Junior analyst review: $15 per question (~15 min @ $60/hr)
    - Senior analyst review: $50 per question (~30 min @ $100/hr)
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])
    n = len(labels)

    all_senior_cost = n * 50

    strategies = [
        {"name": "All senior analyst review", "ai_thresh": 1.01, "junior_thresh": 1.01},
        {"name": "AI>0.9, else senior", "ai_thresh": 0.9, "junior_thresh": 1.01},
        {"name": "AI>0.8, junior 0.5-0.8, else senior", "ai_thresh": 0.8, "junior_thresh": 0.5},
        {"name": "AI>0.7, junior 0.4-0.7, else senior", "ai_thresh": 0.7, "junior_thresh": 0.4},
        {"name": "AI>0.6, junior 0.3-0.6, else senior", "ai_thresh": 0.6, "junior_thresh": 0.3},
    ]

    results = []
    for strat in strategies:
        ai_mask = scores >= strat["ai_thresh"]
        junior_mask = (~ai_mask) & (scores >= strat["junior_thresh"])
        senior_mask = ~ai_mask & ~junior_mask

        ai_cost = ai_mask.sum() * 0.05
        junior_cost = junior_mask.sum() * 15
        senior_cost = senior_mask.sum() * 50
        total_cost = ai_cost + junior_cost + senior_cost

        ai_errors = (ai_mask & (labels == 0)).sum()
        ai_error_rate = ai_errors / ai_mask.sum() if ai_mask.sum() > 0 else 0

        results.append({
            "strategy": strat["name"],
            "ai_pct": float(ai_mask.mean()),
            "junior_pct": float(junior_mask.mean()),
            "senior_pct": float(senior_mask.mean()),
            "total_cost": float(total_cost),
            "cost_per_question": float(total_cost / n),
            "cost_savings": float(1 - total_cost / all_senior_cost),
            "ai_errors": int(ai_errors),
            "ai_error_rate": float(ai_error_rate),
        })

    return {"all_senior_cost": float(all_senior_cost),
            "cost_per_question_baseline": 50.0,
            "strategies": results}


# ============================================================
# SCENARIO 4: RETAIL INVESTOR WARNINGS
# ============================================================

def investor_warnings(samples):
    """Plain-language disclaimers for retail investors.

    HIGH: "This information is based on publicly available data."
    MEDIUM: "This analysis may be incomplete. Consider consulting a
            licensed financial advisor before acting."
    LOW: "WARNING: This information may contain material errors. Do NOT
          make investment decisions based solely on this output."
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])

    warnings = [
        {"threshold": 0.8, "risk": "LOW",
         "message": "Based on publicly available data. Not personalized investment advice.",
         "caveat": "Past performance does not guarantee future results."},
        {"threshold": 0.5, "risk": "MEDIUM",
         "message": "This analysis may be incomplete or based on limited information.",
         "caveat": "Consider consulting a licensed financial advisor before acting."},
        {"threshold": 0.0, "risk": "HIGH",
         "message": "WARNING: This information may contain material errors.",
         "caveat": "Do NOT make investment decisions based solely on this output."},
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
        })
        prev_threshold = w["threshold"]

    return results


# ============================================================
# SCENARIO 5: COMPLIANCE AUDIT LOG
# ============================================================

def compliance_audit(samples):
    """Generate SEC/FINRA compliance audit summary.

    Regulation Best Interest (Reg BI) requires firms to have a
    "reasonable basis" for recommendations. The calibrator provides
    a systematic quality-check process that supports this defense.
    """
    labels = np.array([s["is_correct"] for s in samples])
    scores = np.array([s.get("p_correct") or 0.5 for s in samples])

    review_threshold = 0.7
    reviewed = scores < review_threshold
    auto_approved = ~reviewed

    errors_in_reviewed = (labels[reviewed] == 0).sum()
    errors_slipped = (labels[auto_approved] == 0).sum()
    total_errors = (labels == 0).sum()

    # Time savings: analyst review = 15 min per note
    analyst_hours_saved = auto_approved.sum() * 0.25  # 15 min per item
    cost_saved = analyst_hours_saved * 60  # $60/hr junior analyst rate

    return {
        "review_threshold": review_threshold,
        "total_items": len(labels),
        "auto_approved": int(auto_approved.sum()),
        "analyst_reviewed": int(reviewed.sum()),
        "errors_caught_by_review": int(errors_in_reviewed),
        "errors_slipped_through": int(errors_slipped),
        "total_errors": int(total_errors),
        "catch_rate": float(errors_in_reviewed / total_errors) if total_errors > 0 else 1.0,
        "slip_rate": float(errors_slipped / total_errors) if total_errors > 0 else 0,
        "analyst_hours_saved": float(analyst_hours_saved),
        "cost_saved_dollars": float(cost_saved),
        "reg_bi_defensible": errors_slipped < total_errors * 0.10,  # <10% slip
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_finance_dashboard(triage, tiering, routing, warnings, audit, fig_dir, tag=""):
    """5-panel dashboard for financial services deployment."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # Panel 1: Analyst triage — errors caught vs review effort
    ax = fig.add_subplot(gs[0, 0])
    rows = triage["triage"]
    review_pcts = [r["review_pct"] for r in rows]
    catch_rates = [r["catch_rate"] for r in rows]
    ax.plot(review_pcts, catch_rates, "o-", color="#2c3e50", linewidth=2.5, markersize=8)
    ax.axhline(y=triage["junior_catch_rate"], color="#e74c3c", linestyle="--",
               linewidth=2, label=f"Junior analyst ({triage['junior_catch_rate']:.0%} catch)")
    ax.fill_between(review_pcts, catch_rates, alpha=0.1, color="#2c3e50")
    ax.set_xlabel("Fraction of Notes Reviewed", fontsize=10)
    ax.set_ylabel("Error Catch Rate", fontsize=10)
    ax.set_title("(a) Analyst Triage:\nReview Effort vs Errors Caught", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    for r in rows:
        if r["catch_rate"] >= triage["junior_catch_rate"]:
            ax.annotate(f"Match junior at {r['review_pct']:.0%}",
                       xy=(r["review_pct"], r["catch_rate"]),
                       xytext=(r["review_pct"] + 0.08, r["catch_rate"] - 0.12),
                       arrowprops=dict(arrowstyle="->", color="#2c3e50"),
                       fontsize=8, color="#2c3e50")
            break

    # Panel 2: Risk tiering with false-rate emphasis
    ax = fig.add_subplot(gs[0, 1])
    tiers = tiering["tiers"]
    names = [t["name"] for t in tiers]
    pcts = [t["pct"] for t in tiers]
    false_rates = [t["false_rate"] for t in tiers]
    colors = [t["color"] for t in tiers]

    x = np.arange(len(names))
    ax.bar(x - 0.2, pcts, 0.35, color=colors, alpha=0.7, label="% of outputs")
    ax.bar(x + 0.2, false_rates, 0.35, color=colors, edgecolor="black",
           linewidth=1.5, label="Error rate in tier")

    for i, (p, fr, n) in enumerate(zip(pcts, false_rates, [t["n"] for t in tiers])):
        ax.text(i - 0.2, p + 0.02, f"{p:.0%}\n(n={n})", ha="center", fontsize=7)
        ax.text(i + 0.2, fr + 0.02, f"{fr:.0%}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({t['label'][:20]})" for n, t in zip(names, tiers)], fontsize=8)
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_title(f"(b) Risk Tiers — False GREEN: {tiering['false_green_rate']:.1%}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, axis="y")

    # Panel 3: Robo-advisor routing cost
    ax = fig.add_subplot(gs[0, 2])
    strats = routing["strategies"]
    strat_names = [s["strategy"][:30] for s in strats]
    costs = [s["total_cost"] for s in strats]
    errors = [s["ai_errors"] for s in strats]
    colors_bar = ["#e74c3c" if e > 0 else "#27ae60" for e in errors]
    ax.barh(range(len(strat_names)), costs, color=colors_bar, edgecolor="white")
    ax.set_yticks(range(len(strat_names)))
    ax.set_yticklabels(strat_names, fontsize=7)
    ax.set_xlabel("Total Cost ($)")
    ax.set_title("(c) Robo-Advisor Routing Cost\n(red = has AI errors)", fontsize=11)
    for i, (cost, err) in enumerate(zip(costs, errors)):
        ax.text(cost + max(costs) * 0.02, i, f"${cost:,.0f} ({err} err)",
                va="center", fontsize=7)

    # Panel 4: Retail investor warnings
    ax = fig.add_subplot(gs[1, 0])
    for i, w in enumerate(warnings):
        color = "#27ae60" if w["risk"] == "LOW" else "#f39c12" if w["risk"] == "MEDIUM" else "#e74c3c"
        ax.barh(i, w["pct"], color=color, edgecolor="white", height=0.6)
        ax.text(w["pct"] + 0.02, i,
                f'{w["actual_accuracy"]:.0%} accurate ({w["n"]} answers)',
                va="center", fontsize=9)

    ax.set_yticks(range(len(warnings)))
    labels = []
    for w in warnings:
        msg = w["message"]
        if len(msg) > 45:
            msg = msg[:42] + "..."
        labels.append(msg)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Fraction of Responses", fontsize=10)
    ax.set_title("(d) Retail Investor Warnings", fontsize=11)
    ax.set_xlim(0, 1.0)

    # Panel 5: Compliance audit summary
    ax = fig.add_subplot(gs[1, 1:])
    audit_text = f"""SEC/FINRA COMPLIANCE AUDIT
{'='*50}
Total AI-generated analyses:           {audit['total_items']:,}
Auto-approved (confidence >= {audit['review_threshold']}):  {audit['auto_approved']:,}
Analyst-reviewed:                      {audit['analyst_reviewed']:,}

Errors caught by analyst review:       {audit['errors_caught_by_review']}
Errors in auto-approved (slipped):     {audit['errors_slipped_through']}
Overall error catch rate:              {audit['catch_rate']:.0%}
Error slip rate:                       {audit['slip_rate']:.1%}

Analyst hours saved:                   {audit['analyst_hours_saved']:.0f} hrs
Cost savings (@ $60/hr junior):       ${audit['cost_saved_dollars']:,.0f}

RISK TIER SUMMARY:
  GREEN (reliable):   {tiering['tiers'][0]['pct']:.0%} of outputs, {tiering['tiers'][0]['accuracy']:.0%} accurate
  YELLOW (verify):    {tiering['tiers'][1]['pct']:.0%} of outputs, {tiering['tiers'][1]['accuracy']:.0%} accurate
  RED (unreliable):   {tiering['tiers'][2]['pct']:.0%} of outputs, {tiering['tiers'][2]['accuracy']:.0%} accurate

False GREEN rate:                      {tiering['false_green_rate']:.1%}
Reg BI defensible (<10% slip):         {'YES' if audit['reg_bi_defensible'] else 'NO'}"""

    ax.text(0.05, 0.95, audit_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(e) Regulatory Compliance Summary", fontsize=12)

    plt.suptitle(f"Financial AI Quality Assurance Dashboard{tag}",
                 fontsize=14, fontweight="bold")
    suffix = tag.replace(" ", "_").replace("(", "").replace(")", "").lower()
    path = os.path.join(fig_dir, f"finance_dashboard{suffix}.png")
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
    parser.add_argument("--output_dir", default="data/use_cases/finance_realistic")
    parser.add_argument("--fig_dir", default="figures/finance_realistic")
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

    # Finance-relevant benchmarks: quantitative reasoning, knowledge, factuality
    finance_benchmarks = {"gpqa", "simpleqa", "mmlu", "livebench", "bbeh",
                          "hle", "omnimath", "mathvista", "mathvision", "mathverse"}
    in_dist_finance = [s for s in in_dist if s["benchmark"] in finance_benchmarks]

    datasets = [("In-Distribution", in_dist_finance)]
    if unseen:
        datasets.append(("Out-of-Distribution (unseen benchmarks)", unseen))

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
        print("\n--- Scenario 1: Analyst Triage ---")
        triage = analyst_triage(samples)
        print(f"  Total errors: {triage['n_errors']}")
        print(f"  Junior analyst would miss: {triage['junior_errors_missed']} errors")
        print(f"  {'Review%':>8} {'Caught':>8} {'Slipped':>8} {'Catch%':>8} {'Better?':>8}")
        for r in triage["triage"]:
            print(f"  {r['review_pct']:>8.0%} {r['errors_caught']:>8} {r['errors_slipped']:>8} "
                  f"{r['catch_rate']:>8.0%} {'YES' if r['better_than_junior'] else 'no':>8}")

        print("\n--- Scenario 2: Risk Tiering ---")
        tiering = risk_tiering(samples)
        for t in tiering["tiers"]:
            print(f"  {t['name']:8s} | {t['pct']:5.0%} of outputs | "
                  f"accuracy {t['accuracy']:.1%} | {t['n_false']} false labels")
        print(f"  REGULATORY RISK: {tiering['n_false_greens']} wrong outputs labeled GREEN "
              f"({tiering['false_green_rate']:.1%} false green rate)")

        print("\n--- Scenario 3: Robo-Advisor Routing ---")
        routing = roboadvisor_routing(samples)
        for s in routing["strategies"]:
            print(f"    {s['strategy']:42s} | ${s['total_cost']:>8,.0f} | "
                  f"savings={s['cost_savings']:.0%} | AI errors={s['ai_errors']}")

        print("\n--- Scenario 4: Retail Investor Warnings ---")
        warnings = investor_warnings(samples)
        for w in warnings:
            print(f"  [{w['risk']:6s}] {w['pct']:5.0%} | acc={w['actual_accuracy']:.0%} | "
                  f'"{w["message"][:55]}"')

        print("\n--- Scenario 5: Compliance Audit ---")
        audit = compliance_audit(samples)
        print(f"  Auto-approved: {audit['auto_approved']} | Reviewed: {audit['analyst_reviewed']}")
        print(f"  Errors caught: {audit['errors_caught_by_review']} | "
              f"Slipped: {audit['errors_slipped_through']}")
        print(f"  Hours saved: {audit['analyst_hours_saved']:.0f} | "
              f"Cost saved: ${audit['cost_saved_dollars']:,.0f}")
        print(f"  Reg BI defensible: {'YES' if audit['reg_bi_defensible'] else 'NO'} "
              f"(slip rate: {audit['slip_rate']:.1%})")

        # Plot
        tag = " (in-distribution)" if "In-Dist" in name else " (out-of-distribution)"
        plot_finance_dashboard(triage, tiering, routing, warnings, audit,
                               args.fig_dir, tag)

        # Store results
        key = "in_distribution" if "In-Dist" in name else "out_of_distribution"
        all_results[key] = {
            "n": len(samples),
            "base_accuracy": float(base_acc),
            "auroc": float(auroc),
            "analyst_triage": triage,
            "risk_tiering": tiering,
            "roboadvisor_routing": routing,
            "investor_warnings": warnings,
            "compliance_audit": audit,
        }

    # Save results
    out_path = os.path.join(args.output_dir, "finance_realistic_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: int(x) if isinstance(x, (np.integer,))
                  else float(x) if isinstance(x, (np.floating,))
                  else bool(x) if isinstance(x, (np.bool_,)) else x)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
