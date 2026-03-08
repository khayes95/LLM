#!/usr/bin/env python3
"""Demo: Professional Advice Safety Triage via UQ Calibrator.

Shows how ANY professional domain (legal, medical, financial, engineering)
can use the UQ calibrator as a safety layer. Compares different deployment
strategies and quantifies the cost-benefit tradeoff.

Use cases:
1. Junior Associate Replacement: How much review work does UQ eliminate?
2. Pro-Se / Self-Help Safety: When should users be warned to consult an expert?
3. Contract/Document Review: Confidence per-section for prioritizing human review
4. Cross-Model Trust: Which LLM to use for which domain?

Usage:
    python scripts/demo_professional_safety.py
    python scripts/demo_professional_safety.py --smoke_test
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


def load_all_scored(scored_dir, smoke_test=False):
    all_samples = []
    for fn in sorted(os.listdir(scored_dir)):
        if fn.endswith("_scored.jsonl"):
            path = os.path.join(scored_dir, fn)
            with open(path) as f:
                samples = [json.loads(line) for line in f]
            if smoke_test:
                samples = samples[:20]
            all_samples.extend(samples)
    return all_samples


def junior_associate_analysis(samples):
    """Simulate replacing junior associate review with UQ-guided triage.

    A junior associate costs ~$200/hr and reviews ~10 items/hr.
    The AI generates answers, the UQ calibrator triages them:
    - High confidence → skip review ($0)
    - Low confidence → associate reviews ($20/item)
    """
    labels = np.array([s["is_correct"] for s in samples])
    cal_scores = np.array([s.get("p_correct") or 0.5 for s in samples])
    n = len(labels)

    cost_per_review = 20  # dollars per item (junior associate)
    cost_full_review = n * cost_per_review

    strategies = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        auto_approve = cal_scores >= threshold
        needs_review = ~auto_approve

        auto_acc = labels[auto_approve].mean() if auto_approve.sum() > 0 else 0
        review_count = needs_review.sum()
        review_cost = review_count * cost_per_review

        # Errors that slip through (auto-approved but wrong)
        slipped = (auto_approve & (labels == 0)).sum()
        slip_rate = slipped / auto_approve.sum() if auto_approve.sum() > 0 else 0

        # Assume human catches all errors in reviewed items
        final_errors = slipped  # only errors in auto-approved items
        final_accuracy = 1 - final_errors / n

        savings_pct = 1 - review_cost / cost_full_review

        strategies.append({
            "threshold": threshold,
            "auto_approve_pct": float(auto_approve.mean()),
            "auto_accuracy": float(auto_acc),
            "review_count": int(review_count),
            "review_cost": review_cost,
            "cost_savings_pct": float(savings_pct),
            "errors_slipped": int(slipped),
            "slip_rate": float(slip_rate),
            "final_accuracy": float(final_accuracy),
        })

    return {
        "full_review_cost": cost_full_review,
        "n_items": n,
        "base_accuracy": float(labels.mean()),
        "strategies": strategies,
    }


def cross_model_trust(samples):
    """Per-model analysis: which LLM should you trust for which task?"""
    model_bench = defaultdict(lambda: defaultdict(list))
    for s in samples:
        model = s["target_model"]
        bench = s["benchmark"]
        model_bench[model][bench].append(s)

    results = {}
    for model in sorted(model_bench.keys()):
        model_results = {}
        for bench in sorted(model_bench[model].keys()):
            samps = model_bench[model][bench]
            labels = np.array([s["is_correct"] for s in samps])
            scores = np.array([s["p_correct"] for s in samps])
            if len(set(labels)) < 2 or len(samps) < 10:
                continue
            auroc = float(roc_auc_score(labels, scores))
            acc = float(labels.mean())
            mean_conf = float(scores.mean())
            model_results[bench] = {
                "auroc": auroc,
                "accuracy": acc,
                "mean_confidence": mean_conf,
                "n": int(len(samps)),
                "calibrated": bool(abs(mean_conf - acc) < 0.1),
            }
        results[model] = model_results
    return results


def safety_warning_system(samples):
    """Design a 3-tier warning system for non-expert users.

    Green: Safe to use (confidence > 0.8, historically 95%+ correct)
    Yellow: Use with caution (0.5-0.8, verify key claims)
    Red: Do not rely on this (< 0.5, likely wrong)
    """
    labels = np.array([s["is_correct"] for s in samples])
    cal_scores = np.array([s.get("p_correct") or 0.5 for s in samples])

    tiers = [
        {"name": "GREEN (Safe)", "lo": 0.8, "hi": 1.01, "color": "#2ecc71"},
        {"name": "YELLOW (Caution)", "lo": 0.5, "hi": 0.8, "color": "#f39c12"},
        {"name": "RED (Don't Trust)", "lo": 0.0, "hi": 0.5, "color": "#e74c3c"},
    ]

    tier_results = []
    for tier in tiers:
        mask = (cal_scores >= tier["lo"]) & (cal_scores < tier["hi"])
        if mask.sum() == 0:
            tier_results.append({**tier, "n": 0, "accuracy": 0, "pct": 0})
            continue
        acc = labels[mask].mean()
        tier_results.append({
            **tier,
            "n": int(mask.sum()),
            "pct": float(mask.mean()),
            "accuracy": float(acc),
        })

    return tier_results


def plot_cost_benefit(assoc, fig_dir):
    """Plot cost-benefit analysis of UQ-guided review."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    strategies = assoc["strategies"]
    thresholds = [s["threshold"] for s in strategies]

    # Panel 1: Cost savings vs accuracy risk
    ax = axes[0]
    savings = [s["cost_savings_pct"] for s in strategies]
    slip_rates = [s["slip_rate"] for s in strategies]
    ax.plot(thresholds, savings, "o-", color="#2ecc71", linewidth=2, markersize=8,
            label="Cost Savings (%)")
    ax2 = ax.twinx()
    ax2.plot(thresholds, slip_rates, "s-", color="#e74c3c", linewidth=2, markersize=8,
             label="Error Slip Rate")
    ax.set_xlabel("Auto-Approve Threshold", fontsize=11)
    ax.set_ylabel("Cost Savings (%)", fontsize=11, color="#2ecc71")
    ax2.set_ylabel("Error Slip Rate", fontsize=11, color="#e74c3c")
    ax.set_title("(a) Cost vs Risk Tradeoff", fontsize=12)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: What you'd actually spend
    ax = axes[1]
    review_costs = [s["review_cost"] for s in strategies]
    full_cost = assoc["full_review_cost"]
    ax.bar(range(len(thresholds)), review_costs, color="#3498db", edgecolor="white")
    ax.axhline(y=full_cost, color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Full Review: ${full_cost:,}")
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f">{t}" for t in thresholds], fontsize=10)
    ax.set_xlabel("Auto-Approve If Confidence >", fontsize=11)
    ax.set_ylabel("Total Review Cost ($)", fontsize=11)
    ax.set_title("(b) Review Cost by Strategy", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # Panel 3: Final accuracy after human review of flagged items
    ax = axes[2]
    final_accs = [s["final_accuracy"] for s in strategies]
    base_acc = assoc["base_accuracy"]
    ax.bar(range(len(thresholds)), final_accs, color="#9b59b6", edgecolor="white")
    ax.axhline(y=base_acc, color="#FF9800", linestyle="--", linewidth=2,
               label=f"No Review: {base_acc:.1%}")
    ax.axhline(y=1.0, color="#2ecc71", linestyle="--", linewidth=2,
               label="Full Review: 100%")
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f">{t}" for t in thresholds], fontsize=10)
    ax.set_xlabel("Auto-Approve If Confidence >", fontsize=11)
    ax.set_ylabel("Final Accuracy (after human review of flagged)", fontsize=11)
    ax.set_title("(c) Final Accuracy by Strategy", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0.8, 1.02)
    ax.grid(alpha=0.3, axis="y")

    plt.suptitle("Junior Associate Replacement: Cost-Benefit of UQ-Guided Review",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(fig_dir, "cost_benefit_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_safety_tiers(tiers, fig_dir):
    """Traffic-light visualization for non-expert users."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = [t["name"] for t in tiers]
    pcts = [t["pct"] for t in tiers]
    accs = [t["accuracy"] for t in tiers]
    colors = [t["color"] for t in tiers]

    bars = ax.bar(range(len(names)), pcts, color=colors, edgecolor="white", width=0.6)

    for i, (bar, acc, n) in enumerate(zip(bars, accs, [t["n"] for t in tiers])):
        height = bar.get_height()
        ax.text(i, height + 0.02, f"Accuracy: {acc:.1%}\n(n={n})",
                ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=12, fontweight="bold")
    ax.set_ylabel("Fraction of Responses", fontsize=12)
    ax.set_title("Safety Tier System for Non-Expert Users\n"
                 "'Should I trust this AI answer?'", fontsize=14)
    ax.set_ylim(0, max(pcts) + 0.15)
    ax.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    path = os.path.join(fig_dir, "safety_tier_system.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_cross_model_heatmap(cross_model, fig_dir):
    """Heatmap: model × benchmark AUROC — which model to trust for what."""
    models = sorted(cross_model.keys())
    all_benches = set()
    for m in models:
        all_benches.update(cross_model[m].keys())
    benches = sorted(all_benches)

    matrix = np.full((len(models), len(benches)), np.nan)
    for i, m in enumerate(models):
        for j, b in enumerate(benches):
            if b in cross_model[m]:
                matrix[i, j] = cross_model[m][b]["accuracy"]

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels(benches, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)

    for i in range(len(models)):
        for j in range(len(benches)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if matrix[i, j] > 0.3 else "white")

    plt.colorbar(im, label="Accuracy")
    ax.set_title("Which LLM to Trust for Which Domain?\n"
                 "(UQ calibrator works across all)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(fig_dir, "cross_model_trust_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2")
    parser.add_argument("--output_dir", default="data/use_cases/professional_safety")
    parser.add_argument("--fig_dir", default="figures/professional_safety")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    samples = load_all_scored(args.scored_dir, args.smoke_test)
    print(f"Loaded {len(samples)} samples")

    # 1. Junior associate replacement
    print("\n=== Junior Associate Replacement Analysis ===")
    assoc = junior_associate_analysis(samples)
    print(f"Full review cost: ${assoc['full_review_cost']:,} ({assoc['n_items']} items × $20)")
    print(f"Base AI accuracy: {assoc['base_accuracy']:.1%}")
    print(f"\n{'Threshold':>10} {'Auto%':>8} {'Savings':>10} {'Slip Rate':>10} {'Final Acc':>10}")
    for s in assoc["strategies"]:
        print(f"{s['threshold']:>10.2f} {s['auto_approve_pct']:>8.0%} "
              f"{s['cost_savings_pct']:>10.0%} {s['slip_rate']:>10.2%} "
              f"{s['final_accuracy']:>10.1%}")

    # 2. Safety warning system
    print("\n=== Safety Tier System ===")
    tiers = safety_warning_system(samples)
    for t in tiers:
        print(f"  {t['name']:25s} | {t['pct']:.0%} of answers | accuracy: {t['accuracy']:.1%}")

    # 3. Cross-model trust
    print("\n=== Cross-Model Trust Analysis ===")
    cross_model = cross_model_trust(samples)
    for model in sorted(cross_model.keys()):
        benches = cross_model[model]
        mean_acc = np.mean([b["accuracy"] for b in benches.values()])
        mean_auroc = np.mean([b["auroc"] for b in benches.values()])
        n_calibrated = sum(1 for b in benches.values() if b["calibrated"])
        print(f"  {model:15s} | mean_acc={mean_acc:.3f} | mean_auroc={mean_auroc:.3f} "
              f"| {n_calibrated}/{len(benches)} benchmarks well-calibrated")

    # Figures
    print("\n=== Generating Figures ===")
    plot_cost_benefit(assoc, args.fig_dir)
    plot_safety_tiers(tiers, args.fig_dir)
    plot_cross_model_heatmap(cross_model, args.fig_dir)

    # Save
    output = {
        "junior_associate": assoc,
        "safety_tiers": tiers,
        "cross_model_trust": cross_model,
    }
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out_path = os.path.join(args.output_dir, "professional_safety_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved: {out_path}")

    # Narrative
    best = min(assoc["strategies"], key=lambda s: abs(s["threshold"] - 0.8))
    print("\n" + "=" * 70)
    print("USE CASE NARRATIVES")
    print("=" * 70)
    print(f"""
1. JUNIOR ASSOCIATE REPLACEMENT
   A law firm uses GPT-5 to draft research memos. Currently, a junior associate
   ($200/hr) reviews every output. With the UQ calibrator at threshold 0.8:
   - {best['auto_approve_pct']:.0%} of outputs auto-approved (no review needed)
   - Cost savings: {best['cost_savings_pct']:.0%} (${assoc['full_review_cost'] - best['review_cost']:,} saved)
   - Final accuracy: {best['final_accuracy']:.1%} (only {best['errors_slipped']} errors slip through)

2. PRO-SE SAFETY SYSTEM
   A self-represented litigant uses an AI legal assistant.
   Three-tier warning system:""")
    for t in tiers:
        print(f"   - {t['name']}: {t['pct']:.0%} of answers, {t['accuracy']:.1%} accurate")
    print(f"""
3. CROSS-MODEL TRUST
   Different LLMs excel at different domains. The UQ calibrator works across
   all of them, letting organizations route questions to the best model:""")
    for model in sorted(cross_model.keys()):
        best_bench = max(cross_model[model].items(), key=lambda x: x[1]["accuracy"])
        worst_bench = min(cross_model[model].items(), key=lambda x: x[1]["accuracy"])
        print(f"   - {model}: Best at {best_bench[0]} ({best_bench[1]['accuracy']:.0%}), "
              f"weakest at {worst_bench[0]} ({worst_bench[1]['accuracy']:.0%})")


if __name__ == "__main__":
    main()
