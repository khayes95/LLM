#!/usr/bin/env python3
"""UC-G: Human-in-the-Loop Escalation — "Route to Human When Uncertain".

For enterprise deployment, responses are routed into tiers:
  - GREEN: auto-send (high confidence)
  - YELLOW: spot-check by human (moderate confidence)
  - RED: human must approve before sending (low confidence)

This script quantifies how UQ enables efficient human review allocation:

Analyses:
1. Three-tier routing — volume, accuracy, error share per tier.
2. Error concentration — show that the red tier captures disproportionate errors.
3. Human review efficiency — accuracy improvement per reviewed query.
4. Threshold optimization — minimize total cost = auto_errors * err_cost + reviews * review_cost.
5. Workload reduction — at target accuracy, how much review is needed with/without UQ?
6. Method comparison — all baselines.
7. Per-model breakdown — optimal routing by target model.

Outputs:
    {output_dir}/uc_g_results.json — all metrics
    {fig_dir}/uc_g_tier_distribution.pdf — tier volume/accuracy/error share
    {fig_dir}/uc_g_efficiency.pdf — review efficiency curve

Usage:
    python scripts/uc_g_human_escalation.py
    python scripts/uc_g_human_escalation.py --scored_dir data/use_cases/scored_test_only_v2
    python scripts/uc_g_human_escalation.py --smoke_test
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
# Data loading
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
# Analysis 1: Three-tier routing
# ---------------------------------------------------------------------------

def three_tier_routing(samples, green_threshold=0.8, red_threshold=0.3,
                       score_key="p_correct"):
    """Split samples into green/yellow/red tiers."""
    valid = [s for s in samples if s.get(score_key) is not None]
    if not valid:
        return {}

    n_total = len(valid)
    n_wrong_total = sum(1 for s in valid if not s["is_correct"])

    green = [s for s in valid if s[score_key] >= green_threshold]
    red = [s for s in valid if s[score_key] < red_threshold]
    yellow = [s for s in valid if red_threshold <= s[score_key] < green_threshold]

    tiers = {}
    for tier_name, tier_samples in [("green", green), ("yellow", yellow), ("red", red)]:
        n = len(tier_samples)
        n_wrong = sum(1 for s in tier_samples if not s["is_correct"])
        accuracy = np.mean([s["is_correct"] for s in tier_samples]) if n > 0 else None
        error_share = n_wrong / n_wrong_total if n_wrong_total > 0 else 0.0

        tiers[tier_name] = {
            "n_samples": n,
            "pct_volume": n / n_total if n_total > 0 else 0.0,
            "n_wrong": n_wrong,
            "accuracy": float(accuracy) if accuracy is not None else None,
            "error_rate": n_wrong / n if n > 0 else 0.0,
            "error_share": float(error_share),
        }

    return {
        "green_threshold": green_threshold,
        "red_threshold": red_threshold,
        "n_total": n_total,
        "n_wrong_total": n_wrong_total,
        "base_accuracy": float(np.mean([s["is_correct"] for s in valid])),
        "tiers": tiers,
    }


# ---------------------------------------------------------------------------
# Analysis 2: Error concentration (Lorenz-style)
# ---------------------------------------------------------------------------

def error_concentration(samples, score_key="p_correct"):
    """How concentrated are errors in the lowest-confidence samples?

    Returns: at each percentile of queries (sorted by ascending confidence),
    what fraction of total errors is contained?
    """
    valid = [s for s in samples if s.get(score_key) is not None]
    if not valid:
        return []

    # Sort by ascending score (lowest confidence first)
    sorted_samples = sorted(valid, key=lambda s: s[score_key])
    n_total = len(sorted_samples)
    n_wrong_total = sum(1 for s in sorted_samples if not s["is_correct"])

    if n_wrong_total == 0:
        return []

    percentiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0]
    results = []
    for p in percentiles:
        k = max(1, int(n_total * p))
        subset = sorted_samples[:k]
        errors_in_subset = sum(1 for s in subset if not s["is_correct"])
        results.append({
            "query_percentile": p,
            "n_queries": k,
            "pct_queries": k / n_total,
            "errors_captured": errors_in_subset,
            "pct_errors_captured": errors_in_subset / n_wrong_total,
        })
    return results


# ---------------------------------------------------------------------------
# Analysis 3: Human review efficiency
# ---------------------------------------------------------------------------

def review_efficiency(samples, score_key="p_correct"):
    """If humans review queries in order of ascending confidence (lowest first),
    how many errors are caught per N reviews?

    Returns a curve: n_reviewed -> accuracy_of_remaining (auto-sent).
    """
    valid = [s for s in samples if s.get(score_key) is not None]
    if not valid:
        return []

    # Sort ascending (review lowest confidence first)
    sorted_samples = sorted(valid, key=lambda s: s[score_key])
    n_total = len(sorted_samples)
    n_wrong_total = sum(1 for s in sorted_samples if not s["is_correct"])

    review_counts = [0, 5, 10, 20, 50, 100, 200, 500]
    # Add percentage-based counts
    for pct in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        review_counts.append(int(n_total * pct))
    review_counts = sorted(set(c for c in review_counts if c <= n_total))

    results = []
    for n_reviewed in review_counts:
        reviewed = sorted_samples[:n_reviewed]
        remaining = sorted_samples[n_reviewed:]

        errors_caught = sum(1 for s in reviewed if not s["is_correct"])
        remaining_errors = n_wrong_total - errors_caught
        remaining_acc = (np.mean([s["is_correct"] for s in remaining])
                         if remaining else 1.0)

        # Compare to random review
        if n_reviewed > 0 and n_total > 0:
            random_errors_caught = n_wrong_total * (n_reviewed / n_total)
        else:
            random_errors_caught = 0

        results.append({
            "n_reviewed": n_reviewed,
            "pct_reviewed": n_reviewed / n_total if n_total > 0 else 0.0,
            "errors_caught": errors_caught,
            "errors_caught_pct": errors_caught / n_wrong_total if n_wrong_total > 0 else 0.0,
            "remaining_accuracy": float(remaining_acc),
            "random_errors_caught": float(random_errors_caught),
            "efficiency_vs_random": (errors_caught / random_errors_caught
                                     if random_errors_caught > 0 else float("inf")),
        })
    return results


# ---------------------------------------------------------------------------
# Analysis 4: Threshold optimization
# ---------------------------------------------------------------------------

def optimize_thresholds(samples, score_key="p_correct",
                        error_cost=10.0, review_cost=1.0):
    """Find optimal green/red thresholds to minimize total cost.

    Cost model:
    - Green tier errors: error_cost each
    - Yellow tier: review_cost each (all reviewed)
    - Red tier: review_cost each (all reviewed)
    - Reviewed samples have errors caught (cost = 0 for those)
    """
    valid = [s for s in samples if s.get(score_key) is not None]
    if not valid:
        return {}

    green_candidates = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    red_candidates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

    n_total = len(valid)
    best_cost = float("inf")
    best_config = None
    all_configs = []

    for gt in green_candidates:
        for rt in red_candidates:
            if rt >= gt:
                continue

            green = [s for s in valid if s[score_key] >= gt]
            yellow = [s for s in valid if rt <= s[score_key] < gt]
            red = [s for s in valid if s[score_key] < rt]

            # Green: auto-send, errors are costly
            green_errors = sum(1 for s in green if not s["is_correct"])
            # Yellow + Red: reviewed, errors caught
            n_reviewed = len(yellow) + len(red)

            total_cost = green_errors * error_cost + n_reviewed * review_cost
            cost_per_query = total_cost / n_total if n_total > 0 else 0.0

            config = {
                "green_threshold": gt,
                "red_threshold": rt,
                "n_green": len(green),
                "n_yellow": len(yellow),
                "n_red": len(red),
                "green_errors": green_errors,
                "n_reviewed": n_reviewed,
                "total_cost": float(total_cost),
                "cost_per_query": float(cost_per_query),
                "green_accuracy": float(np.mean([s["is_correct"] for s in green])) if green else None,
                "review_rate": n_reviewed / n_total if n_total > 0 else 0.0,
            }
            all_configs.append(config)

            if total_cost < best_cost:
                best_cost = total_cost
                best_config = config

    return {
        "error_cost": error_cost,
        "review_cost": review_cost,
        "best_config": best_config,
        "all_configs": all_configs,
    }


# ---------------------------------------------------------------------------
# Analysis 5: Workload reduction at target accuracy
# ---------------------------------------------------------------------------

def workload_at_target_accuracy(samples, target_accuracies=None,
                                 score_key="p_correct"):
    """At each target accuracy, how many queries must be reviewed?
    Compare UQ-guided review (lowest confidence first) vs random review.
    """
    if target_accuracies is None:
        target_accuracies = [0.90, 0.95, 0.97, 0.99]

    valid = [s for s in samples if s.get(score_key) is not None]
    if not valid:
        return {}

    sorted_samples = sorted(valid, key=lambda s: s[score_key])
    n_total = len(sorted_samples)
    n_wrong_total = sum(1 for s in sorted_samples if not s["is_correct"])
    base_acc = np.mean([s["is_correct"] for s in valid])

    results = {}
    for target in target_accuracies:
        if base_acc >= target:
            results[f"target_{target}"] = {
                "target_accuracy": target,
                "already_met": True,
                "base_accuracy": float(base_acc),
                "uq_reviews_needed": 0,
                "random_reviews_needed": 0,
            }
            continue

        # UQ-guided: review lowest confidence first
        uq_reviews = 0
        errors_caught = 0
        for i, s in enumerate(sorted_samples):
            if not s["is_correct"]:
                errors_caught += 1
            remaining = sorted_samples[i + 1:]
            if remaining:
                remaining_acc = np.mean([s["is_correct"] for s in remaining])
                if remaining_acc >= target:
                    uq_reviews = i + 1
                    break
        else:
            uq_reviews = n_total  # can't reach target

        # Random review: need to review enough to catch enough errors
        # After reviewing k random queries, expected remaining errors = n_wrong * (1 - k/n)
        # remaining accuracy = 1 - n_wrong*(1-k/n)/(n-k)
        random_reviews = n_total
        for k in range(n_total):
            remaining = n_total - k
            if remaining == 0:
                break
            expected_remaining_errors = n_wrong_total * (1 - k / n_total)
            expected_remaining_acc = 1 - expected_remaining_errors / remaining
            if expected_remaining_acc >= target:
                random_reviews = k
                break

        results[f"target_{target}"] = {
            "target_accuracy": target,
            "already_met": False,
            "base_accuracy": float(base_acc),
            "uq_reviews_needed": uq_reviews,
            "uq_review_pct": uq_reviews / n_total if n_total > 0 else 0.0,
            "random_reviews_needed": random_reviews,
            "random_review_pct": random_reviews / n_total if n_total > 0 else 0.0,
            "workload_reduction": (1 - uq_reviews / random_reviews
                                   if random_reviews > 0 else 0.0),
        }
    return results


# ---------------------------------------------------------------------------
# Analysis 6: Method comparison
# ---------------------------------------------------------------------------

def compare_escalation_methods(samples):
    """Compare methods by error concentration (area under error capture curve)."""
    methods = {
        "calibrator": "p_correct",
        "verbalized": "verbalized_confidence",
        "platt_verbalized": "p_platt_verbalized",
        "isotonic_verbalized": "p_isotonic_verbalized",
        "length_baseline": "p_length_baseline",
        "combined_baseline": "p_combined_baseline",
    }

    results = {}
    for name, key in methods.items():
        ec = error_concentration(samples, score_key=key)
        if not ec:
            continue

        # Area under error capture curve (higher = more concentrated errors)
        pct_queries = [e["pct_queries"] for e in ec]
        pct_errors = [e["pct_errors_captured"] for e in ec]
        _trapz = getattr(np, "trapezoid", np.trapz)
        auecc = float(_trapz(pct_errors, pct_queries))

        # Error capture at 20% review
        ec_20 = 0.0
        for e in ec:
            if abs(e["pct_queries"] - 0.20) < 0.05:
                ec_20 = e["pct_errors_captured"]
                break

        results[name] = {
            "area_under_error_capture_curve": auecc,
            "errors_captured_at_20pct_review": ec_20,
            "error_concentration": ec,
        }
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tier_distribution(all_model_results, fig_path):
    """Stacked bar chart: tier volume, accuracy, and error share."""
    n_models = len(all_model_results)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    model_names = {
        "gpt5mini": "GPT-5-mini",
        "gpt52": "GPT-5.2",
        "qwen35": "Qwen3.5",
    }

    targets = list(all_model_results.keys())
    x = np.arange(len(targets))
    w = 0.6

    tier_colors = {"green": "#2ecc71", "yellow": "#f39c12", "red": "#e74c3c"}

    # Plot 1: Volume distribution
    ax = axes[0]
    bottoms = np.zeros(len(targets))
    for tier in ["green", "yellow", "red"]:
        vals = [all_model_results[t]["three_tier"]["tiers"][tier]["pct_volume"]
                for t in targets]
        ax.bar(x, vals, w, bottom=bottoms, color=tier_colors[tier],
               label=tier.capitalize(), alpha=0.85)
        for i, v in enumerate(vals):
            if v > 0.05:
                ax.text(x[i], bottoms[i] + v / 2, f"{v:.0%}",
                        ha="center", va="center", fontsize=10, fontweight="bold")
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([model_names.get(t, t) for t in targets], fontsize=11)
    ax.set_ylabel("Fraction of queries", fontsize=12)
    ax.set_title("Volume per Tier", fontsize=13)
    ax.legend(fontsize=10)

    # Plot 2: Accuracy per tier
    ax = axes[1]
    for i, tier in enumerate(["green", "yellow", "red"]):
        vals = []
        for t in targets:
            acc = all_model_results[t]["three_tier"]["tiers"][tier]["accuracy"]
            vals.append(acc if acc is not None else 0)
        positions = x + (i - 1) * w / 3
        ax.bar(positions, vals, w / 3, color=tier_colors[tier],
               label=tier.capitalize(), alpha=0.85)
        for j, v in enumerate(vals):
            if v > 0:
                ax.text(positions[j], v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([model_names.get(t, t) for t in targets], fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy per Tier", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)

    # Plot 3: Error share per tier
    ax = axes[2]
    bottoms = np.zeros(len(targets))
    for tier in ["red", "yellow", "green"]:  # reversed for visual emphasis
        vals = [all_model_results[t]["three_tier"]["tiers"][tier]["error_share"]
                for t in targets]
        ax.bar(x, vals, w, bottom=bottoms, color=tier_colors[tier],
               label=tier.capitalize(), alpha=0.85)
        for i, v in enumerate(vals):
            if v > 0.05:
                ax.text(x[i], bottoms[i] + v / 2, f"{v:.0%}",
                        ha="center", va="center", fontsize=10, fontweight="bold")
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([model_names.get(t, t) for t in targets], fontsize=11)
    ax.set_ylabel("Share of total errors", fontsize=12)
    ax.set_title("Error Concentration by Tier", fontsize=13)
    ax.legend(fontsize=10)

    fig.suptitle("UC-G: Human Escalation — Three-Tier Routing",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {fig_path}")


def plot_efficiency(all_model_results, fig_path):
    """Review efficiency: errors caught vs % reviewed, UQ vs random."""
    n_models = len(all_model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), squeeze=False)

    model_names = {
        "gpt5mini": "GPT-5-mini",
        "gpt52": "GPT-5.2",
        "qwen35": "Qwen3.5",
    }

    for col, (target, data) in enumerate(all_model_results.items()):
        ax = axes[0, col]
        re = data["review_efficiency"]

        pct_reviewed = [r["pct_reviewed"] for r in re]
        errors_caught = [r["errors_caught_pct"] for r in re]
        random_caught = [r["pct_reviewed"] for r in re]  # random = diagonal

        ax.plot(pct_reviewed, errors_caught, "o-", color="C0", linewidth=2.5,
                label="UQ-guided review")
        ax.plot(pct_reviewed, random_caught, ":", color="gray", linewidth=2,
                label="Random review")
        ax.fill_between(pct_reviewed, random_caught, errors_caught,
                        alpha=0.15, color="C0")

        # Mark key points
        for r in re:
            if abs(r["pct_reviewed"] - 0.20) < 0.03:
                ax.annotate(f"{r['errors_caught_pct']:.0%} errors\nat {r['pct_reviewed']:.0%} review",
                            xy=(r["pct_reviewed"], r["errors_caught_pct"]),
                            xytext=(15, -10), textcoords="offset points",
                            fontsize=9, arrowprops=dict(arrowstyle="->", color="C0"))
                break

        ax.plot([0, 1], [0, 1], ":", color="gray", alpha=0.3)
        ax.set_xlabel("Fraction of queries reviewed", fontsize=12)
        ax.set_ylabel("Fraction of errors caught", fontsize=12)
        ax.set_title(f"{model_names.get(target, target)} (N={data['n_samples']})",
                     fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle("UC-G: Human Review Efficiency — UQ-Guided vs Random",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UC-G: Human-in-the-Loop Escalation")
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
        base_acc = float(np.mean([s["is_correct"] for s in samples]))

        print(f"\n{'='*70}")
        print(f"UC-G: Human Escalation — {target_names.get(target, target)}")
        print(f"{'='*70}")
        print(f"Samples: {n_total}, Base accuracy: {base_acc:.3f}, "
              f"Wrong: {n_wrong} ({n_wrong/n_total:.1%})")

        # --- 1. Three-tier routing ---
        print(f"\n--- Three-Tier Routing (green>=0.8, red<0.3) ---")
        tt = three_tier_routing(samples, green_threshold=0.8, red_threshold=0.3)

        print(f"  {'Tier':<8} {'Volume':>7} {'Vol%':>6} {'Wrong':>6} "
              f"{'Accuracy':>9} {'ErrRate':>8} {'ErrShare':>9}")
        print(f"  {'-'*57}")
        for tier_name in ["green", "yellow", "red"]:
            td = tt["tiers"][tier_name]
            acc_s = f"{td['accuracy']:.3f}" if td['accuracy'] is not None else "N/A"
            print(f"  {tier_name:<8} {td['n_samples']:>7} {td['pct_volume']:>5.1%} "
                  f"{td['n_wrong']:>6} {acc_s:>9} {td['error_rate']:>7.3f} "
                  f"{td['error_share']:>8.1%}")

        # --- 2. Error concentration ---
        print(f"\n--- Error Concentration (lowest confidence first) ---")
        ec = error_concentration(samples)

        print(f"  {'Review%':>8} {'Queries':>8} {'ErrsCaught':>11} {'ErrCaught%':>11}")
        print(f"  {'-'*42}")
        for e in ec:
            print(f"  {e['pct_queries']:>7.0%} {e['n_queries']:>8} "
                  f"{e['errors_captured']:>11} {e['pct_errors_captured']:>10.1%}")

        # --- 3. Review efficiency ---
        print(f"\n--- Review Efficiency (UQ-guided vs random) ---")
        re = review_efficiency(samples)

        print(f"  {'Reviewed':>9} {'Rev%':>6} {'ErrsCaught':>11} {'Caught%':>8} "
              f"{'RandCaught':>11} {'Efficiency':>11}")
        print(f"  {'-'*60}")
        for r in re:
            if r["n_reviewed"] == 0:
                continue
            eff_s = f"{r['efficiency_vs_random']:.1f}x" if r["efficiency_vs_random"] < 100 else ">100x"
            print(f"  {r['n_reviewed']:>9} {r['pct_reviewed']:>5.1%} "
                  f"{r['errors_caught']:>11} {r['errors_caught_pct']:>7.1%} "
                  f"{r['random_errors_caught']:>11.1f} {eff_s:>11}")

        # --- 4. Threshold optimization ---
        print(f"\n--- Optimal Thresholds (error_cost=10x) ---")
        opt = optimize_thresholds(samples, error_cost=10.0, review_cost=1.0)
        if opt.get("best_config"):
            bc = opt["best_config"]
            print(f"  Green >= {bc['green_threshold']:.2f}, "
                  f"Red < {bc['red_threshold']:.2f}")
            print(f"  Green: {bc['n_green']} ({bc['n_green']/n_total:.1%}), "
                  f"Yellow: {bc['n_yellow']}, Red: {bc['n_red']}")
            print(f"  Green accuracy: {bc['green_accuracy']:.3f}, "
                  f"Review rate: {bc['review_rate']:.1%}")
            print(f"  Cost/query: {bc['cost_per_query']:.3f}")

        # Also at 50x
        opt_50 = optimize_thresholds(samples, error_cost=50.0, review_cost=1.0)

        # --- 5. Workload reduction ---
        print(f"\n--- Workload Reduction at Target Accuracy ---")
        wl = workload_at_target_accuracy(samples)

        print(f"  {'Target':>7} {'UQ Reviews':>11} {'UQ Rev%':>8} "
              f"{'Rand Reviews':>13} {'Rand Rev%':>10} {'Reduction':>10}")
        print(f"  {'-'*63}")
        for key, wd in wl.items():
            if wd.get("already_met"):
                print(f"  {wd['target_accuracy']:>6.0%} Already met (base={wd['base_accuracy']:.3f})")
            else:
                print(f"  {wd['target_accuracy']:>6.0%} {wd['uq_reviews_needed']:>11} "
                      f"{wd['uq_review_pct']:>7.1%} {wd['random_reviews_needed']:>13} "
                      f"{wd['random_review_pct']:>9.1%} {wd['workload_reduction']:>9.1%}")

        # --- 6. Method comparison ---
        print(f"\n--- Method Comparison (Error Concentration) ---")
        mc = compare_escalation_methods(samples)

        print(f"  {'Method':<25} {'AUECC':>7} {'ErrAt20%Rev':>12}")
        print(f"  {'-'*46}")
        for method, md in sorted(mc.items(),
                                 key=lambda x: x[1]["area_under_error_capture_curve"],
                                 reverse=True):
            print(f"  {method:<25} {md['area_under_error_capture_curve']:>7.3f} "
                  f"{md['errors_captured_at_20pct_review']:>11.1%}")

        # Store
        all_results[target] = {
            "n_samples": n_total,
            "n_wrong": n_wrong,
            "base_accuracy": base_acc,
            "three_tier": tt,
            "error_concentration": ec,
            "review_efficiency": re,
            "threshold_optimization_10x": opt,
            "threshold_optimization_50x": opt_50,
            "workload_reduction": wl,
            "method_comparison": mc,
        }

    if not all_results:
        print("\nERROR: No scored data found.")
        return

    # --- Figures ---
    plot_tier_distribution(all_results, f"{args.fig_dir}/uc_g_tier_distribution.pdf")
    plot_efficiency(all_results, f"{args.fig_dir}/uc_g_efficiency.pdf")

    # --- Save JSON ---
    out_path = f"{args.output_dir}/uc_g_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- Final Summary ---
    print(f"\n{'='*70}")
    print("UC-G Summary: Human Escalation Value")
    print(f"{'='*70}")
    for target in all_results:
        d = all_results[target]
        tt = d["three_tier"]
        mc = d["method_comparison"]
        cal_auecc = mc.get("calibrator", {}).get("area_under_error_capture_curve", 0)

        green_t = tt["tiers"]["green"]
        red_t = tt["tiers"]["red"]

        print(f"  {target_names.get(target, target)}:")
        green_acc_s = f"{green_t['accuracy']:.3f}" if green_t['accuracy'] is not None else "N/A"
        red_acc_s = f"{red_t['accuracy']:.3f}" if red_t['accuracy'] is not None else "N/A"
        print(f"    Green tier: {green_t['pct_volume']:.0%} volume, "
              f"{green_acc_s} accuracy, "
              f"{green_t['error_share']:.0%} of errors")
        print(f"    Red tier: {red_t['pct_volume']:.0%} volume, "
              f"{red_t['error_share']:.0%} of errors (captured)")
        print(f"    AUECC: {cal_auecc:.3f}")

        # Workload headline
        wl = d["workload_reduction"]
        for key, wd in wl.items():
            if wd.get("target_accuracy") == 0.95 and not wd.get("already_met"):
                print(f"    95% accuracy: review {wd['uq_review_pct']:.0%} "
                      f"(vs {wd['random_review_pct']:.0%} random) — "
                      f"{wd['workload_reduction']:.0%} less work")


if __name__ == "__main__":
    main()
