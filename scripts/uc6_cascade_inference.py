#!/usr/bin/env python3
"""UC6 [LONGSHOT]: Cascade Inference with UQ Gating.

3-tier pipeline: cheap → expensive → human, with calibrator deciding when to stop.
Extends UC2 by adding a human review tier and optimizing over 2 thresholds.

Usage:
    python scripts/uc6_cascade_inference.py
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_scored(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def build_paired(mini_samples, gpt52_samples):
    """Match samples by (benchmark, id)."""
    mini_by_id = {}
    for s in mini_samples:
        mini_by_id[(s["benchmark"], s["id"])] = s

    paired = []
    for s52 in gpt52_samples:
        key = (s52["benchmark"], s52["id"])
        if key in mini_by_id:
            sm = mini_by_id[key]
            paired.append({
                "id": s52["id"],
                "benchmark": s52["benchmark"],
                "mini_correct": sm["is_correct"],
                "mini_p": sm["p_correct"],
                "mini_verb": sm.get("verbalized_confidence"),
                "gpt52_correct": s52["is_correct"],
                "gpt52_p": s52["p_correct"],
                "gpt52_verb": s52.get("verbalized_confidence"),
            })
    return paired


def cascade_simulation(paired, theta1, theta2, cost_ratio=4.75, human_cost=100.0,
                        human_accuracy=1.0, score_key_1="mini_p", score_key_2="gpt52_p"):
    """Simulate 3-tier cascade with given thresholds.

    Tier 1: GPT-5-mini. If score >= theta1, accept. Cost = 1.0
    Tier 2: GPT-5.2. If score >= theta2, accept. Cost = 1.0 + cost_ratio
    Tier 3: Human. Always correct (or human_accuracy). Cost = 1.0 + cost_ratio + human_cost
    """
    tier_counts = [0, 0, 0]
    correct = 0
    total_cost = 0.0

    for p in paired:
        s1 = p.get(score_key_1) or 0.5
        s2 = p.get(score_key_2) or 0.5

        if s1 >= theta1:
            # Accept tier 1
            tier_counts[0] += 1
            correct += p["mini_correct"]
            total_cost += 1.0
        elif s2 >= theta2:
            # Accept tier 2
            tier_counts[1] += 1
            correct += p["gpt52_correct"]
            total_cost += 1.0 + cost_ratio
        else:
            # Human review
            tier_counts[2] += 1
            correct += int(np.random.random() < human_accuracy)
            total_cost += 1.0 + cost_ratio + human_cost

    n = len(paired)
    return {
        "theta1": float(theta1),
        "theta2": float(theta2),
        "accuracy": float(correct / n),
        "avg_cost": float(total_cost / n),
        "tier1_frac": float(tier_counts[0] / n),
        "tier2_frac": float(tier_counts[1] / n),
        "tier3_frac": float(tier_counts[2] / n),
        "tier_counts": tier_counts,
    }


def find_pareto_frontier(results):
    """Find Pareto-optimal points (minimize cost, maximize accuracy)."""
    sorted_by_cost = sorted(results, key=lambda r: r["avg_cost"])
    pareto = []
    best_acc = -1
    for r in sorted_by_cost:
        if r["accuracy"] > best_acc:
            pareto.append(r)
            best_acc = r["accuracy"]
    return pareto


def plot_cascade(pareto_cal, pareto_verb, baselines, output_path):
    """Plot cascade results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Cost vs Accuracy (Pareto frontier)
    ax = axes[0]
    if pareto_cal:
        costs = [p["avg_cost"] for p in pareto_cal]
        accs = [p["accuracy"] for p in pareto_cal]
        ax.plot(costs, accs, "C0-o", markersize=4, label="Calibrator cascade", linewidth=2)

    if pareto_verb:
        costs = [p["avg_cost"] for p in pareto_verb]
        accs = [p["accuracy"] for p in pareto_verb]
        ax.plot(costs, accs, "C1--s", markersize=4, label="Verbalized cascade", linewidth=1.5)

    for name, data in baselines.items():
        ax.scatter([data["cost"]], [data["accuracy"]], marker="^", s=100,
                   label=f"{name} ({data['accuracy']:.1%})", zorder=5)

    ax.set_xlabel("Average Cost per Query (1.0 = GPT-5-mini)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Cascade Inference: Cost vs Accuracy", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Tier distribution at optimal points
    ax = axes[1]
    if pareto_cal:
        # Pick 5 representative points
        indices = np.linspace(0, len(pareto_cal)-1, min(5, len(pareto_cal)), dtype=int)
        selected = [pareto_cal[i] for i in indices]

        labels_cascade = [f"θ1={p['theta1']:.1f}\nθ2={p['theta2']:.1f}" for p in selected]
        tier1 = [p["tier1_frac"] for p in selected]
        tier2 = [p["tier2_frac"] for p in selected]
        tier3 = [p["tier3_frac"] for p in selected]

        x = np.arange(len(selected))
        ax.bar(x, tier1, label="Tier 1: GPT-5-mini", color="C2")
        ax.bar(x, tier2, bottom=tier1, label="Tier 2: GPT-5.2", color="C1")
        ax.bar(x, tier3, bottom=[t1+t2 for t1, t2 in zip(tier1, tier2)],
               label="Tier 3: Human", color="C3")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_cascade, fontsize=8)
        for i, p in enumerate(selected):
            ax.text(i, 1.02, f"Acc: {p['accuracy']:.1%}\nCost: {p['avg_cost']:.1f}",
                    ha="center", fontsize=7)

    ax.set_ylabel("Fraction of Queries", fontsize=12)
    ax.set_title("Query Distribution Across Tiers", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2")
    parser.add_argument("--output_dir", default="data/use_cases/results_test_only_v2")
    parser.add_argument("--fig_dir", default="figures/use_cases_v2")
    parser.add_argument("--human_accuracy", type=float, default=1.0)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    mini_samples = load_scored(Path(args.scored_dir) / "gpt5mini_scored.jsonl")
    gpt52_samples = load_scored(Path(args.scored_dir) / "gpt52_scored.jsonl")

    if not mini_samples or not gpt52_samples:
        print("ERROR: Need both gpt5mini and gpt52 scored data")
        return

    paired = build_paired(mini_samples, gpt52_samples)

    print("=" * 70)
    print("UC6 [LONGSHOT]: Cascade Inference with UQ Gating")
    print("=" * 70)
    print(f"Paired samples: {len(paired)}")
    print(f"Human accuracy assumption: {args.human_accuracy:.0%}")

    if len(paired) < 10:
        print("ERROR: Too few paired samples")
        return

    mini_acc = np.mean([p["mini_correct"] for p in paired])
    gpt52_acc = np.mean([p["gpt52_correct"] for p in paired])
    cost_ratio = 4.75

    # Grid search over (theta1, theta2)
    print("\nGrid search over cascade thresholds...")
    np.random.seed(42)

    all_results_cal = []
    all_results_verb = []
    thetas = np.arange(0.1, 1.0, 0.05)

    for t1 in thetas:
        for t2 in thetas:
            r = cascade_simulation(paired, t1, t2, cost_ratio=cost_ratio,
                                    human_cost=100.0, human_accuracy=args.human_accuracy)
            all_results_cal.append(r)

            # Verbalized version
            rv = cascade_simulation(paired, t1, t2, cost_ratio=cost_ratio,
                                     human_cost=100.0, human_accuracy=args.human_accuracy,
                                     score_key_1="mini_verb", score_key_2="gpt52_verb")
            all_results_verb.append(rv)

    pareto_cal = find_pareto_frontier(all_results_cal)
    pareto_verb = find_pareto_frontier(all_results_verb)

    # Baselines
    baselines = {
        "All GPT-5-mini": {"cost": 1.0, "accuracy": mini_acc},
        "All GPT-5.2": {"cost": 1.0 + cost_ratio, "accuracy": gpt52_acc},
        "All Human": {"cost": 1.0 + cost_ratio + 100.0, "accuracy": args.human_accuracy},
    }

    # Print key Pareto points
    print(f"\n  Calibrator Pareto frontier (top 8):")
    print(f"  {'θ1':>5} {'θ2':>5} {'Accuracy':>10} {'Cost':>8} {'T1%':>6} {'T2%':>6} {'Human%':>7}")
    print(f"  {'-'*52}")
    for p in pareto_cal[:8]:
        print(f"  {p['theta1']:>5.2f} {p['theta2']:>5.2f} {p['accuracy']:>10.3f} "
              f"{p['avg_cost']:>8.1f} {p['tier1_frac']:>6.1%} {p['tier2_frac']:>6.1%} "
              f"{p['tier3_frac']:>7.1%}")

    # 2-tier cascade: just cheap vs expensive (no human), for cleaner cost-accuracy curve
    print(f"\n  --- 2-Tier Cascade (no human tier) ---")
    two_tier_cal = []
    two_tier_verb = []
    for t1 in np.arange(0.05, 1.0, 0.025):
        n_cheap = 0
        n_exp = 0
        correct = 0
        for p in paired:
            s1 = p.get("mini_p") or 0.5
            if s1 >= t1:
                n_cheap += 1
                correct += p["mini_correct"]
            else:
                n_exp += 1
                correct += p["gpt52_correct"]
        n = len(paired)
        cost = (n_cheap * 1.0 + n_exp * (1.0 + cost_ratio)) / n
        two_tier_cal.append({
            "threshold": float(t1),
            "accuracy": float(correct / n),
            "cost": float(cost),
            "pct_cheap": float(n_cheap / n),
        })

        # Verbalized version
        n_cheap_v = 0
        n_exp_v = 0
        correct_v = 0
        for p in paired:
            sv = p.get("mini_verb") or 0.5
            if sv >= t1:
                n_cheap_v += 1
                correct_v += p["mini_correct"]
            else:
                n_exp_v += 1
                correct_v += p["gpt52_correct"]
        cost_v = (n_cheap_v * 1.0 + n_exp_v * (1.0 + cost_ratio)) / n
        two_tier_verb.append({
            "threshold": float(t1),
            "accuracy": float(correct_v / n),
            "cost": float(cost_v),
            "pct_cheap": float(n_cheap_v / n),
        })

    # Break-even: match GPT-5.2 accuracy with minimum cost
    break_even = None
    for pt in two_tier_cal:
        if pt["accuracy"] >= gpt52_acc - 0.005:
            break_even = pt
            break

    if break_even:
        full_cost = 1.0 + cost_ratio
        savings = (1 - break_even["cost"] / full_cost) * 100
        print(f"  Break-even: matches GPT-5.2 accuracy ({gpt52_acc:.1%}) at "
              f"threshold={break_even['threshold']:.2f}")
        print(f"    {break_even['pct_cheap']:.0%} queries handled by cheap model")
        print(f"    Cost savings: {savings:.0f}% vs always using GPT-5.2")
    else:
        print(f"  2-tier cascade never reaches GPT-5.2 accuracy ({gpt52_acc:.1%})")

    # Sensitivity analysis for human accuracy
    print(f"\n  Sensitivity: Human accuracy impact")
    for ha in [0.90, 0.95, 1.0]:
        best = None
        for t1 in [0.5, 0.6, 0.7, 0.8]:
            for t2 in [0.3, 0.4, 0.5, 0.6]:
                r = cascade_simulation(paired, t1, t2, cost_ratio=cost_ratio,
                                        human_cost=100.0, human_accuracy=ha)
                if best is None or (r["accuracy"] >= 0.90 and r["avg_cost"] < best["avg_cost"]):
                    best = r
        if best:
            print(f"    Human@{ha:.0%}: best cascade acc={best['accuracy']:.3f}, "
                  f"cost={best['avg_cost']:.1f}, human_frac={best['tier3_frac']:.1%}")

    # Plot
    plot_cascade(pareto_cal, pareto_verb, baselines,
                 f"{args.fig_dir}/uc6_cascade.pdf")

    # Save
    out_path = f"{args.output_dir}/uc6_results.json"
    save_data = {
        "n_paired": len(paired),
        "mini_accuracy": float(mini_acc),
        "gpt52_accuracy": float(gpt52_acc),
        "cost_ratio": cost_ratio,
        "human_accuracy": args.human_accuracy,
        "pareto_frontier_calibrator": pareto_cal[:10],
        "pareto_frontier_verbalized": pareto_verb[:10],
        "baselines": baselines,
        "two_tier_calibrator": two_tier_cal,
        "two_tier_verbalized": two_tier_verb,
    }
    if break_even:
        save_data["break_even"] = break_even
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
