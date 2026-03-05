#!/usr/bin/env python3
"""UC2: Model Routing — cheap model for easy, expensive for hard.

Score GPT-5-mini response with calibrator. If confident → keep cheap answer.
Otherwise → use GPT-5.2 answer. Compute accuracy vs cost tradeoff.

Usage:
    python scripts/uc2_model_routing.py
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

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


def build_paired_data(mini_samples, gpt52_samples):
    """Match samples by ID to get paired (cheap, expensive) answers."""
    mini_by_id = {}
    for s in mini_samples:
        key = (s["benchmark"], s["id"])
        mini_by_id[key] = s

    paired = []
    for s52 in gpt52_samples:
        key = (s52["benchmark"], s52["id"])
        if key in mini_by_id:
            sm = mini_by_id[key]
            paired.append({
                "id": s52["id"],
                "benchmark": s52["benchmark"],
                "mini_correct": sm["is_correct"],
                "mini_p_correct": sm["p_correct"],
                "mini_verbalized": sm.get("verbalized_confidence"),
                "mini_tokens": sm.get("total_tokens") or 0,
                "gpt52_correct": s52["is_correct"],
                "gpt52_tokens": s52.get("total_tokens") or 0,
            })
    return paired


def routing_analysis(paired, score_key="mini_p_correct", n_thresholds=200):
    """Compute routing curve: sweep threshold, track accuracy and cost."""
    if not paired:
        return []

    # Cost model: GPT-5.2 costs ~4.75x GPT-5-mini per query
    # Use actual token counts where available
    avg_mini_tokens = np.mean([p["mini_tokens"] for p in paired if p["mini_tokens"] > 0]) or 1000
    avg_52_tokens = np.mean([p["gpt52_tokens"] for p in paired if p["gpt52_tokens"] > 0]) or 5000
    cost_ratio = (avg_52_tokens / avg_mini_tokens) * 3  # 3x per-token premium for reasoning model

    results = []
    thresholds = np.linspace(0, 1, n_thresholds + 1)

    for t in thresholds:
        n_keep_cheap = 0
        n_escalate = 0
        correct = 0

        for p in paired:
            score = p.get(score_key, 0.5) or 0.5
            if score >= t:
                # Keep cheap model answer
                n_keep_cheap += 1
                correct += p["mini_correct"]
            else:
                # Escalate to expensive model
                n_escalate += 1
                correct += p["gpt52_correct"]

        n_total = len(paired)
        accuracy = correct / n_total
        escalation_rate = n_escalate / n_total
        # Normalized cost: all-cheap = 1.0, all-expensive = cost_ratio
        cost = (n_keep_cheap * 1.0 + n_escalate * cost_ratio) / n_total

        results.append({
            "threshold": float(t),
            "accuracy": float(accuracy),
            "escalation_rate": float(escalation_rate),
            "cost_normalized": float(cost),
            "n_cheap": n_keep_cheap,
            "n_expensive": n_escalate,
        })

    return results, float(cost_ratio)


def plot_routing(results_by_method, cost_ratio, mini_acc, gpt52_acc, output_path):
    """Plot routing curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Cost vs Accuracy
    ax = axes[0]
    styles = {
        "Calibrator": {"color": "C0", "linewidth": 2.5},
        "Verbalized": {"color": "C1", "linewidth": 2, "linestyle": "--"},
        "Random": {"color": "gray", "linewidth": 1, "linestyle": ":"},
        "Oracle": {"color": "C4", "linewidth": 1.5, "linestyle": "-."},
    }
    for method, (curve, _) in results_by_method.items():
        costs = [p["cost_normalized"] for p in curve]
        accs = [p["accuracy"] for p in curve]
        ax.plot(costs, accs, label=method, **styles.get(method, {}))

    ax.axhline(y=mini_acc, color="C3", linestyle=":", alpha=0.7, label=f"All GPT-5-mini ({mini_acc:.1%})")
    ax.axhline(y=gpt52_acc, color="C4", linestyle=":", alpha=0.7, label=f"All GPT-5.2 ({gpt52_acc:.1%})")
    ax.set_xlabel("Normalized Cost (1.0 = all cheap)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Model Routing: Cost vs Accuracy", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Escalation rate vs Accuracy
    ax = axes[1]
    for method, (curve, _) in results_by_method.items():
        esc = [p["escalation_rate"] for p in curve]
        accs = [p["accuracy"] for p in curve]
        ax.plot(esc, accs, label=method, **styles.get(method, {}))

    ax.set_xlabel("Escalation Rate (fraction sent to GPT-5.2)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Escalation Rate vs Accuracy", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2")
    parser.add_argument("--output_dir", default="data/use_cases/results_test_only_v2")
    parser.add_argument("--fig_dir", default="figures/use_cases_v2")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    mini_path = Path(args.scored_dir) / "gpt5mini_scored.jsonl"
    gpt52_path = Path(args.scored_dir) / "gpt52_scored.jsonl"

    if not mini_path.exists() or not gpt52_path.exists():
        print(f"ERROR: Need both {mini_path} and {gpt52_path}")
        return

    mini_samples = load_scored(mini_path)
    gpt52_samples = load_scored(gpt52_path)

    print("=" * 70)
    print("UC2: Model Routing (Cost Optimization)")
    print("=" * 70)

    # Build paired data
    paired = build_paired_data(mini_samples, gpt52_samples)
    print(f"Paired samples: {len(paired)}")

    if len(paired) < 10:
        print("ERROR: Too few paired samples for routing analysis")
        return

    mini_acc = np.mean([p["mini_correct"] for p in paired])
    gpt52_acc = np.mean([p["gpt52_correct"] for p in paired])
    print(f"GPT-5-mini accuracy (paired): {mini_acc:.3f}")
    print(f"GPT-5.2 accuracy (paired): {gpt52_acc:.3f}")

    # Per-benchmark paired counts
    bench_counts = defaultdict(int)
    for p in paired:
        bench_counts[p["benchmark"]] += 1
    print(f"\nPaired samples by benchmark:")
    for b in sorted(bench_counts, key=bench_counts.get, reverse=True):
        print(f"  {b}: {bench_counts[b]}")

    # Routing with calibrator
    results_by_method = {}
    cal_curve, cost_ratio = routing_analysis(paired, "mini_p_correct")
    results_by_method["Calibrator"] = (cal_curve, cost_ratio)

    # Routing with verbalized confidence
    verb_paired = [p for p in paired if p.get("mini_verbalized") is not None]
    if verb_paired:
        verb_curve, _ = routing_analysis(verb_paired, "mini_verbalized")
        results_by_method["Verbalized"] = (verb_curve, cost_ratio)

    # Random routing
    rng = np.random.RandomState(42)
    random_paired = [{**p, "random_score": rng.random()} for p in paired]
    rand_curve, _ = routing_analysis(random_paired, "random_score")
    results_by_method["Random"] = (rand_curve, cost_ratio)

    # Key metrics
    print(f"\nCost ratio: GPT-5.2 = {cost_ratio:.2f}x GPT-5-mini")
    print(f"\n  {'Method':<15} {'Acc@50%esc':>10} {'Acc@30%esc':>10} {'Cost@isoAcc':>12}")
    print(f"  {'-'*50}")

    output_summary = {"n_paired": len(paired), "cost_ratio": cost_ratio,
                      "mini_accuracy": float(mini_acc), "gpt52_accuracy": float(gpt52_acc),
                      "methods": {}}

    for method, (curve, _) in results_by_method.items():
        # Accuracy at 50% escalation
        acc_50 = None
        for p in curve:
            if abs(p["escalation_rate"] - 0.50) < 0.02:
                acc_50 = p["accuracy"]
                break

        # Accuracy at 30% escalation
        acc_30 = None
        for p in curve:
            if abs(p["escalation_rate"] - 0.30) < 0.02:
                acc_30 = p["accuracy"]
                break

        # Cost to match GPT-5.2 accuracy
        cost_iso = None
        for p in curve:
            if p["accuracy"] >= gpt52_acc - 0.005:
                cost_iso = p["cost_normalized"]
                break

        a50_str = f"{acc_50:.3f}" if acc_50 else "N/A"
        a30_str = f"{acc_30:.3f}" if acc_30 else "N/A"
        cost_str = f"{cost_iso:.2f}x" if cost_iso else "N/A"
        savings_str = f"({(1 - cost_iso/cost_ratio)*100:.0f}% saved)" if cost_iso else ""
        print(f"  {method:<15} {a50_str:>10} {a30_str:>10} {cost_str:>12} {savings_str}")

        output_summary["methods"][method] = {
            "accuracy_at_50pct_escalation": acc_50,
            "accuracy_at_30pct_escalation": acc_30,
            "cost_at_iso_accuracy": cost_iso,
        }

    # Oracle routing: route to the model that's actually correct
    oracle_paired = [{**p, "oracle_score": float(p["mini_correct"])} for p in paired]
    oracle_curve, _ = routing_analysis(oracle_paired, "oracle_score")
    results_by_method["Oracle"] = (oracle_curve, cost_ratio)

    oracle_correct = sum(
        max(p["mini_correct"], p["gpt52_correct"]) for p in paired
    )
    oracle_acc = oracle_correct / len(paired)
    both_correct = sum(p['mini_correct'] and p['gpt52_correct'] for p in paired)
    both_wrong = sum(not p['mini_correct'] and not p['gpt52_correct'] for p in paired)
    only_mini = sum(p['mini_correct'] and not p['gpt52_correct'] for p in paired)
    only_52 = sum(not p['mini_correct'] and p['gpt52_correct'] for p in paired)

    print(f"\n  Oracle (always pick better): {oracle_acc:.3f}")
    print(f"  Both correct: {both_correct} ({both_correct/len(paired):.1%})")
    print(f"  Both wrong: {both_wrong} ({both_wrong/len(paired):.1%})")
    print(f"  Only mini correct: {only_mini} ({only_mini/len(paired):.1%})")
    print(f"  Only 5.2 correct: {only_52} ({only_52/len(paired):.1%})")

    output_summary["oracle_accuracy"] = float(oracle_acc)
    output_summary["agreement_stats"] = {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "only_mini_correct": only_mini,
        "only_52_correct": only_52,
    }

    # Break-even analysis: find threshold where routing achieves GPT-5.2 accuracy
    # at minimum cost
    cal_curve_data = results_by_method["Calibrator"][0]
    break_even = None
    for p in cal_curve_data:
        if p["accuracy"] >= gpt52_acc - 0.005:
            break_even = p
            break

    print(f"\n--- Break-Even Analysis ---")
    if break_even:
        savings_pct = (1 - break_even["cost_normalized"] / cost_ratio) * 100
        cheap_pct = break_even["n_cheap"] / len(paired) * 100
        print(f"  To match GPT-5.2 accuracy ({gpt52_acc:.1%}):")
        print(f"    Routes {cheap_pct:.0f}% of queries to GPT-5-mini (cheap)")
        print(f"    Normalized cost: {break_even['cost_normalized']:.2f}x "
              f"(vs {cost_ratio:.2f}x for all GPT-5.2)")
        print(f"    Cost savings: {savings_pct:.0f}%")
        output_summary["break_even"] = {
            "threshold": break_even["threshold"],
            "accuracy": break_even["accuracy"],
            "cost_normalized": break_even["cost_normalized"],
            "pct_cheap": float(cheap_pct),
            "pct_savings": float(savings_pct),
        }
    else:
        print(f"  Calibrator routing never reaches GPT-5.2 accuracy ({gpt52_acc:.1%})")

    # Plot
    plot_routing(results_by_method, cost_ratio, mini_acc, gpt52_acc,
                 f"{args.fig_dir}/uc2_model_routing.pdf")

    # Save
    out_path = f"{args.output_dir}/uc2_results.json"
    with open(out_path, "w") as f:
        json.dump(output_summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
