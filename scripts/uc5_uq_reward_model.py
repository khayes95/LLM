#!/usr/bin/env python3
"""UC5: UQ as Reward Model — cross-model response selection.

Given multiple models' responses to the same question, can P(correct)
correctly identify the best response?

Key approach: Use the UNIFIED calibrator (scored_test_only_v2) so that P(correct)
comes from the same model across all targets. This is the honest cross-model
evaluation on test-only data (no data leakage).

Usage:
    # Unified calibrator, test-only (recommended)
    python scripts/uc5_uq_reward_model.py --scored_dir data/use_cases/scored_test_only_v2
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest


def load_scored(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def compute_normalizations(all_scored):
    """Compute per-model normalization statistics."""
    stats = {}
    for target, samples in all_scored.items():
        scores = [s["p_correct"] for s in samples]
        stats[target] = {
            "mean": np.mean(scores),
            "std": np.std(scores) + 1e-8,
            "scores_sorted": np.sort(scores),
            "n": len(scores),
        }
    return stats


def apply_normalizations(all_scored, stats):
    """Add normalized score fields to each sample."""
    for target, samples in all_scored.items():
        mu = stats[target]["mean"]
        sigma = stats[target]["std"]
        sorted_scores = stats[target]["scores_sorted"]
        n = stats[target]["n"]

        # Per-benchmark stats
        bench_stats = defaultdict(lambda: {"scores": []})
        for s in samples:
            bench_stats[s["benchmark"]]["scores"].append(s["p_correct"])
        for bench in bench_stats:
            scores = bench_stats[bench]["scores"]
            bench_stats[bench]["mean"] = np.mean(scores)
            bench_stats[bench]["std"] = np.std(scores) + 1e-8

        for s in samples:
            # Z-score (global per model)
            s["z_score_global"] = (s["p_correct"] - mu) / sigma

            # Percentile (rank within model)
            s["percentile"] = np.searchsorted(sorted_scores, s["p_correct"]) / n

            # Z-score (per benchmark, min N=30)
            bench = s["benchmark"]
            if len(bench_stats[bench]["scores"]) >= 30:
                b_mu = bench_stats[bench]["mean"]
                b_std = bench_stats[bench]["std"]
                s["z_score_bench"] = (s["p_correct"] - b_mu) / b_std
            else:
                s["z_score_bench"] = s["z_score_global"]  # fallback

            # Response length (output tokens as a proxy)
            s["response_length"] = s.get("output_tokens") or 0


def build_multi_model_data(all_scored):
    """Build per-question data across models."""
    question_data = defaultdict(dict)
    for target, samples in all_scored.items():
        for s in samples:
            key = (s["benchmark"], s["id"])
            question_data[key][target] = s
    return question_data


def pairwise_ranking_accuracy(question_data, model_a, model_b, score_key):
    """For discriminative pairs (one correct, one wrong),
    does the higher score pick the correct response?"""
    n_disc = 0
    n_correct = 0

    for key, models in question_data.items():
        if model_a not in models or model_b not in models:
            continue

        a = models[model_a]
        b = models[model_b]

        if a["is_correct"] == b["is_correct"]:
            continue

        n_disc += 1
        sa = a.get(score_key)
        sb = b.get(score_key)
        if sa is None or sb is None:
            continue

        if a["is_correct"] > b["is_correct"]:
            if sa > sb:
                n_correct += 1
        else:
            if sb > sa:
                n_correct += 1

    acc = n_correct / n_disc if n_disc > 0 else 0.5
    # Binomial test vs 50% (chance)
    p_value = None
    if n_disc >= 10:
        result = binomtest(n_correct, n_disc, 0.5, alternative="greater")
        p_value = result.pvalue

    return {
        "pairwise_accuracy": float(acc),
        "n_discriminative": n_disc,
        "n_correct": n_correct,
        "p_value": float(p_value) if p_value is not None else None,
        "significant": p_value is not None and p_value < 0.05,
    }


def best_of_n_selection(question_data, models, score_key):
    """Select response with highest score. Is it correct more often than random?"""
    n_questions = 0
    n_selected_correct = 0
    n_random_correct = 0

    for key, model_scores in question_data.items():
        available = {m: model_scores[m] for m in models if m in model_scores}
        if len(available) < 2:
            continue

        # Skip if score_key is None/missing for any
        if any(available[m].get(score_key) is None for m in available):
            continue

        n_questions += 1
        best_model = max(available, key=lambda m: available[m].get(score_key, 0.5))
        n_selected_correct += available[best_model]["is_correct"]
        n_correct = sum(v["is_correct"] for v in available.values())
        n_random_correct += n_correct / len(available)

    return {
        "n_questions": n_questions,
        "selected_accuracy": float(n_selected_correct / n_questions) if n_questions else 0,
        "random_accuracy": float(n_random_correct / n_questions) if n_questions else 0,
    }


def plot_reward_model(all_pairwise, bon_results, output_path):
    """Plot comprehensive reward model comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Pairwise ranking by method for each model pair
    ax = axes[0]
    pair_names = sorted(set(p for results in all_pairwise.values() for p in results.keys()))
    methods = ["p_correct", "z_score_global", "percentile", "verbalized"]
    method_labels = ["Calibrator (raw)", "Z-score", "Percentile", "Verbalized"]
    colors = ["C0", "C1", "C2", "C3"]

    x = np.arange(len(pair_names))
    w = 0.2
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        accs = []
        for pair in pair_names:
            res = all_pairwise.get(method, {}).get(pair, {})
            accs.append(res.get("pairwise_accuracy", 0.5))
        offset = (i - 1.5) * w
        bars = ax.bar(x + offset, accs, w, label=label, color=color)
        # Mark significant results
        for j, pair in enumerate(pair_names):
            res = all_pairwise.get(method, {}).get(pair, {})
            if res.get("significant"):
                ax.text(x[j] + offset, accs[j] + 0.01, "*", ha="center", fontsize=12, fontweight="bold")

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_vs_", "\nvs ") for p in pair_names], fontsize=9)
    ax.set_ylabel("Pairwise Ranking Accuracy", fontsize=12)
    ax.set_title("Which Response is Correct?\n(* = p < 0.05 vs chance)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.35, 0.75)

    # Right: Best-of-N
    ax = axes[1]
    bon_methods = ["p_correct", "z_score_global", "percentile", "verbalized", "response_length"]
    bon_labels = ["Calibrator\n(raw)", "Z-score", "Percentile", "Verbalized", "Resp.\nLength"]
    sel_accs = [bon_results.get(m, {}).get("selected_accuracy", 0) for m in bon_methods]
    rand_acc = bon_results.get("p_correct", {}).get("random_accuracy", 0.5)

    bars = ax.bar(range(len(bon_methods)), sel_accs, color=colors + ["C4"])
    ax.axhline(y=rand_acc, color="gray", linestyle=":", alpha=0.7, label=f"Random ({rand_acc:.3f})")
    ax.set_xticks(range(len(bon_methods)))
    ax.set_xticklabels(bon_labels, fontsize=9)
    ax.set_ylabel("Accuracy of Selected Response", fontsize=12)
    ax.set_title("Best-of-N Response Selection", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2")
    parser.add_argument("--output_dir", default="data/use_cases/results_test_only_v2")
    parser.add_argument("--fig_dir", default="figures/use_cases")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    all_scored = {}
    for target in ["gpt5mini", "gpt52", "qwen35"]:
        path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if path.exists():
            all_scored[target] = load_scored(path)
            print(f"  {target}: {len(all_scored[target])} samples")

    if len(all_scored) < 2:
        print("ERROR: Need at least 2 scored targets")
        return

    print("=" * 70)
    print("UC5: UQ as Reward Model / Response Selection")
    print("  (model-specific calibrators + normalization strategies)")
    print("=" * 70)

    # Compute and apply normalizations
    stats = compute_normalizations(all_scored)
    apply_normalizations(all_scored, stats)

    for target, s in stats.items():
        print(f"  {target}: mean_p={s['mean']:.3f}, std_p={s['std']:.3f}")

    question_data = build_multi_model_data(all_scored)
    models = list(all_scored.keys())

    n_multi = sum(1 for k, v in question_data.items() if len(v) >= 2)
    n_triple = sum(1 for k, v in question_data.items() if len(v) >= 3)
    print(f"\nQuestions with 2+ models: {n_multi}")
    print(f"Questions with 3 models: {n_triple}")

    # Score methods to evaluate
    score_methods = {
        "p_correct": "Calibrator (raw P)",
        "z_score_global": "Z-score (global)",
        "z_score_bench": "Z-score (per-bench)",
        "percentile": "Percentile",
        "verbalized": "Verbalized Conf.",
        "response_length": "Response Length",
    }

    # Pairwise ranking
    print(f"\n--- Pairwise Ranking Accuracy ---")
    print(f"{'':2}{'Method':<22} ", end="")
    model_pairs = []
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            pair = f"{m1}_vs_{m2}"
            model_pairs.append((m1, m2, pair))
            print(f"  {pair:>20}", end="")
    print()
    print("  " + "-" * (22 + 22 * len(model_pairs)))

    all_pairwise = {}
    for method_key, method_label in score_methods.items():
        all_pairwise[method_key] = {}
        print(f"  {method_label:<22}", end="")
        for m1, m2, pair in model_pairs:
            # Use "verbalized_confidence" field for verbalized
            sk = "verbalized_confidence" if method_key == "verbalized" else method_key
            res = pairwise_ranking_accuracy(question_data, m1, m2, sk)
            all_pairwise[method_key][pair] = res
            sig = "*" if res.get("significant") else " "
            print(f"  {res['pairwise_accuracy']:>6.3f}{sig} (n={res['n_discriminative']:>4})", end="")
        print()

    # Best-of-N
    print(f"\n--- Best-of-N Response Selection ---")
    bon_results = {}
    for method_key, method_label in score_methods.items():
        sk = "verbalized_confidence" if method_key == "verbalized" else method_key
        bon = best_of_n_selection(question_data, models, sk)
        bon_results[method_key] = bon
        delta = bon["selected_accuracy"] - bon["random_accuracy"]
        print(f"  {method_label:<22}: {bon['selected_accuracy']:.3f} "
              f"(random: {bon['random_accuracy']:.3f}, delta: {delta:+.3f}, N={bon['n_questions']})")

    # Plot
    plot_reward_model(all_pairwise, bon_results,
                      f"{args.fig_dir}/uc5_reward_model.pdf")

    # Save
    out_path = f"{args.output_dir}/uc5_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "pairwise_ranking": all_pairwise,
            "best_of_n": bon_results,
            "normalization_stats": {t: {"mean": s["mean"], "std": s["std"], "n": s["n"]}
                                    for t, s in stats.items()},
            "n_multi_model_questions": n_multi,
            "n_triple_model_questions": n_triple,
        }, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
