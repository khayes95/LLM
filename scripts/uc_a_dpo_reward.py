#!/usr/bin/env python3
"""UC-A: UQ as Reward Signal for DPO — Stage 1 (Dataset Construction + Oracle Analysis).

Uses calibrator P(correct) scores to construct DPO preference pairs from
cross-model responses.  For each question answered by 2+ models, creates
(chosen, rejected) pairs where chosen = response with higher p_correct.
Analyzes whether the calibrator provides a useful reward signal compared to
verbalized confidence, response length, random selection, and an oracle
that uses ground-truth correctness.

Analyses:
1. Build cross-model DPO preference pairs
2. Pair quality metrics (pairwise accuracy, strong pairwise accuracy)
3. Margin analysis — accuracy by |delta_p_correct| quartile
4. Comparison baselines (calibrator, verbalized, length, random, oracle)
5. Per-benchmark breakdown of pair quality
6. DPO dataset statistics (pair types, easy/informative fractions)

Outputs:
    {output_dir}/uc_a_results.json — all metrics and tables
    {output_dir}/uc_a_dpo_pairs.jsonl — constructed DPO preference pairs
    {fig_dir}/uc_a_pair_quality.pdf — bar chart comparing selection methods
    {fig_dir}/uc_a_margin_analysis.pdf — accuracy by margin quartile

Usage:
    python scripts/uc_a_dpo_reward.py
    python scripts/uc_a_dpo_reward.py --scored_dir data/use_cases/scored_unified
    python scripts/uc_a_dpo_reward.py --smoke_test
"""
import argparse
import json
from collections import defaultdict
from itertools import combinations
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
# Pair construction
# ---------------------------------------------------------------------------

def build_question_index(all_scored):
    """Index samples by (benchmark, id) -> {model: sample}."""
    question_data = defaultdict(dict)
    for target, samples in all_scored.items():
        for s in samples:
            key = (s["benchmark"], s["id"])
            question_data[key][target] = s
    return question_data


def build_dpo_pairs(question_data):
    """For each question with 2+ models, create all pairwise DPO preference
    pairs.  Chosen = response with higher p_correct.

    Returns list of pair dicts.
    """
    pairs = []
    for (bench, qid), models in question_data.items():
        if len(models) < 2:
            continue
        model_names = sorted(models.keys())
        for m_a, m_b in combinations(model_names, 2):
            a = models[m_a]
            b = models[m_b]
            # Chosen = higher p_correct
            if a["p_correct"] >= b["p_correct"]:
                chosen, rejected = a, b
                chosen_model, rejected_model = m_a, m_b
            else:
                chosen, rejected = b, a
                chosen_model, rejected_model = m_b, m_a

            pairs.append({
                "question_id": qid,
                "benchmark": bench,
                "chosen_model": chosen_model,
                "rejected_model": rejected_model,
                "chosen_p_correct": chosen["p_correct"],
                "rejected_p_correct": rejected["p_correct"],
                "chosen_correct": chosen["is_correct"],
                "rejected_correct": rejected["is_correct"],
                "chosen_verbalized": chosen["verbalized_confidence"],
                "rejected_verbalized": rejected["verbalized_confidence"],
                "chosen_output_tokens": chosen["output_tokens"],
                "rejected_output_tokens": rejected["output_tokens"],
                "chosen_response_preview": chosen.get("response_preview", ""),
                "rejected_response_preview": rejected.get("response_preview", ""),
                "delta_p_correct": abs(chosen["p_correct"] - rejected["p_correct"]),
            })
    return pairs


# ---------------------------------------------------------------------------
# Pair quality evaluation
# ---------------------------------------------------------------------------

def evaluate_pairs_by_method(pairs):
    """Evaluate pair quality under different selection signals.

    For each method, we re-select chosen/rejected using that signal and
    check whether the selection is concordant with ground truth.

    Returns dict of {method: metrics_dict}.
    """
    results = {}

    methods = {
        "calibrator": lambda p: (p["chosen_p_correct"], p["rejected_p_correct"]),
        "verbalized": lambda p: (p["chosen_verbalized"], p["rejected_verbalized"]),
        "response_length": lambda p: (p["chosen_output_tokens"], p["rejected_output_tokens"]),
        "random": None,  # handled separately
        "oracle": None,  # handled separately
    }

    rng = np.random.RandomState(42)

    for method_name, score_fn in methods.items():
        n_total = 0
        n_pairwise_correct = 0   # chosen_correct >= rejected_correct
        n_strong_correct = 0     # chosen_correct=1 AND rejected_correct=0
        n_informative = 0        # exactly one correct in the pair
        n_info_correct = 0       # among informative, did we pick the correct one?

        for p in pairs:
            chosen_correct = p["chosen_correct"]
            rejected_correct = p["rejected_correct"]

            # Determine which response this method selects as "chosen"
            if method_name == "random":
                # Coin flip — swap with 50% probability
                if rng.random() < 0.5:
                    sel_chosen_correct = chosen_correct
                    sel_rejected_correct = rejected_correct
                else:
                    sel_chosen_correct = rejected_correct
                    sel_rejected_correct = chosen_correct
            elif method_name == "oracle":
                # Oracle always picks the correct one (or chosen if tied)
                if chosen_correct >= rejected_correct:
                    sel_chosen_correct = chosen_correct
                    sel_rejected_correct = rejected_correct
                else:
                    sel_chosen_correct = rejected_correct
                    sel_rejected_correct = chosen_correct
            else:
                scores = score_fn(p)
                if scores[0] is None or scores[1] is None:
                    continue  # skip if signal unavailable
                # The pair is already ordered by calibrator; re-order by this method
                if scores[0] >= scores[1]:
                    # Method agrees with calibrator ordering
                    sel_chosen_correct = chosen_correct
                    sel_rejected_correct = rejected_correct
                else:
                    # Method disagrees — swap
                    sel_chosen_correct = rejected_correct
                    sel_rejected_correct = chosen_correct

            n_total += 1

            # Pairwise accuracy: chosen at least as correct as rejected
            if sel_chosen_correct >= sel_rejected_correct:
                n_pairwise_correct += 1

            # Strong accuracy: chosen=1, rejected=0
            if sel_chosen_correct == 1 and sel_rejected_correct == 0:
                n_strong_correct += 1

            # Informative pair analysis (exactly one correct)
            if chosen_correct != rejected_correct:
                n_informative += 1
                if sel_chosen_correct > sel_rejected_correct:
                    n_info_correct += 1

        results[method_name] = {
            "n_total": n_total,
            "pairwise_accuracy": n_pairwise_correct / n_total if n_total > 0 else 0.0,
            "strong_accuracy": n_strong_correct / n_total if n_total > 0 else 0.0,
            "n_informative": n_informative,
            "informative_accuracy": n_info_correct / n_informative if n_informative > 0 else 0.0,
        }

    return results


def margin_analysis(pairs, n_quartiles=4):
    """Analyze pairwise accuracy by delta_p_correct quartile.

    Higher margins should yield more reliable pair ordering.
    """
    if not pairs:
        return []

    deltas = np.array([p["delta_p_correct"] for p in pairs])
    quartile_edges = np.quantile(deltas, np.linspace(0, 1, n_quartiles + 1))

    results = []
    for q in range(n_quartiles):
        lo = quartile_edges[q]
        hi = quartile_edges[q + 1]

        # Include right edge for the last quartile
        if q == n_quartiles - 1:
            subset = [p for p in pairs if lo <= p["delta_p_correct"] <= hi]
        else:
            subset = [p for p in pairs if lo <= p["delta_p_correct"] < hi]

        if not subset:
            results.append({
                "quartile": q + 1,
                "delta_lo": float(lo),
                "delta_hi": float(hi),
                "n_pairs": 0,
                "pairwise_accuracy": None,
                "strong_accuracy": None,
                "informative_accuracy": None,
            })
            continue

        n = len(subset)
        n_pw = sum(1 for p in subset if p["chosen_correct"] >= p["rejected_correct"])
        n_strong = sum(1 for p in subset
                       if p["chosen_correct"] == 1 and p["rejected_correct"] == 0)
        info = [p for p in subset if p["chosen_correct"] != p["rejected_correct"]]
        n_info_correct = sum(1 for p in info if p["chosen_correct"] > p["rejected_correct"])

        results.append({
            "quartile": q + 1,
            "delta_lo": float(lo),
            "delta_hi": float(hi),
            "n_pairs": n,
            "pairwise_accuracy": n_pw / n,
            "strong_accuracy": n_strong / n,
            "n_informative": len(info),
            "informative_accuracy": n_info_correct / len(info) if info else None,
        })

    return results


def per_benchmark_pair_quality(pairs):
    """Compute pair quality metrics per benchmark."""
    by_bench = defaultdict(list)
    for p in pairs:
        by_bench[p["benchmark"]].append(p)

    results = {}
    for bench in sorted(by_bench.keys()):
        bp = by_bench[bench]
        n = len(bp)
        n_pw = sum(1 for p in bp if p["chosen_correct"] >= p["rejected_correct"])
        n_strong = sum(1 for p in bp
                       if p["chosen_correct"] == 1 and p["rejected_correct"] == 0)
        info = [p for p in bp if p["chosen_correct"] != p["rejected_correct"]]
        n_info_correct = sum(1 for p in info if p["chosen_correct"] > p["rejected_correct"])

        results[bench] = {
            "n_pairs": n,
            "pairwise_accuracy": n_pw / n if n > 0 else 0.0,
            "strong_accuracy": n_strong / n if n > 0 else 0.0,
            "n_informative": len(info),
            "informative_accuracy": n_info_correct / len(info) if info else None,
        }
    return results


# ---------------------------------------------------------------------------
# DPO dataset statistics
# ---------------------------------------------------------------------------

def compute_dataset_stats(pairs):
    """Compute statistics about the constructed DPO dataset."""
    n_total = len(pairs)
    if n_total == 0:
        return {}

    # Unique questions
    unique_questions = set((p["question_id"], p["benchmark"]) for p in pairs)

    # Model pair distribution
    model_pair_counts = defaultdict(int)
    for p in pairs:
        pair_key = f"{p['chosen_model']}-{p['rejected_model']}"
        # Normalize order for counting
        pair_key_sorted = "-".join(sorted([p["chosen_model"], p["rejected_model"]]))
        model_pair_counts[pair_key_sorted] += 1

    # Easy vs informative
    n_both_correct = sum(1 for p in pairs
                         if p["chosen_correct"] == 1 and p["rejected_correct"] == 1)
    n_both_wrong = sum(1 for p in pairs
                       if p["chosen_correct"] == 0 and p["rejected_correct"] == 0)
    n_informative = sum(1 for p in pairs
                        if p["chosen_correct"] != p["rejected_correct"])

    # Benchmark distribution
    bench_counts = defaultdict(int)
    for p in pairs:
        bench_counts[p["benchmark"]] += 1

    return {
        "n_total_pairs": n_total,
        "n_unique_questions": len(unique_questions),
        "model_pair_counts": dict(model_pair_counts),
        "n_both_correct": n_both_correct,
        "n_both_wrong": n_both_wrong,
        "n_informative": n_informative,
        "frac_both_correct": n_both_correct / n_total,
        "frac_both_wrong": n_both_wrong / n_total,
        "frac_informative": n_informative / n_total,
        "frac_easy": (n_both_correct + n_both_wrong) / n_total,
        "benchmark_distribution": dict(bench_counts),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pair_quality(method_results, output_path):
    """Bar chart comparing selection methods on pairwise, strong, and
    informative accuracy."""
    methods_order = ["calibrator", "verbalized", "response_length", "random", "oracle"]
    method_labels = {
        "calibrator": "Calibrator\n(p_correct)",
        "verbalized": "Verbalized\nConfidence",
        "response_length": "Response\nLength",
        "random": "Random",
        "oracle": "Oracle\n(ground truth)",
    }
    colors = {
        "calibrator": "C0",
        "verbalized": "C1",
        "response_length": "C2",
        "random": "gray",
        "oracle": "C4",
    }

    present = [m for m in methods_order if m in method_results]
    n_methods = len(present)

    fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 5))
    metrics = [
        ("pairwise_accuracy", "Pairwise Accuracy\n(chosen_correct >= rejected_correct)"),
        ("strong_accuracy", "Strong Accuracy\n(chosen=1, rejected=0)"),
        ("informative_accuracy", "Informative Pair Accuracy\n(among pairs with exactly 1 correct)"),
    ]

    for ax, (metric_key, metric_title) in zip(axes, metrics):
        vals = []
        labels = []
        bar_colors = []
        for m in present:
            v = method_results[m].get(metric_key)
            if v is None:
                v = 0.0
            vals.append(v)
            labels.append(method_labels.get(m, m))
            bar_colors.append(colors.get(m, "C0"))

        x = np.arange(n_methods)
        bars = ax.bar(x, vals, color=bar_colors, edgecolor="black", linewidth=0.5)

        # Add value labels on bars
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(metric_title, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("UC-A: DPO Pair Quality by Selection Method", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {output_path}")


def plot_margin_analysis(margin_data, output_path):
    """Accuracy by delta_p_correct quartile."""
    quartiles = [d for d in margin_data if d["n_pairs"] > 0]
    if not quartiles:
        print("  No margin data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: pairwise + strong accuracy by quartile
    ax = axes[0]
    x = np.arange(len(quartiles))
    labels = [f"Q{d['quartile']}\n[{d['delta_lo']:.2f}, {d['delta_hi']:.2f}]" for d in quartiles]
    w = 0.35

    pw_accs = [d["pairwise_accuracy"] for d in quartiles]
    strong_accs = [d["strong_accuracy"] for d in quartiles]

    bars1 = ax.bar(x - w / 2, pw_accs, w, label="Pairwise Accuracy", color="C0", alpha=0.85)
    bars2 = ax.bar(x + w / 2, strong_accs, w, label="Strong Accuracy", color="C3", alpha=0.85)

    for xi, v in zip(x, pw_accs):
        ax.text(xi - w / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x, strong_accs):
        ax.text(xi + w / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("|delta_p_correct| quartile", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Pair Quality by Calibrator Margin", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: informative accuracy + pair count
    ax = axes[1]
    info_accs = [d.get("informative_accuracy") or 0.0 for d in quartiles]
    n_pairs = [d["n_pairs"] for d in quartiles]

    ax_twin = ax.twinx()
    bars = ax.bar(x, info_accs, 0.6, color="C0", alpha=0.85, label="Informative Accuracy")
    for xi, v in zip(x, info_accs):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax_twin.plot(x, n_pairs, "D-", color="C1", markersize=7, linewidth=2, label="# Pairs")
    for xi, n in zip(x, n_pairs):
        ax_twin.text(xi + 0.1, n + max(n_pairs) * 0.02, str(n),
                     ha="center", fontsize=8, color="C1")

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("|delta_p_correct| quartile", fontsize=11)
    ax.set_ylabel("Informative Pair Accuracy", fontsize=11)
    ax_twin.set_ylabel("Number of Pairs", fontsize=11, color="C1")
    ax.set_title("Informative Pairs: Accuracy & Count by Margin", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    fig.suptitle("UC-A: Margin Analysis — Larger Calibrator Margins = More Reliable Pairs",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UC-A: UQ as Reward Signal for DPO (Stage 1 — Dataset Construction + Oracle Analysis)")
    parser.add_argument("--scored_dir", default="data/use_cases/scored_unified",
                        help="Directory with scored JSONL files")
    parser.add_argument("--output_dir", default="data/use_cases/results_unified",
                        help="Directory for results JSON and DPO pairs")
    parser.add_argument("--fig_dir", default="figures/use_cases_unified",
                        help="Directory for output figures")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Only use first 50 samples per model")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    targets = ["gpt5mini", "gpt52", "qwen35"]
    target_names = {
        "gpt5mini": "GPT-5-mini",
        "gpt52": "GPT-5.2",
        "qwen35": "Qwen3.5",
    }
    all_scored = {}
    for target in targets:
        path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if not path.exists():
            print(f"  Skipping {target}: {path} not found")
            continue
        samples = load_scored(path)
        if args.smoke_test:
            samples = samples[:50]
        all_scored[target] = samples
        print(f"  {target}: {len(samples)} samples loaded")

    if len(all_scored) < 2:
        print("ERROR: Need at least 2 scored targets for cross-model pairs.")
        return

    print()
    print("=" * 70)
    print("UC-A: UQ as Reward Signal for DPO")
    print("  Stage 1 — Dataset Construction + Oracle Analysis (CPU-only)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Build cross-model DPO pairs
    # -----------------------------------------------------------------------
    question_data = build_question_index(all_scored)
    n_multi = sum(1 for v in question_data.values() if len(v) >= 2)
    n_triple = sum(1 for v in question_data.values() if len(v) >= 3)
    print(f"\nQuestions total: {len(question_data)}")
    print(f"Questions with 2+ models: {n_multi}")
    print(f"Questions with 3 models (triple): {n_triple}")

    pairs = build_dpo_pairs(question_data)
    print(f"DPO preference pairs constructed: {len(pairs)}")

    if not pairs:
        print("ERROR: No pairs could be constructed. Check that questions overlap across models.")
        return

    # -----------------------------------------------------------------------
    # 2. DPO dataset statistics
    # -----------------------------------------------------------------------
    ds_stats = compute_dataset_stats(pairs)

    print(f"\n--- DPO Dataset Statistics ---")
    print(f"  Total pairs: {ds_stats['n_total_pairs']}")
    print(f"  Unique questions: {ds_stats['n_unique_questions']}")
    print(f"\n  Model pair distribution:")
    for pair_type, count in sorted(ds_stats["model_pair_counts"].items()):
        print(f"    {pair_type}: {count} ({count / ds_stats['n_total_pairs']:.1%})")
    print(f"\n  Pair composition:")
    print(f"    Both correct (easy):       {ds_stats['n_both_correct']:>5} ({ds_stats['frac_both_correct']:.1%})")
    print(f"    Both wrong (easy):         {ds_stats['n_both_wrong']:>5} ({ds_stats['frac_both_wrong']:.1%})")
    print(f"    Exactly one correct (informative): {ds_stats['n_informative']:>5} ({ds_stats['frac_informative']:.1%})")
    print(f"    Easy (uninformative for DPO):       {ds_stats['n_both_correct'] + ds_stats['n_both_wrong']:>5} ({ds_stats['frac_easy']:.1%})")

    # -----------------------------------------------------------------------
    # 3. Pair quality by selection method
    # -----------------------------------------------------------------------
    method_results = evaluate_pairs_by_method(pairs)

    print(f"\n--- Pair Quality by Selection Method ---")
    print(f"  {'Method':<22} {'Pairwise':>10} {'Strong':>10} {'Info Acc':>10} {'N pairs':>8} {'N info':>8}")
    print(f"  {'-'*72}")

    method_display_order = ["calibrator", "verbalized", "response_length", "random", "oracle"]
    method_labels = {
        "calibrator": "Calibrator (p_correct)",
        "verbalized": "Verbalized Conf.",
        "response_length": "Response Length",
        "random": "Random (coin flip)",
        "oracle": "Oracle (ground truth)",
    }
    for method in method_display_order:
        if method not in method_results:
            continue
        r = method_results[method]
        info_str = f"{r['informative_accuracy']:.3f}" if r["informative_accuracy"] else "N/A"
        print(f"  {method_labels.get(method, method):<22} "
              f"{r['pairwise_accuracy']:>10.3f} "
              f"{r['strong_accuracy']:>10.3f} "
              f"{info_str:>10} "
              f"{r['n_total']:>8} "
              f"{r['n_informative']:>8}")

    # -----------------------------------------------------------------------
    # 4. Margin analysis
    # -----------------------------------------------------------------------
    margin_data = margin_analysis(pairs, n_quartiles=4)

    print(f"\n--- Margin Analysis (by |delta_p_correct| quartile) ---")
    print(f"  {'Quartile':>8} {'Range':>18} {'N pairs':>8} {'Pairwise':>10} {'Strong':>10} {'Info Acc':>10}")
    print(f"  {'-'*68}")
    for d in margin_data:
        if d["n_pairs"] == 0:
            continue
        pw_str = f"{d['pairwise_accuracy']:.3f}" if d["pairwise_accuracy"] is not None else "N/A"
        st_str = f"{d['strong_accuracy']:.3f}" if d["strong_accuracy"] is not None else "N/A"
        ia_str = f"{d['informative_accuracy']:.3f}" if d.get("informative_accuracy") is not None else "N/A"
        print(f"  Q{d['quartile']:>6} [{d['delta_lo']:.3f}, {d['delta_hi']:.3f}] "
              f"{d['n_pairs']:>8} {pw_str:>10} {st_str:>10} {ia_str:>10}")

    # -----------------------------------------------------------------------
    # 5. Per-benchmark breakdown
    # -----------------------------------------------------------------------
    bench_quality = per_benchmark_pair_quality(pairs)

    print(f"\n--- Per-Benchmark Pair Quality (calibrator selection) ---")
    print(f"  {'Benchmark':<22} {'N pairs':>8} {'Pairwise':>10} {'Strong':>10} {'N info':>8} {'Info Acc':>10}")
    print(f"  {'-'*72}")
    for bench, bq in sorted(bench_quality.items(), key=lambda x: -x[1]["n_pairs"]):
        ia_str = f"{bq['informative_accuracy']:.3f}" if bq["informative_accuracy"] is not None else "N/A"
        print(f"  {bench:<22} {bq['n_pairs']:>8} {bq['pairwise_accuracy']:>10.3f} "
              f"{bq['strong_accuracy']:>10.3f} {bq['n_informative']:>8} {ia_str:>10}")

    # -----------------------------------------------------------------------
    # 6. Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("UC-A Summary: Calibrator as DPO Reward Signal")
    print(f"{'='*70}")

    cal = method_results.get("calibrator", {})
    rand = method_results.get("random", {})
    oracle = method_results.get("oracle", {})

    pw_lift = cal.get("pairwise_accuracy", 0) - rand.get("pairwise_accuracy", 0.5)
    info_lift = (cal.get("informative_accuracy", 0) or 0) - (rand.get("informative_accuracy", 0.5) or 0.5)

    print(f"  Calibrator pairwise accuracy:     {cal.get('pairwise_accuracy', 0):.3f}")
    print(f"  Random baseline pairwise:         {rand.get('pairwise_accuracy', 0):.3f}")
    print(f"  Lift over random (pairwise):      {pw_lift:+.3f}")
    print(f"  Oracle ceiling (pairwise):        {oracle.get('pairwise_accuracy', 0):.3f}")
    print(f"  Calibrator informative accuracy:  {cal.get('informative_accuracy', 0):.3f}")
    print(f"  Lift over random (informative):   {info_lift:+.3f}")
    print(f"  Informative pair fraction:         {ds_stats.get('frac_informative', 0):.1%}")
    print(f"  Total DPO pairs:                   {ds_stats.get('n_total_pairs', 0)}")

    # -----------------------------------------------------------------------
    # Save DPO pairs JSONL
    # -----------------------------------------------------------------------
    pairs_path = Path(args.output_dir) / "uc_a_dpo_pairs.jsonl"
    with open(pairs_path, "w") as f:
        for p in pairs:
            row = {
                "question_id": p["question_id"],
                "benchmark": p["benchmark"],
                "chosen_model": p["chosen_model"],
                "rejected_model": p["rejected_model"],
                "chosen_p_correct": p["chosen_p_correct"],
                "rejected_p_correct": p["rejected_p_correct"],
                "chosen_correct": p["chosen_correct"],
                "rejected_correct": p["rejected_correct"],
                "chosen_response_preview": p["chosen_response_preview"],
                "rejected_response_preview": p["rejected_response_preview"],
            }
            f.write(json.dumps(row) + "\n")
    print(f"\n  DPO pairs saved: {pairs_path} ({len(pairs)} pairs)")

    # -----------------------------------------------------------------------
    # Save results JSON
    # -----------------------------------------------------------------------
    results = {
        "dataset_stats": ds_stats,
        "method_comparison": {k: v for k, v in method_results.items()},
        "margin_analysis": margin_data,
        "per_benchmark": bench_quality,
        "n_multi_model_questions": n_multi,
        "n_triple_model_questions": n_triple,
        "summary": {
            "calibrator_pairwise_accuracy": cal.get("pairwise_accuracy"),
            "calibrator_informative_accuracy": cal.get("informative_accuracy"),
            "random_pairwise_accuracy": rand.get("pairwise_accuracy"),
            "oracle_pairwise_accuracy": oracle.get("pairwise_accuracy"),
            "pairwise_lift_over_random": pw_lift,
            "informative_lift_over_random": info_lift,
        },
    }

    results_path = Path(args.output_dir) / "uc_a_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {results_path}")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    plot_pair_quality(method_results,
                      f"{args.fig_dir}/uc_a_pair_quality.pdf")
    plot_margin_analysis(margin_data,
                         f"{args.fig_dir}/uc_a_margin_analysis.pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()
