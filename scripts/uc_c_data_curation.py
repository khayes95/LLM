#!/usr/bin/env python3
"""UC-C: Synthetic Data Curation / Distillation — Stage 1 (Quality-Quantity Tradeoff).

For each teacher model (gpt5mini, gpt52, qwen35), filter scored predictions at
various calibrator p_correct thresholds. Measures the tradeoff between data
retention and data quality, comparing the calibrator against verbalized
confidence, random subsampling, and response length baselines.

Analyses:
1. Quality-Quantity Tradeoff — accuracy vs retention at each threshold
2. Precision-Recall for Correct Filtering — treating p_correct > t as "correct"
3. Per-Benchmark Breakdown — accuracy gain per benchmark at p > 0.7
4. Summary Metrics — accuracy at 50% retention, threshold for 85%/90% accuracy,
   area under the accuracy-vs-retention curve (AUAR)

Outputs:
    {output_dir}/uc_c_results.json — all tables, thresholds, per-benchmark breakdowns
    {fig_dir}/uc_c_quality_quantity.pdf — main quality vs quantity plot
    {fig_dir}/uc_c_per_benchmark.pdf — per-benchmark bar chart

Usage:
    python scripts/uc_c_data_curation.py
    python scripts/uc_c_data_curation.py --scored_dir data/use_cases/scored_unified
    python scripts/uc_c_data_curation.py --smoke_test
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
            # Ensure numeric types (handle both string and native)
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
# Quality-Quantity Tradeoff
# ---------------------------------------------------------------------------

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def quality_quantity_calibrator(samples, thresholds=THRESHOLDS):
    """Filter by p_correct > t for each threshold."""
    n_total = len(samples)
    results = []
    for t in thresholds:
        retained = [s for s in samples if s["p_correct"] > t]
        n_ret = len(retained)
        if n_ret == 0:
            results.append({
                "threshold": t, "n_retained": 0, "pct_retained": 0.0,
                "accuracy": None, "benchmark_dist": {},
            })
            continue
        acc = np.mean([s["is_correct"] for s in retained])
        bench_dist = defaultdict(int)
        for s in retained:
            bench_dist[s["benchmark"]] += 1
        results.append({
            "threshold": t,
            "n_retained": n_ret,
            "pct_retained": n_ret / n_total,
            "accuracy": float(acc),
            "benchmark_dist": dict(bench_dist),
        })
    return results


def quality_quantity_verbalized(samples, thresholds=THRESHOLDS):
    """Filter by verbalized_confidence > t (skip samples with None)."""
    valid = [s for s in samples if s["verbalized_confidence"] is not None]
    n_total = len(valid)
    if n_total == 0:
        return []
    results = []
    for t in thresholds:
        retained = [s for s in valid if s["verbalized_confidence"] > t]
        n_ret = len(retained)
        if n_ret == 0:
            results.append({
                "threshold": t, "n_retained": 0,
                "pct_retained": 0.0, "accuracy": None,
            })
            continue
        acc = np.mean([s["is_correct"] for s in retained])
        results.append({
            "threshold": t,
            "n_retained": n_ret,
            "pct_retained": n_ret / n_total,
            "accuracy": float(acc),
        })
    return results


def quality_quantity_random(samples, thresholds=THRESHOLDS, n_trials=100, seed=42):
    """Random subsampling baseline: for each threshold, match the calibrator's
    retention count and average accuracy over n_trials random draws."""
    rng = np.random.RandomState(seed)
    n_total = len(samples)
    labels = np.array([s["is_correct"] for s in samples])

    # Pre-compute calibrator retention counts
    cal_counts = []
    for t in thresholds:
        n_ret = sum(1 for s in samples if s["p_correct"] > t)
        cal_counts.append(n_ret)

    results = []
    for t, n_ret in zip(thresholds, cal_counts):
        if n_ret == 0:
            results.append({
                "threshold": t, "n_retained": 0,
                "pct_retained": 0.0, "accuracy": None,
            })
            continue
        accs = []
        for _ in range(n_trials):
            idx = rng.choice(n_total, size=n_ret, replace=False)
            accs.append(float(labels[idx].mean()))
        results.append({
            "threshold": t,
            "n_retained": n_ret,
            "pct_retained": n_ret / n_total,
            "accuracy": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
        })
    return results


def quality_quantity_length(samples, thresholds=THRESHOLDS):
    """Response length baseline: sort by output_tokens descending, take top-k
    to match calibrator retention count at each threshold."""
    n_total = len(samples)
    # Sort by descending output_tokens
    sorted_by_length = sorted(samples, key=lambda s: s["output_tokens"], reverse=True)

    cal_counts = []
    for t in thresholds:
        n_ret = sum(1 for s in samples if s["p_correct"] > t)
        cal_counts.append(n_ret)

    results = []
    for t, n_ret in zip(thresholds, cal_counts):
        if n_ret == 0:
            results.append({
                "threshold": t, "n_retained": 0,
                "pct_retained": 0.0, "accuracy": None,
            })
            continue
        retained = sorted_by_length[:n_ret]
        acc = np.mean([s["is_correct"] for s in retained])
        results.append({
            "threshold": t,
            "n_retained": n_ret,
            "pct_retained": n_ret / n_total,
            "accuracy": float(acc),
        })
    return results


# ---------------------------------------------------------------------------
# Precision-Recall for Correct Filtering
# ---------------------------------------------------------------------------

def precision_recall_correct(samples, thresholds=THRESHOLDS):
    """Treat p_correct > t as predicting 'this sample is correct'.
    Precision = accuracy of retained set (TP / (TP + FP)).
    Recall = fraction of truly correct samples retained (TP / (TP + FN)).
    """
    n_correct_total = sum(s["is_correct"] for s in samples)
    results = []
    for t in thresholds:
        retained = [s for s in samples if s["p_correct"] > t]
        n_ret = len(retained)
        if n_ret == 0:
            results.append({
                "threshold": t, "precision": None, "recall": None,
                "n_retained": 0,
            })
            continue
        tp = sum(s["is_correct"] for s in retained)
        precision = tp / n_ret  # accuracy of retained
        recall = tp / n_correct_total if n_correct_total > 0 else 0.0
        results.append({
            "threshold": t,
            "precision": float(precision),
            "recall": float(recall),
            "n_retained": n_ret,
            "true_positives": tp,
        })
    return results


# ---------------------------------------------------------------------------
# Per-Benchmark Breakdown
# ---------------------------------------------------------------------------

def per_benchmark_breakdown(samples, threshold=0.7):
    """At a fixed threshold, show accuracy before and after filtering
    for each benchmark."""
    by_bench = defaultdict(list)
    for s in samples:
        by_bench[s["benchmark"]].append(s)

    results = {}
    for bench, bench_samples in sorted(by_bench.items()):
        n_total = len(bench_samples)
        acc_before = np.mean([s["is_correct"] for s in bench_samples])
        retained = [s for s in bench_samples if s["p_correct"] > threshold]
        n_ret = len(retained)
        acc_after = float(np.mean([s["is_correct"] for s in retained])) if n_ret > 0 else None
        results[bench] = {
            "n_total": n_total,
            "n_retained": n_ret,
            "pct_retained": n_ret / n_total,
            "accuracy_before": float(acc_before),
            "accuracy_after": acc_after,
            "accuracy_delta": float(acc_after - acc_before) if acc_after is not None else None,
        }
    return results


# ---------------------------------------------------------------------------
# Summary Metrics
# ---------------------------------------------------------------------------

def compute_auar(thresholds_data):
    """Area Under Accuracy-vs-Retention curve.

    Integrates accuracy over pct_retained. Higher is better.
    Uses trapezoidal rule on (pct_retained, accuracy) pairs, sorted by
    ascending retention.
    """
    points = [
        (d["pct_retained"], d["accuracy"])
        for d in thresholds_data
        if d["accuracy"] is not None and d["pct_retained"] > 0
    ]
    if len(points) < 2:
        return 0.0
    points.sort(key=lambda p: p[0])
    retentions = np.array([p[0] for p in points])
    accuracies = np.array([p[1] for p in points])
    # np.trapezoid is numpy >=2.0; fall back to np.trapz for older versions
    _trapz = getattr(np, "trapezoid", np.trapz)
    return float(_trapz(accuracies, retentions))


def accuracy_at_retention(thresholds_data, target_retention=0.50):
    """Find accuracy closest to the target retention level."""
    best = None
    for d in thresholds_data:
        if d["accuracy"] is None or d["pct_retained"] == 0:
            continue
        if best is None or abs(d["pct_retained"] - target_retention) < abs(best["pct_retained"] - target_retention):
            best = d
    return float(best["accuracy"]) if best else None


def threshold_for_accuracy(thresholds_data, target_accuracy):
    """Find lowest threshold that achieves >= target accuracy."""
    candidates = [
        d for d in thresholds_data
        if d["accuracy"] is not None and d["accuracy"] >= target_accuracy
    ]
    if not candidates:
        return None
    # Lowest threshold = highest retention
    return min(candidates, key=lambda d: d["threshold"])["threshold"]


def compute_summary(cal_data, verb_data, rand_data, length_data):
    """Compute key summary metrics across all methods."""
    summary = {}
    methods = {
        "calibrator": cal_data,
        "verbalized": verb_data,
        "random": rand_data,
        "response_length": length_data,
    }
    for name, data in methods.items():
        if not data:
            continue
        entry = {}
        entry["auar"] = compute_auar(data)
        entry["accuracy_at_50pct_retention"] = accuracy_at_retention(data, 0.50)
        entry["threshold_for_85pct_accuracy"] = threshold_for_accuracy(data, 0.85)
        entry["threshold_for_90pct_accuracy"] = threshold_for_accuracy(data, 0.90)
        summary[name] = entry
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_quality_quantity(all_model_results, fig_path):
    """Main figure: quality (accuracy) vs quantity (retention) for all models
    and all methods."""
    n_models = len(all_model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6), squeeze=False)

    model_names = {
        "gpt5mini": "GPT-5-mini",
        "gpt52": "GPT-5.2",
        "qwen35": "Qwen3.5",
    }

    for col, (target, data) in enumerate(all_model_results.items()):
        ax = axes[0, col]

        # Calibrator
        cal = data["calibrator_thresholds"]
        rets = [d["pct_retained"] for d in cal if d["accuracy"] is not None]
        accs = [d["accuracy"] for d in cal if d["accuracy"] is not None]
        if rets:
            ax.plot(rets, accs, "o-", color="C0", linewidth=2.5, markersize=6,
                    label=f"Calibrator (AUAR={data['summary'].get('calibrator', {}).get('auar', 0):.3f})",
                    zorder=5)

        # Verbalized
        verb = data.get("verbalized_thresholds", [])
        rets_v = [d["pct_retained"] for d in verb if d.get("accuracy") is not None]
        accs_v = [d["accuracy"] for d in verb if d.get("accuracy") is not None]
        if rets_v:
            ax.plot(rets_v, accs_v, "s--", color="C1", linewidth=2, markersize=5,
                    label=f"Verbalized (AUAR={data['summary'].get('verbalized', {}).get('auar', 0):.3f})")

        # Random
        rand = data["random_thresholds"]
        rets_r = [d["pct_retained"] for d in rand if d.get("accuracy") is not None]
        accs_r = [d["accuracy"] for d in rand if d.get("accuracy") is not None]
        stds_r = [d.get("accuracy_std", 0) for d in rand if d.get("accuracy") is not None]
        if rets_r:
            ax.plot(rets_r, accs_r, ":", color="gray", linewidth=2,
                    label=f"Random (AUAR={data['summary'].get('random', {}).get('auar', 0):.3f})")
            ax.fill_between(rets_r,
                            np.array(accs_r) - np.array(stds_r),
                            np.array(accs_r) + np.array(stds_r),
                            color="gray", alpha=0.15)

        # Response length
        length = data["length_thresholds"]
        rets_l = [d["pct_retained"] for d in length if d.get("accuracy") is not None]
        accs_l = [d["accuracy"] for d in length if d.get("accuracy") is not None]
        if rets_l:
            ax.plot(rets_l, accs_l, "^-.", color="C2", linewidth=1.5, markersize=5,
                    label=f"Resp. Length (AUAR={data['summary'].get('response_length', {}).get('auar', 0):.3f})")

        # Base accuracy line
        base_acc = data["base_accuracy"]
        ax.axhline(y=base_acc, color="black", linestyle=":", alpha=0.4,
                    label=f"Unfiltered ({base_acc:.3f})")

        ax.set_xlabel("Retention (fraction of data kept)", fontsize=12)
        ax.set_ylabel("Accuracy of retained data", fontsize=12)
        ax.set_title(f"{model_names.get(target, target)} (N={data['n_samples']})", fontsize=13)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle("UC-C: Data Curation — Quality vs Quantity Tradeoff", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {fig_path}")


def plot_per_benchmark(all_model_results, fig_path, threshold=0.7):
    """Per-benchmark bar chart showing accuracy before/after filtering at the
    given threshold, for the first model that has data."""
    # Pick the model with the most benchmarks
    best_target = None
    best_n = 0
    for target, data in all_model_results.items():
        pb = data.get("per_benchmark", {})
        if len(pb) > best_n:
            best_n = len(pb)
            best_target = target

    if best_target is None or best_n == 0:
        print("  No per-benchmark data to plot")
        return

    model_names = {
        "gpt5mini": "GPT-5-mini",
        "gpt52": "GPT-5.2",
        "qwen35": "Qwen3.5",
    }

    n_models = len(all_model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(max(8, best_n * 0.7) * n_models / max(n_models - 1, 1), 7),
                             squeeze=False)

    for col, (target, data) in enumerate(all_model_results.items()):
        ax = axes[0, col]
        pb = data.get("per_benchmark", {})
        if not pb:
            ax.set_visible(False)
            continue

        benchmarks = sorted(pb.keys())
        acc_before = [pb[b]["accuracy_before"] for b in benchmarks]
        acc_after = [pb[b]["accuracy_after"] if pb[b]["accuracy_after"] is not None else 0 for b in benchmarks]
        pct_ret = [pb[b]["pct_retained"] for b in benchmarks]

        x = np.arange(len(benchmarks))
        w = 0.35
        bars1 = ax.bar(x - w / 2, acc_before, w, label="Before filtering", color="C3", alpha=0.7)
        bars2 = ax.bar(x + w / 2, acc_after, w, label=f"After (p>{threshold})", color="C0", alpha=0.9)

        # Annotate retention percentages
        for i, pct in enumerate(pct_ret):
            ax.text(x[i] + w / 2, acc_after[i] + 0.02, f"{pct:.0%}",
                    ha="center", va="bottom", fontsize=7, color="C0")

        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{model_names.get(target, target)} — p_correct > {threshold}", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("UC-C: Per-Benchmark Accuracy Before/After Calibrator Filtering", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UC-C: Synthetic Data Curation / Distillation (Stage 1)")
    parser.add_argument("--scored_dir", default="data/use_cases/scored_unified",
                        help="Directory with scored JSONL files")
    parser.add_argument("--output_dir", default="data/use_cases/results_unified",
                        help="Directory for results JSON")
    parser.add_argument("--fig_dir", default="figures/use_cases_unified",
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
        base_acc = float(np.mean([s["is_correct"] for s in samples]))

        print(f"\n{'='*70}")
        print(f"UC-C: Data Curation — {target_names.get(target, target)}")
        print(f"{'='*70}")
        print(f"Samples: {n_total}, Base accuracy: {base_acc:.3f}")

        # --- 1. Quality-Quantity Tradeoff ---
        print(f"\n--- Quality-Quantity Tradeoff ---")

        cal_data = quality_quantity_calibrator(samples)
        verb_data = quality_quantity_verbalized(samples)
        rand_data = quality_quantity_random(samples)
        length_data = quality_quantity_length(samples)

        # Print table
        print(f"\n  {'Thresh':>7} | {'Calibrator':>18} | {'Verbalized':>18} | {'Random':>18} | {'Resp Length':>18}")
        print(f"  {'':>7} | {'Ret%':>7} {'Acc':>9} | {'Ret%':>7} {'Acc':>9} | {'Ret%':>7} {'Acc':>9} | {'Ret%':>7} {'Acc':>9}")
        print(f"  {'-'*7}-+-{'-'*18}-+-{'-'*18}-+-{'-'*18}-+-{'-'*18}")

        for i, t in enumerate(THRESHOLDS):
            def fmt(data_list, idx):
                if idx >= len(data_list):
                    return "    N/A      N/A"
                d = data_list[idx]
                ret_s = f"{d['pct_retained']:>6.1%}" if d["pct_retained"] > 0 else "  0.0%"
                acc_s = f"{d['accuracy']:.3f}" if d["accuracy"] is not None else "  N/A"
                return f"{ret_s} {acc_s:>9}"

            print(f"  {t:>7.1f} | {fmt(cal_data, i):>18} | {fmt(verb_data, i):>18} | "
                  f"{fmt(rand_data, i):>18} | {fmt(length_data, i):>18}")

        # --- 2. Precision-Recall for Correct Filtering ---
        print(f"\n--- Precision-Recall (treating p_correct > t as 'correct') ---")
        pr_data = precision_recall_correct(samples)

        print(f"  {'Thresh':>7} {'Precision':>10} {'Recall':>10} {'N retained':>11}")
        print(f"  {'-'*42}")
        for d in pr_data:
            prec_s = f"{d['precision']:.3f}" if d["precision"] is not None else "N/A"
            rec_s = f"{d['recall']:.3f}" if d["recall"] is not None else "N/A"
            print(f"  {d['threshold']:>7.1f} {prec_s:>10} {rec_s:>10} {d['n_retained']:>11}")

        # --- 3. Per-Benchmark Breakdown ---
        print(f"\n--- Per-Benchmark Breakdown (p_correct > 0.7) ---")
        pb_data = per_benchmark_breakdown(samples, threshold=0.7)

        print(f"  {'Benchmark':<20} {'N':>5} {'Ret':>5} {'Ret%':>6} {'Acc Before':>10} {'Acc After':>10} {'Delta':>7}")
        print(f"  {'-'*67}")
        for bench, bd in sorted(pb_data.items()):
            acc_a = f"{bd['accuracy_after']:.3f}" if bd["accuracy_after"] is not None else "N/A"
            delta = f"{bd['accuracy_delta']:+.3f}" if bd["accuracy_delta"] is not None else "N/A"
            print(f"  {bench:<20} {bd['n_total']:>5} {bd['n_retained']:>5} "
                  f"{bd['pct_retained']:>5.0%} {bd['accuracy_before']:>10.3f} "
                  f"{acc_a:>10} {delta:>7}")

        # --- 4. Summary Metrics ---
        summary = compute_summary(cal_data, verb_data, rand_data, length_data)

        print(f"\n--- Summary Metrics ---")
        print(f"  {'Method':<20} {'AUAR':>7} {'Acc@50%ret':>11} {'Thresh@85%':>11} {'Thresh@90%':>11}")
        print(f"  {'-'*64}")
        method_labels = {
            "calibrator": "Calibrator",
            "verbalized": "Verbalized",
            "random": "Random",
            "response_length": "Resp. Length",
        }
        for method_key in ["calibrator", "verbalized", "random", "response_length"]:
            if method_key not in summary:
                continue
            s = summary[method_key]
            auar_s = f"{s['auar']:.3f}"
            acc50_s = f"{s['accuracy_at_50pct_retention']:.3f}" if s["accuracy_at_50pct_retention"] is not None else "N/A"
            t85_s = f"{s['threshold_for_85pct_accuracy']:.1f}" if s["threshold_for_85pct_accuracy"] is not None else ">0.9"
            t90_s = f"{s['threshold_for_90pct_accuracy']:.1f}" if s["threshold_for_90pct_accuracy"] is not None else ">0.9"
            print(f"  {method_labels.get(method_key, method_key):<20} {auar_s:>7} {acc50_s:>11} {t85_s:>11} {t90_s:>11}")

        # Store results for this target
        all_results[target] = {
            "n_samples": n_total,
            "base_accuracy": base_acc,
            "calibrator_thresholds": cal_data,
            "verbalized_thresholds": verb_data,
            "random_thresholds": rand_data,
            "length_thresholds": length_data,
            "precision_recall": pr_data,
            "per_benchmark": pb_data,
            "summary": summary,
        }

    if not all_results:
        print("\nERROR: No scored data found. Nothing to do.")
        return

    # --- Figures ---
    plot_quality_quantity(all_results, f"{args.fig_dir}/uc_c_quality_quantity.pdf")
    plot_per_benchmark(all_results, f"{args.fig_dir}/uc_c_per_benchmark.pdf", threshold=0.7)

    # --- Save JSON ---
    out_path = f"{args.output_dir}/uc_c_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- Final Summary ---
    print(f"\n{'='*70}")
    print("UC-C Summary: Calibrator vs Baselines")
    print(f"{'='*70}")
    for target in all_results:
        s = all_results[target]["summary"]
        cal_auar = s.get("calibrator", {}).get("auar", 0)
        rand_auar = s.get("random", {}).get("auar", 0)
        verb_auar = s.get("verbalized", {}).get("auar", 0)
        len_auar = s.get("response_length", {}).get("auar", 0)
        print(f"  {target_names.get(target, target)}:")
        print(f"    AUAR — Calibrator: {cal_auar:.3f}  Verbalized: {verb_auar:.3f}  "
              f"Random: {rand_auar:.3f}  Length: {len_auar:.3f}")
        cal_acc50 = s.get("calibrator", {}).get("accuracy_at_50pct_retention")
        rand_acc50 = s.get("random", {}).get("accuracy_at_50pct_retention")
        if cal_acc50 is not None and rand_acc50 is not None:
            print(f"    Acc@50% retention — Calibrator: {cal_acc50:.3f}  Random: {rand_acc50:.3f}  "
                  f"(+{cal_acc50 - rand_acc50:.3f})")


if __name__ == "__main__":
    main()
