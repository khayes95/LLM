#!/usr/bin/env python3
"""UC8: Deployment Monitoring — batch-level quality estimation.

Uses calibrator P(correct) to monitor model performance in deployment:
- Detect accuracy drops from aggregated batch scores
- Bootstrap analysis for minimum reliable batch size
- Cross-model shift detection
- Alert threshold precision/recall

Usage:
    python scripts/uc8_deployment_monitoring.py
    python scripts/uc8_deployment_monitoring.py --scored_dir data/use_cases/scored_v2
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def load_scored(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def benchmark_level_analysis(samples, label):
    """Compute per-benchmark mean P(correct) vs actual accuracy."""
    benchmarks = defaultdict(lambda: {"p_correct": [], "is_correct": []})
    for s in samples:
        benchmarks[s["benchmark"]]["p_correct"].append(s["p_correct"])
        benchmarks[s["benchmark"]]["is_correct"].append(s["is_correct"])
        if s.get("verbalized_confidence") is not None:
            benchmarks[s["benchmark"]].setdefault("verbalized", []).append(s["verbalized_confidence"])

    results = {}
    for bench, data in benchmarks.items():
        if len(data["p_correct"]) < 10:
            continue
        results[bench] = {
            "n": len(data["p_correct"]),
            "mean_p": float(np.mean(data["p_correct"])),
            "actual_acc": float(np.mean(data["is_correct"])),
            "std_p": float(np.std(data["p_correct"])),
            "mean_verb": float(np.mean(data.get("verbalized", [0.5]))) if data.get("verbalized") else None,
        }

    # Rank correlation
    benches = [b for b in results if results[b]["n"] >= 30]
    if len(benches) >= 5:
        mean_ps = [results[b]["mean_p"] for b in benches]
        actual_accs = [results[b]["actual_acc"] for b in benches]
        r_cal, p_cal = spearmanr(mean_ps, actual_accs)

        mean_verbs = [results[b]["mean_verb"] for b in benches if results[b]["mean_verb"] is not None]
        actual_verbs = [results[b]["actual_acc"] for b in benches if results[b]["mean_verb"] is not None]
        r_verb, p_verb = spearmanr(mean_verbs, actual_verbs) if len(mean_verbs) >= 5 else (None, None)
    else:
        r_cal, p_cal, r_verb, p_verb = None, None, None, None

    return results, r_cal, p_cal, r_verb, p_verb


def bootstrap_batch_detection(samples, batch_sizes=(25, 50, 100, 200), n_bootstrap=1000, seed=42):
    """Simulate batch-level monitoring at different batch sizes.

    For each batch size, draw random batches and check if batch mean P(correct)
    correlates with batch accuracy.
    """
    rng = np.random.RandomState(seed)
    all_p = np.array([s["p_correct"] for s in samples])
    all_correct = np.array([s["is_correct"] for s in samples])
    n_total = len(samples)

    results = {}
    for bs in batch_sizes:
        if bs > n_total // 2:
            continue

        batch_means = []
        batch_accs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_total, size=bs, replace=True)
            batch_means.append(all_p[idx].mean())
            batch_accs.append(all_correct[idx].mean())

        r, p = spearmanr(batch_means, batch_accs)
        results[bs] = {
            "spearman_r": float(r),
            "p_value": float(p),
            "significant": p < 0.05,
            "mean_batch_mean": float(np.mean(batch_means)),
            "std_batch_mean": float(np.std(batch_means)),
        }

    return results


def alert_threshold_analysis(benchmark_results, accuracy_thresholds=(0.3, 0.4, 0.5)):
    """Compute precision/recall of alerting based on mean P(correct) threshold."""
    results = {}

    for acc_threshold in accuracy_thresholds:
        # Ground truth: is benchmark accuracy below threshold?
        benches = [(b, d) for b, d in benchmark_results.items() if d["n"] >= 30]

        best_f1 = 0
        best_config = {}

        for p_threshold in np.arange(0.1, 0.9, 0.05):
            tp = sum(1 for b, d in benches if d["mean_p"] < p_threshold and d["actual_acc"] < acc_threshold)
            fp = sum(1 for b, d in benches if d["mean_p"] < p_threshold and d["actual_acc"] >= acc_threshold)
            fn = sum(1 for b, d in benches if d["mean_p"] >= p_threshold and d["actual_acc"] < acc_threshold)
            tn = sum(1 for b, d in benches if d["mean_p"] >= p_threshold and d["actual_acc"] >= acc_threshold)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_config = {
                    "p_threshold": float(p_threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                }

        results[str(acc_threshold)] = best_config

    return results


def cross_model_shift_detection(all_scored, reference_model="gpt5mini"):
    """Detect when cross-model calibrator scores diverge from reference."""
    if reference_model not in all_scored:
        return {}

    ref_samples = all_scored[reference_model]
    ref_benchmarks = defaultdict(list)
    for s in ref_samples:
        ref_benchmarks[s["benchmark"]].append(s["p_correct"])

    ref_stats = {}
    for bench, scores in ref_benchmarks.items():
        if len(scores) >= 20:
            ref_stats[bench] = {"mean": np.mean(scores), "std": np.std(scores) + 1e-8}

    results = {}
    for target, samples in all_scored.items():
        if target == reference_model:
            continue

        target_benchmarks = defaultdict(lambda: {"p_correct": [], "is_correct": []})
        for s in samples:
            target_benchmarks[s["benchmark"]]["p_correct"].append(s["p_correct"])
            target_benchmarks[s["benchmark"]]["is_correct"].append(s["is_correct"])

        shift_signals = {}
        for bench in target_benchmarks:
            if bench not in ref_stats:
                continue
            t_data = target_benchmarks[bench]
            if len(t_data["p_correct"]) < 20:
                continue

            t_mean = np.mean(t_data["p_correct"])
            t_acc = np.mean(t_data["is_correct"])
            ref_mean = ref_stats[bench]["mean"]
            ref_std = ref_stats[bench]["std"]

            z_batch = (t_mean - ref_mean) / ref_std

            shift_signals[bench] = {
                "target_mean_p": float(t_mean),
                "target_accuracy": float(t_acc),
                "ref_mean_p": float(ref_mean),
                "z_shift": float(z_batch),
                "n_target": len(t_data["p_correct"]),
            }

        results[target] = shift_signals

    return results


def plot_deployment_monitoring(all_bench_results, bootstrap_results, shift_results, output_path):
    """Generate deployment monitoring figures."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: Scatter of mean P(correct) vs actual accuracy
    ax = axes[0, 0]
    colors_map = {"gpt5mini": "C0", "gpt52": "C1", "qwen35": "C2"}
    for target, (bench_results, r, p, _, _) in all_bench_results.items():
        xs = [d["mean_p"] for d in bench_results.values() if d["n"] >= 30]
        ys = [d["actual_acc"] for d in bench_results.values() if d["n"] >= 30]
        labels = [b for b, d in bench_results.items() if d["n"] >= 30]
        ax.scatter(xs, ys, alpha=0.6, label=f"{target} (r={r:.2f})" if r else target,
                   color=colors_map.get(target, "gray"), s=50)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.set_xlabel("Mean P(correct)", fontsize=11)
    ax.set_ylabel("Actual Accuracy", fontsize=11)
    ax.set_title("Benchmark-Level: Predicted vs Actual", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Top-right: Bootstrap batch size analysis
    ax = axes[0, 1]
    for target, bsr in bootstrap_results.items():
        batch_sizes = sorted(bsr.keys())
        correlations = [bsr[bs]["spearman_r"] for bs in batch_sizes]
        ax.plot(batch_sizes, correlations, "o-", label=target, color=colors_map.get(target, "gray"))
    ax.axhline(y=0.7, color="gray", linestyle=":", alpha=0.5, label="r=0.7 threshold")
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("Spearman Correlation\n(batch mean P vs batch acc)", fontsize=11)
    ax.set_title("Minimum Batch Size for Reliable Detection", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Cross-model shift detection
    ax = axes[1, 0]
    for target, shifts in shift_results.items():
        benches = sorted(shifts.keys())
        z_vals = [shifts[b]["z_shift"] for b in benches]
        accs = [shifts[b]["target_accuracy"] for b in benches]
        ax.scatter(z_vals, accs, alpha=0.6, label=target, color=colors_map.get(target, "gray"), s=50)
        # Annotate outliers
        for b, z, a in zip(benches, z_vals, accs):
            if abs(z) > 2:
                ax.annotate(b, (z, a), fontsize=7, alpha=0.7)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Z-score shift (vs reference model)", fontsize=11)
    ax.set_ylabel("Target Model Accuracy", fontsize=11)
    ax.set_title("Cross-Model Shift Detection", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Per-benchmark calibration error
    ax = axes[1, 1]
    # Show calibration error = |mean_p - actual_acc| for each model
    for target, (bench_results, _, _, _, _) in all_bench_results.items():
        benches = sorted([b for b in bench_results if bench_results[b]["n"] >= 50])
        cal_errors = [abs(bench_results[b]["mean_p"] - bench_results[b]["actual_acc"]) for b in benches]
        ax.bar(np.arange(len(benches)) + list(colors_map.keys()).index(target) * 0.25 - 0.25,
               cal_errors, 0.25, label=target, color=colors_map.get(target, "gray"), alpha=0.7)
    # Only show labels for largest model set
    max_target = max(all_bench_results, key=lambda t: len(all_bench_results[t][0]))
    benches = sorted([b for b in all_bench_results[max_target][0] if all_bench_results[max_target][0][b]["n"] >= 50])
    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels(benches, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("|Mean P - Actual Acc|", fontsize=11)
    ax.set_title("Per-Benchmark Calibration Error", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_v2")
    parser.add_argument("--output_dir", default="data/use_cases/results")
    parser.add_argument("--fig_dir", default="figures/use_cases")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    all_scored = {}
    for target in ["gpt5mini", "gpt52", "qwen35"]:
        path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if path.exists():
            all_scored[target] = load_scored(path)

    print("=" * 70)
    print("UC8: Deployment Monitoring / Batch Quality Estimation")
    print("=" * 70)

    # 1. Benchmark-level analysis per model
    all_bench_results = {}
    for target, samples in all_scored.items():
        bench_results, r_cal, p_cal, r_verb, p_verb = benchmark_level_analysis(samples, target)
        all_bench_results[target] = (bench_results, r_cal, p_cal, r_verb, p_verb)

        r_str = f"r={r_cal:.3f} (p={p_cal:.4f})" if r_cal is not None else "N/A"
        v_str = f"r={r_verb:.3f} (p={p_verb:.4f})" if r_verb is not None else "N/A"
        print(f"\n  {target}: Calibrator {r_str}, Verbalized {v_str}")
        print(f"  {'Benchmark':<20} {'N':>5} {'MeanP':>7} {'ActAcc':>7} {'CalErr':>7}")
        print(f"  {'-'*50}")
        for bench in sorted(bench_results, key=lambda b: bench_results[b]["n"], reverse=True):
            d = bench_results[bench]
            cal_err = d["mean_p"] - d["actual_acc"]
            print(f"  {bench:<20} {d['n']:>5} {d['mean_p']:>7.3f} {d['actual_acc']:>7.3f} {cal_err:>+7.3f}")

    # 2. Bootstrap batch simulation
    print(f"\n--- Bootstrap Batch Detection ---")
    bootstrap_results = {}
    for target, samples in all_scored.items():
        bsr = bootstrap_batch_detection(samples, batch_sizes=(25, 50, 100, 200))
        bootstrap_results[target] = bsr
        print(f"\n  {target}:")
        for bs in sorted(bsr.keys()):
            r = bsr[bs]
            sig = "*" if r["significant"] else " "
            print(f"    batch_size={bs:>4}: r={r['spearman_r']:.3f}{sig} (p={r['p_value']:.4f})")

    # 3. Cross-model shift detection
    print(f"\n--- Cross-Model Shift Detection (ref=gpt5mini) ---")
    shift_results = cross_model_shift_detection(all_scored, "gpt5mini")
    for target, shifts in shift_results.items():
        print(f"\n  {target}:")
        print(f"  {'Benchmark':<20} {'TargetP':>8} {'RefP':>8} {'Z-shift':>8} {'TargetAcc':>9}")
        print(f"  {'-'*58}")
        for bench in sorted(shifts, key=lambda b: abs(shifts[b]["z_shift"]), reverse=True):
            d = shifts[bench]
            flag = " **" if abs(d["z_shift"]) > 2 else ""
            print(f"  {bench:<20} {d['target_mean_p']:>8.3f} {d['ref_mean_p']:>8.3f} "
                  f"{d['z_shift']:>+8.2f} {d['target_accuracy']:>9.3f}{flag}")

    # 4. Alert threshold analysis
    print(f"\n--- Alert Threshold Analysis ---")
    for target, (bench_results, _, _, _, _) in all_bench_results.items():
        alert_results = alert_threshold_analysis(bench_results)
        print(f"\n  {target}:")
        for acc_thresh, config in alert_results.items():
            if config:
                print(f"    Accuracy < {acc_thresh}: P_thresh={config['p_threshold']:.2f}, "
                      f"F1={config['f1']:.3f}, Prec={config['precision']:.3f}, Rec={config['recall']:.3f}")

    # Plot
    plot_deployment_monitoring(all_bench_results, bootstrap_results, shift_results,
                              f"{args.fig_dir}/uc8_deployment_monitoring.pdf")

    # Save
    # Convert tuples for JSON
    serializable_bench = {}
    for target, (bench_results, r_cal, p_cal, r_verb, p_verb) in all_bench_results.items():
        serializable_bench[target] = {
            "benchmarks": bench_results,
            "rank_correlation_calibrator": {"r": r_cal, "p": p_cal} if r_cal else None,
            "rank_correlation_verbalized": {"r": r_verb, "p": p_verb} if r_verb else None,
        }

    out_path = f"{args.output_dir}/uc8_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "benchmark_analysis": serializable_bench,
            "bootstrap_batch": bootstrap_results,
            "cross_model_shift": shift_results,
        }, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
