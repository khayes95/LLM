#!/usr/bin/env python3
"""UC4: Difficulty Estimation — rank benchmarks/questions by difficulty.

Analyzes calibrator's P(correct) as a difficulty estimator:
- Benchmark difficulty ranking vs actual accuracy
- Overconfident error identification
- Cross-model difficulty correlation

Usage:
    python scripts/uc4_difficulty_estimation.py
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr


def load_scored(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def benchmark_difficulty(samples):
    """Per-benchmark difficulty analysis."""
    bench_data = defaultdict(lambda: {"p_correct": [], "is_correct": [], "verbalized": []})
    for s in samples:
        bench_data[s["benchmark"]]["p_correct"].append(s["p_correct"])
        bench_data[s["benchmark"]]["is_correct"].append(s["is_correct"])
        if s.get("verbalized_confidence") is not None:
            bench_data[s["benchmark"]]["verbalized"].append(s["verbalized_confidence"])

    results = {}
    for bench, data in bench_data.items():
        if len(data["p_correct"]) < 5:
            continue
        acc = np.mean(data["is_correct"])
        mean_p = np.mean(data["p_correct"])
        overconf = mean_p - acc  # positive = overconfident

        # Overconfident errors: P(correct) > 0.7 but wrong
        overconf_errors = sum(
            1 for p, c in zip(data["p_correct"], data["is_correct"])
            if p > 0.7 and c == 0
        )
        n_wrong = sum(1 - c for c in data["is_correct"])
        overconf_error_rate = overconf_errors / max(n_wrong, 1)

        results[bench] = {
            "n": len(data["p_correct"]),
            "accuracy": float(acc),
            "mean_p_correct": float(mean_p),
            "overconfidence": float(overconf),
            "overconfident_errors": overconf_errors,
            "overconfident_error_rate": float(overconf_error_rate),
            "n_wrong": int(n_wrong),
        }
        if data["verbalized"]:
            results[bench]["mean_verbalized"] = float(np.mean(data["verbalized"]))
            results[bench]["verb_overconfidence"] = float(np.mean(data["verbalized"]) - acc)

    return results


def cross_model_correlation(all_scored):
    """Correlate P(correct) across models for same questions."""
    # Build per-question scores across models
    question_scores = defaultdict(dict)
    for target, samples in all_scored.items():
        for s in samples:
            key = (s["benchmark"], s["id"])
            question_scores[key][target] = s["p_correct"]

    results = {}
    models = list(all_scored.keys())
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            paired_m1 = []
            paired_m2 = []
            for key, scores in question_scores.items():
                if m1 in scores and m2 in scores:
                    paired_m1.append(scores[m1])
                    paired_m2.append(scores[m2])

            if len(paired_m1) >= 20:
                corr, pval = spearmanr(paired_m1, paired_m2)
                results[f"{m1}_vs_{m2}"] = {
                    "spearman_r": float(corr),
                    "p_value": float(pval),
                    "n_paired": len(paired_m1),
                }

    return results


def plot_difficulty(bench_results, target_name, output_path):
    """Plot difficulty estimation results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    benches = sorted(bench_results.keys(), key=lambda b: bench_results[b]["overconfidence"], reverse=True)
    if len(benches) > 20:
        benches = benches[:20]

    # Left: Actual accuracy vs Mean P(correct)
    ax = axes[0]
    accs = [bench_results[b]["accuracy"] for b in benches]
    mean_ps = [bench_results[b]["mean_p_correct"] for b in benches]
    ns = [bench_results[b]["n"] for b in benches]

    ax.scatter(accs, mean_ps, s=[max(20, min(200, n)) for n in ns], alpha=0.7, color="C0")
    for b, a, p in zip(benches, accs, mean_ps):
        ax.annotate(b, (a, p), fontsize=7, alpha=0.8)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.set_xlabel("Actual Accuracy", fontsize=12)
    ax.set_ylabel("Mean P(correct)", fontsize=12)
    ax.set_title(f"Calibration by Benchmark — {target_name}", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Overconfidence bar chart
    ax = axes[1]
    overconfs = [bench_results[b]["overconfidence"] for b in benches]
    colors = ["C3" if oc > 0.2 else "C1" if oc > 0.05 else "C2" for oc in overconfs]
    ax.barh(range(len(benches)), overconfs, color=colors)
    ax.set_yticks(range(len(benches)))
    ax.set_yticklabels(benches, fontsize=8)
    ax.set_xlabel("Overconfidence (mean P - accuracy)", fontsize=12)
    ax.set_title("Overconfidence by Benchmark", fontsize=13)
    ax.axvline(x=0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

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

    targets = ["gpt5mini", "gpt52", "qwen35"]
    names = {"gpt5mini": "GPT-5-mini (in-dist)", "gpt52": "GPT-5.2",
             "qwen35": "Qwen3.5"}

    all_results = {}
    all_scored = {}

    for target in targets:
        scored_path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if not scored_path.exists():
            continue

        samples = load_scored(scored_path)
        all_scored[target] = samples

        print(f"\n{'='*70}")
        print(f"UC4: Difficulty Estimation — {names[target]}")
        print(f"{'='*70}")

        bench_results = benchmark_difficulty(samples)

        # Rank correlations (multiple measures)
        benches = [b for b in bench_results if bench_results[b]["n"] >= 10]
        if len(benches) >= 5:
            accs = [bench_results[b]["accuracy"] for b in benches]
            mean_ps = [bench_results[b]["mean_p_correct"] for b in benches]

            spearman_r, spearman_p = spearmanr(accs, mean_ps)
            kendall_r, kendall_p = kendalltau(accs, mean_ps)
            pearson_r, pearson_p = pearsonr(accs, mean_ps)

            print(f"Calibrator rank correlations (accuracy vs mean P):")
            print(f"  Spearman r={spearman_r:.3f} (p={spearman_p:.4f})")
            print(f"  Kendall tau={kendall_r:.3f} (p={kendall_p:.4f})")
            print(f"  Pearson r={pearson_r:.3f} (p={pearson_p:.4f})")

            corr = spearman_r
            pval = spearman_p

            # Verbalized baseline
            verb_benches = [b for b in benches if "mean_verbalized" in bench_results[b]]
            if verb_benches:
                v_accs = [bench_results[b]["accuracy"] for b in verb_benches]
                v_verbs = [bench_results[b]["mean_verbalized"] for b in verb_benches]
                v_spearman, v_sp = spearmanr(v_accs, v_verbs)
                v_kendall, v_kp = kendalltau(v_accs, v_verbs)
                print(f"\nVerbalized rank correlations (accuracy vs verbalized):")
                print(f"  Spearman r={v_spearman:.3f} (p={v_sp:.4f})")
                print(f"  Kendall tau={v_kendall:.3f} (p={v_kp:.4f})")

                # Significance of difference (Fisher z-transform)
                n_b = len(verb_benches)
                z_cal = np.arctanh(spearman_r)
                z_verb = np.arctanh(v_spearman)
                se_diff = np.sqrt(2 / (n_b - 3))
                z_test = (z_cal - z_verb) / se_diff
                print(f"\n  Cal vs Verb difference: z={z_test:.2f} "
                      f"({'significant' if abs(z_test) > 1.96 else 'not significant'} at p<0.05)")

        # Print benchmark table
        print(f"\n  {'Benchmark':<20} {'Acc':>6} {'MeanP':>6} {'Overconf':>9} "
              f"{'OC Errors':>10} {'N':>5}")
        print(f"  {'-'*60}")
        for b in sorted(bench_results, key=lambda x: bench_results[x]["overconfidence"], reverse=True):
            r = bench_results[b]
            print(f"  {b:<20} {r['accuracy']:>6.1%} {r['mean_p_correct']:>6.3f} "
                  f"{r['overconfidence']:>+9.3f} {r['overconfident_errors']:>5}/{r['n_wrong']:<4} "
                  f"{r['n']:>5}")

        all_results[target] = {
            "benchmarks": bench_results,
            "spearman_r": float(spearman_r) if len(benches) >= 5 else None,
            "kendall_tau": float(kendall_r) if len(benches) >= 5 else None,
            "pearson_r": float(pearson_r) if len(benches) >= 5 else None,
            "rank_correlation": float(corr) if len(benches) >= 5 else None,
        }

        plot_difficulty(bench_results, names[target],
                        f"{args.fig_dir}/uc4_difficulty_{target}.pdf")

    # Cross-model correlation
    if len(all_scored) >= 2:
        print(f"\n{'='*70}")
        print("Cross-Model Difficulty Correlation")
        print(f"{'='*70}")
        xm_corr = cross_model_correlation(all_scored)
        for pair, data in xm_corr.items():
            print(f"  {pair}: r={data['spearman_r']:.3f} (n={data['n_paired']}, p={data['p_value']:.4f})")
        all_results["cross_model_correlation"] = xm_corr

    out_path = f"{args.output_dir}/uc4_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
