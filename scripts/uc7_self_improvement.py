#!/usr/bin/env python3
"""UC7 [LONGSHOT]: UQ-Guided Self-Improvement — find systematic weaknesses.

Analyzes where models are most overconfident (high P but wrong),
identifies cross-model weakness patterns, and proposes targeted training.

Usage:
    python scripts/uc7_self_improvement.py
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


DOMAIN_MAP = {
    "factual": ["simpleqa", "gpqa", "chembench", "hallusionbench"],
    "reasoning": ["bbeh", "arc_agi", "multichallenge"],
    "math": ["omnimath", "mathvista", "mathvision", "mathverse"],
    "knowledge": ["mmlu_pro", "mmmu", "hle"],
    "multimodal": ["charxiv", "mmstar", "mmvet", "realworldqa", "vizwiz"],
    "coding": ["prbench"],
    "long_context": ["livebench", "oolong", "longbench"],
}


def load_scored(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def get_domain(benchmark):
    for domain, benches in DOMAIN_MAP.items():
        if benchmark in benches:
            return domain
    return "other"


def overconfident_error_analysis(samples, p_threshold=0.7):
    """Find overconfident errors: P(correct) > threshold but actually wrong."""
    bench_data = defaultdict(lambda: {"total": 0, "wrong": 0, "oc_errors": 0,
                                       "p_correct": [], "is_correct": []})

    for s in samples:
        b = s["benchmark"]
        bench_data[b]["total"] += 1
        bench_data[b]["p_correct"].append(s["p_correct"])
        bench_data[b]["is_correct"].append(s["is_correct"])
        if not s["is_correct"]:
            bench_data[b]["wrong"] += 1
            if s["p_correct"] > p_threshold:
                bench_data[b]["oc_errors"] += 1

    results = {}
    for bench, data in bench_data.items():
        if data["total"] < 5:
            continue
        results[bench] = {
            "n": data["total"],
            "n_wrong": data["wrong"],
            "n_overconfident_errors": data["oc_errors"],
            "overconfident_error_rate": data["oc_errors"] / max(data["wrong"], 1),
            "error_rate": data["wrong"] / data["total"],
            "accuracy": 1 - data["wrong"] / data["total"],
            "mean_p": float(np.mean(data["p_correct"])),
            "domain": get_domain(bench),
        }
    return results


def cross_model_weakness_agreement(all_scored, p_threshold=0.7):
    """Do models share the same weak spots?"""
    model_weaknesses = {}
    for target, samples in all_scored.items():
        oc = overconfident_error_analysis(samples, p_threshold)
        model_weaknesses[target] = {
            bench: data["overconfident_error_rate"]
            for bench, data in oc.items()
        }

    # Pairwise correlation of overconfident error rates across benchmarks
    models = list(model_weaknesses.keys())
    correlations = {}
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            shared = set(model_weaknesses[m1].keys()) & set(model_weaknesses[m2].keys())
            if len(shared) >= 5:
                rates1 = [model_weaknesses[m1][b] for b in shared]
                rates2 = [model_weaknesses[m2][b] for b in shared]
                corr, pval = spearmanr(rates1, rates2)
                correlations[f"{m1}_vs_{m2}"] = {
                    "spearman_r": float(corr),
                    "p_value": float(pval),
                    "n_benchmarks": len(shared),
                }

    return model_weaknesses, correlations


def difficulty_stratification(samples, n_bins=5):
    """Bin questions by P(correct), analyze error patterns in each bin."""
    ps = np.array([s["p_correct"] for s in samples])
    cs = np.array([s["is_correct"] for s in samples])
    bins = np.linspace(0, 1, n_bins + 1)

    results = []
    for i in range(n_bins):
        mask = (ps >= bins[i]) & (ps < bins[i+1] + (1e-6 if i == n_bins-1 else 0))
        if mask.sum() == 0:
            continue
        bin_ps = ps[mask]
        bin_cs = cs[mask]
        results.append({
            "bin_low": float(bins[i]),
            "bin_high": float(bins[i+1]),
            "n": int(mask.sum()),
            "accuracy": float(bin_cs.mean()),
            "mean_p": float(bin_ps.mean()),
            "overconfidence": float(bin_ps.mean() - bin_cs.mean()),
            "n_errors": int((1 - bin_cs).sum()),
        })
    return results


def plot_self_improvement(all_oc, correlations, strat, output_path):
    """Plot self-improvement analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: Overconfident error rate heatmap
    ax = axes[0]
    models = sorted(all_oc.keys())
    benchmarks = set()
    for m in models:
        benchmarks.update(all_oc[m].keys())
    benchmarks = sorted(benchmarks)

    if benchmarks and models:
        data = np.full((len(benchmarks), len(models)), np.nan)
        for j, m in enumerate(models):
            for i, b in enumerate(benchmarks):
                if b in all_oc[m]:
                    data[i, j] = all_oc[m][b]

        im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace("gpt5mini", "GPT-5-mini").replace("gpt52", "GPT-5.2").replace("qwen35", "Qwen3.5")
                            for m in models], fontsize=9)
        ax.set_yticks(range(len(benchmarks)))
        ax.set_yticklabels(benchmarks, fontsize=7)
        ax.set_title("Overconfident Error Rate\n(P>0.7 but wrong / all wrong)", fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Middle: Difficulty stratification
    ax = axes[1]
    if strat:
        bin_labels = [f"{s['bin_low']:.1f}-{s['bin_high']:.1f}" for s in strat]
        accs = [s["accuracy"] for s in strat]
        mean_ps = [s["mean_p"] for s in strat]
        ns = [s["n"] for s in strat]

        x = np.arange(len(strat))
        ax.bar(x, accs, color="C0", alpha=0.7, label="Actual accuracy")
        ax.bar(x, mean_ps, color="C1", alpha=0.3, label="Mean P(correct)")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=9)
        ax.set_xlabel("P(correct) bin", fontsize=12)
        ax.set_ylabel("Rate", fontsize=12)
        ax.set_title("Calibration by Difficulty Bin", fontsize=13)
        ax.legend(fontsize=9)

        # Annotate N per bin
        for i, n in enumerate(ns):
            ax.text(i, 0.02, f"n={n}", ha="center", fontsize=7, color="gray")

    # Right: Domain-level weakness comparison
    ax = axes[2]
    if all_oc:
        # Aggregate by domain for first model
        first_model = models[0]
        domain_oc = defaultdict(list)
        for bench, rate in all_oc[first_model].items():
            domain_oc[get_domain(bench)].append(rate)

        domains = sorted(domain_oc.keys())
        mean_rates = [np.mean(domain_oc[d]) for d in domains]
        colors = ["C3" if r > 0.5 else "C1" if r > 0.2 else "C2" for r in mean_rates]
        ax.barh(range(len(domains)), mean_rates, color=colors)
        ax.set_yticks(range(len(domains)))
        ax.set_yticklabels(domains, fontsize=10)
        ax.set_xlabel("Mean Overconfident Error Rate", fontsize=12)
        ax.set_title("Weakness by Domain", fontsize=13)
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2")
    parser.add_argument("--output_dir", default="data/use_cases/results_test_only_v2")
    parser.add_argument("--fig_dir", default="figures/use_cases")
    parser.add_argument("--p_threshold", type=float, default=0.7,
                        help="P(correct) threshold for 'overconfident'")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    all_scored = {}
    for target in ["gpt5mini", "gpt52", "qwen35"]:
        path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if path.exists():
            all_scored[target] = load_scored(path)

    if not all_scored:
        print("ERROR: No scored data found")
        return

    print("=" * 70)
    print("UC7 [LONGSHOT]: UQ-Guided Self-Improvement")
    print("=" * 70)

    # Overconfident error analysis per model
    all_oc = {}
    for target, samples in all_scored.items():
        print(f"\n--- {target} ({len(samples)} samples) ---")
        oc = overconfident_error_analysis(samples, args.p_threshold)

        print(f"\n  {'Benchmark':<20} {'Acc':>6} {'Errors':>7} {'OC Errors':>10} {'OC Rate':>8}")
        print(f"  {'-'*55}")
        for bench in sorted(oc, key=lambda b: oc[b]["overconfident_error_rate"], reverse=True):
            r = oc[bench]
            print(f"  {bench:<20} {r['accuracy']:>6.1%} {r['n_wrong']:>7} "
                  f"{r['n_overconfident_errors']:>10} {r['overconfident_error_rate']:>8.1%}")

        # Domain summary
        domain_oc = defaultdict(lambda: {"errors": 0, "oc_errors": 0, "total": 0})
        for bench, data in oc.items():
            d = data["domain"]
            domain_oc[d]["errors"] += data["n_wrong"]
            domain_oc[d]["oc_errors"] += data["n_overconfident_errors"]
            domain_oc[d]["total"] += data["n"]

        print(f"\n  Domain summary:")
        for d in sorted(domain_oc, key=lambda x: domain_oc[x]["oc_errors"] / max(domain_oc[x]["errors"], 1), reverse=True):
            dd = domain_oc[d]
            oc_rate = dd["oc_errors"] / max(dd["errors"], 1)
            print(f"    {d:<15} OC rate: {oc_rate:.1%} ({dd['oc_errors']}/{dd['errors']} errors overconfident)")

        all_oc[target] = {b: oc[b]["overconfident_error_rate"] for b in oc}

    # Cross-model weakness agreement
    model_weaknesses, correlations = cross_model_weakness_agreement(all_scored, args.p_threshold)

    if correlations:
        print(f"\n--- Cross-Model Weakness Agreement ---")
        for pair, data in correlations.items():
            print(f"  {pair}: r={data['spearman_r']:.3f} (n={data['n_benchmarks']}, p={data['p_value']:.4f})")

    # Difficulty stratification (use first target)
    first_target = list(all_scored.keys())[0]
    strat = difficulty_stratification(all_scored[first_target])

    print(f"\n--- Difficulty Stratification ({first_target}) ---")
    print(f"  {'P(correct) bin':<15} {'N':>5} {'Accuracy':>10} {'Overconf':>10}")
    print(f"  {'-'*45}")
    for s in strat:
        print(f"  {s['bin_low']:.1f}-{s['bin_high']:.1f}        {s['n']:>5} "
              f"{s['accuracy']:>10.3f} {s['overconfidence']:>+10.3f}")

    # === Simulated Retry Experiment ===
    print(f"\n--- Simulated Retry Experiment ---")
    print("  (For questions where calibrator says P(correct)<threshold, simulate a retry)")
    retry_results = {}
    for target, samples in all_scored.items():
        labels = np.array([s["is_correct"] for s in samples])
        p_correct = np.array([s["p_correct"] for s in samples])
        base_acc = float(labels.mean())

        # Sweep retry budgets: retry the bottom K% by P(correct)
        print(f"\n  {target} (base accuracy: {base_acc:.3f}):")
        print(f"    {'Budget':>8} {'Calibrator':>12} {'Verbalized':>12} {'Random':>12}")
        print(f"    {'-'*50}")

        target_retry = {}
        for budget_frac in [0.05, 0.10, 0.20, 0.30, 0.50]:
            k = max(1, int(budget_frac * len(samples)))

            # Calibrator: retry lowest P(correct)
            retry_idx = np.argsort(p_correct)[:k]
            # Assume retry gives benchmark-average accuracy for those questions
            new_labels = labels.copy()
            for idx in retry_idx:
                bench = samples[idx]["benchmark"]
                # Average accuracy for this benchmark
                bench_acc = np.mean([s["is_correct"] for s in samples if s["benchmark"] == bench])
                # Simulate: correct with probability bench_acc
                new_labels[idx] = max(new_labels[idx], int(np.random.RandomState(42 + idx).random() < bench_acc))
            cal_retry_acc = float(new_labels.mean())

            # Verbalized: retry lowest verbalized confidence
            verb_confs = np.array([
                s.get("verbalized_confidence") if s.get("verbalized_confidence") is not None else 0.5
                for s in samples
            ])
            verb_retry_idx = np.argsort(verb_confs)[:k]
            verb_labels = labels.copy()
            for idx in verb_retry_idx:
                bench = samples[idx]["benchmark"]
                bench_acc = np.mean([s["is_correct"] for s in samples if s["benchmark"] == bench])
                verb_labels[idx] = max(verb_labels[idx], int(np.random.RandomState(42 + idx).random() < bench_acc))
            verb_retry_acc = float(verb_labels.mean())

            # Random: retry random questions
            rng = np.random.RandomState(42)
            rand_retry_idx = rng.choice(len(samples), size=k, replace=False)
            rand_labels = labels.copy()
            for idx in rand_retry_idx:
                bench = samples[idx]["benchmark"]
                bench_acc = np.mean([s["is_correct"] for s in samples if s["benchmark"] == bench])
                rand_labels[idx] = max(rand_labels[idx], int(np.random.RandomState(42 + idx).random() < bench_acc))
            rand_retry_acc = float(rand_labels.mean())

            print(f"    {budget_frac:>7.0%} {cal_retry_acc:>12.3f} (+{cal_retry_acc-base_acc:.3f})"
                  f" {verb_retry_acc:>8.3f} (+{verb_retry_acc-base_acc:.3f})"
                  f" {rand_retry_acc:>8.3f} (+{rand_retry_acc-base_acc:.3f})")

            target_retry[f"{int(budget_frac*100)}pct"] = {
                "budget_frac": budget_frac,
                "calibrator_acc": cal_retry_acc,
                "verbalized_acc": verb_retry_acc,
                "random_acc": rand_retry_acc,
                "cal_improvement": cal_retry_acc - base_acc,
                "verb_improvement": verb_retry_acc - base_acc,
                "rand_improvement": rand_retry_acc - base_acc,
            }

        retry_results[target] = {"base_accuracy": base_acc, "budgets": target_retry}

    # Actionable recommendations
    print(f"\n--- Actionable Recommendations ---")
    for target, samples in all_scored.items():
        oc = overconfident_error_analysis(samples, args.p_threshold)
        worst = sorted(oc.items(), key=lambda x: x[1]["overconfident_error_rate"], reverse=True)[:3]
        print(f"  {target}: Focus on {', '.join(b for b, _ in worst)} "
              f"(highest overconfident error rates)")

    # Plot
    plot_self_improvement(all_oc, correlations, strat,
                          f"{args.fig_dir}/uc7_self_improvement.pdf")

    # Save
    out_path = f"{args.output_dir}/uc7_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "overconfident_error_rates": {t: all_oc[t] for t in all_oc},
            "cross_model_agreement": correlations,
            "difficulty_stratification": strat,
            "simulated_retry": retry_results,
            "p_threshold": args.p_threshold,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
