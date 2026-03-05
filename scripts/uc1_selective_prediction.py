#!/usr/bin/env python3
"""UC1: Selective Prediction — trade coverage for accuracy.

Reads pre-scored JSONL and computes:
- Selective prediction curves (coverage vs accuracy)
- AURC (Area Under Risk-Coverage curve)
- Coverage @ target accuracy thresholds (90%, 95%)
- Comparison: calibrator vs verbalized vs response length vs random vs oracle
- Bootstrap 95% CIs on AURC and coverage@acc metrics

Usage:
    python scripts/uc1_selective_prediction.py
    python scripts/uc1_selective_prediction.py --scored_dir data/use_cases/scored_test_only_v2
    python scripts/uc1_selective_prediction.py --smoke_test
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
from sklearn.metrics import roc_auc_score


def load_scored(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def selective_prediction_curve(samples, score_key, n_points=200):
    """Compute coverage vs accuracy curve."""
    scores = np.array([s[score_key] for s in samples if s.get(score_key) is not None])
    labels = np.array([s["is_correct"] for s in samples if s.get(score_key) is not None])

    thresholds = np.linspace(0, 1, n_points + 1)
    curve = []
    for t in sorted(thresholds, reverse=True):
        mask = scores >= t
        if mask.sum() == 0:
            continue
        coverage = mask.mean()
        accuracy = labels[mask].mean()
        curve.append({"threshold": float(t), "coverage": float(coverage),
                       "accuracy": float(accuracy), "n_answered": int(mask.sum())})
    return curve


def length_prediction_curve(samples, n_points=200):
    """Response length baseline: sort by descending output_tokens, sweep coverage."""
    tokens = np.array([s.get("output_tokens") or 0 for s in samples])
    labels = np.array([s["is_correct"] for s in samples])
    n = len(labels)

    # Sort by descending tokens (longest = most confident)
    order = np.argsort(-tokens)
    sorted_labels = labels[order]

    curve = []
    for cov_frac in np.linspace(0.01, 1.0, n_points):
        k = max(1, int(n * cov_frac))
        accuracy = sorted_labels[:k].mean()
        curve.append({"coverage": float(cov_frac), "accuracy": float(accuracy)})
    return curve


def compute_aurc(curve):
    """Area Under Risk-Coverage curve (lower is better)."""
    coverages = [p["coverage"] for p in curve]
    risks = [1 - p["accuracy"] for p in curve]
    if len(coverages) < 2:
        return 1.0
    _trapz = getattr(np, "trapezoid", np.trapz)
    return float(_trapz(risks, coverages))


def random_baseline_curve(samples, n_points=200):
    """Random selection baseline."""
    labels = np.array([s["is_correct"] for s in samples])
    base_acc = labels.mean()
    curve = []
    for cov_frac in np.linspace(0.05, 1.0, n_points):
        curve.append({"coverage": float(cov_frac), "accuracy": float(base_acc)})
    return curve


def oracle_curve(samples, n_points=200):
    """Oracle: always select correct samples first."""
    labels = np.array([s["is_correct"] for s in samples])
    n = len(labels)
    n_correct = labels.sum()
    curve = []
    for cov_frac in np.linspace(0.01, 1.0, n_points):
        k = max(1, int(n * cov_frac))
        correct_selected = min(k, n_correct)
        accuracy = correct_selected / k
        curve.append({"coverage": float(cov_frac), "accuracy": float(accuracy)})
    return curve


def coverage_at_accuracy(curve, target_acc):
    """Find maximum coverage that achieves target accuracy."""
    best_cov = 0.0
    for p in curve:
        if p["accuracy"] >= target_acc:
            best_cov = max(best_cov, p["coverage"])
    return best_cov


def accuracy_at_coverage(curve, target_cov):
    """Find accuracy at target coverage level."""
    best = None
    for p in curve:
        if abs(p["coverage"] - target_cov) < 0.02:
            if best is None or abs(p["coverage"] - target_cov) < abs(best["coverage"] - target_cov):
                best = p
    return best["accuracy"] if best else None


def bootstrap_aurc_ci(samples, score_key, n_bootstrap=1000, seed=42):
    """Bootstrap 95% CI for AURC and coverage@90%/95%."""
    rng = np.random.RandomState(seed)
    n = len(samples)
    aurcs = []
    c90s = []
    c95s = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bs = [samples[i] for i in idx]
        curve = selective_prediction_curve(bs, score_key, n_points=100)
        if len(curve) < 2:
            continue
        aurcs.append(compute_aurc(curve))
        c90s.append(coverage_at_accuracy(curve, 0.90))
        c95s.append(coverage_at_accuracy(curve, 0.95))

    ci = {}
    if aurcs:
        ci["aurc_ci"] = [float(np.percentile(aurcs, 2.5)), float(np.percentile(aurcs, 97.5))]
    if c90s:
        ci["cov90_ci"] = [float(np.percentile(c90s, 2.5)), float(np.percentile(c90s, 97.5))]
    if c95s:
        ci["cov95_ci"] = [float(np.percentile(c95s, 2.5)), float(np.percentile(c95s, 97.5))]
    return ci


def plot_curves(results, output_path, target_name):
    """Plot selective prediction curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Coverage vs Accuracy
    ax = axes[0]
    styles = {
        "Calibrator P(correct)": {"color": "C0", "linewidth": 2.5},
        "Verbalized Confidence": {"color": "C1", "linewidth": 2, "linestyle": "--"},
        "Response Length": {"color": "C4", "linewidth": 1.5, "linestyle": "-."},
        "Random": {"color": "gray", "linewidth": 1, "linestyle": ":"},
        "Oracle": {"color": "C2", "linewidth": 1, "linestyle": "-."},
    }
    for method, data in results.items():
        curve = data["curve"]
        covs = [p["coverage"] for p in curve]
        accs = [p["accuracy"] for p in curve]
        ax.plot(covs, accs, label=method, **styles.get(method, {}))

    ax.set_xlabel("Coverage (fraction of questions answered)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Selective Prediction — {target_name}", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Right: Coverage vs Risk
    ax = axes[1]
    for method, data in results.items():
        curve = data["curve"]
        covs = [p["coverage"] for p in curve]
        risks = [1 - p["accuracy"] for p in curve]
        ax.plot(covs, risks, label=f"{method} (AURC={data.get('aurc', 0):.3f})",
                **styles.get(method, {}))

    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("Risk (1 - Accuracy)", fontsize=12)
    ax.set_title(f"Risk-Coverage Curve — {target_name}", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
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
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    n_bootstrap = 50 if args.smoke_test else args.n_bootstrap
    all_results = {}
    targets = ["gpt5mini", "gpt52", "qwen35"]
    target_names = {"gpt5mini": "GPT-5-mini (in-dist)", "gpt52": "GPT-5.2 (cross-model)",
                    "qwen35": "Qwen3.5 (cross-model)"}

    for target in targets:
        scored_path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if not scored_path.exists():
            print(f"Skipping {target}: {scored_path} not found")
            continue

        samples = load_scored(scored_path)
        if args.smoke_test:
            samples = samples[:100]

        print(f"\n{'='*70}")
        print(f"UC1: Selective Prediction — {target_names[target]}")
        print(f"{'='*70}")
        print(f"Samples: {len(samples)}, Accuracy: {np.mean([s['is_correct'] for s in samples]):.3f}")

        results = {}

        # Calibrator P(correct)
        curve = selective_prediction_curve(samples, "p_correct")
        aurc = compute_aurc(curve)
        results["Calibrator P(correct)"] = {"curve": curve, "aurc": aurc}

        # Bootstrap CIs for calibrator
        cal_ci = bootstrap_aurc_ci(samples, "p_correct", n_bootstrap)
        results["Calibrator P(correct)"].update(cal_ci)

        # Verbalized confidence
        verb_samples = [s for s in samples if s.get("verbalized_confidence") is not None]
        if verb_samples:
            verb_curve = selective_prediction_curve(verb_samples, "verbalized_confidence")
            verb_aurc = compute_aurc(verb_curve)
            results["Verbalized Confidence"] = {"curve": verb_curve, "aurc": verb_aurc}

        # Response length baseline
        len_curve = length_prediction_curve(samples)
        len_aurc = compute_aurc(len_curve)
        results["Response Length"] = {"curve": len_curve, "aurc": len_aurc}

        # Random baseline
        rand_curve = random_baseline_curve(samples)
        results["Random"] = {"curve": rand_curve, "aurc": compute_aurc(rand_curve)}

        # Oracle
        oracle = oracle_curve(samples)
        results["Oracle"] = {"curve": oracle, "aurc": compute_aurc(oracle)}

        # Key metrics table
        print(f"\n  {'Method':<25} {'AURC':>8} {'Cov@90%':>8} {'Cov@95%':>8} {'Acc@50%':>8}")
        print(f"  {'-'*58}")
        for method, data in results.items():
            c90 = coverage_at_accuracy(data["curve"], 0.90)
            c95 = coverage_at_accuracy(data["curve"], 0.95)
            a50 = accuracy_at_coverage(data["curve"], 0.50)
            data["coverage_at_90"] = c90
            data["coverage_at_95"] = c95
            data["accuracy_at_50"] = a50
            a50_str = f"{a50:.3f}" if a50 else "N/A"
            print(f"  {method:<25} {data['aurc']:>8.4f} {c90:>8.1%} {c95:>8.1%} {a50_str:>8}")

        # Print bootstrap CIs
        if "aurc_ci" in results["Calibrator P(correct)"]:
            ci = results["Calibrator P(correct)"]
            print(f"\n  Calibrator Bootstrap 95% CIs:")
            print(f"    AURC: [{ci['aurc_ci'][0]:.4f}, {ci['aurc_ci'][1]:.4f}]")
            if "cov90_ci" in ci:
                print(f"    Cov@90%: [{ci['cov90_ci'][0]:.1%}, {ci['cov90_ci'][1]:.1%}]")
            if "cov95_ci" in ci:
                print(f"    Cov@95%: [{ci['cov95_ci'][0]:.1%}, {ci['cov95_ci'][1]:.1%}]")

        # AUROC comparison
        labels = [s["is_correct"] for s in samples]
        preds = [s["p_correct"] for s in samples]
        if len(set(labels)) > 1:
            auroc = roc_auc_score(labels, preds)
            print(f"\n  Calibrator AUROC: {auroc:.4f}")
            if verb_samples:
                verb_auroc = roc_auc_score(
                    [s["is_correct"] for s in verb_samples],
                    [s["verbalized_confidence"] for s in verb_samples]
                )
                print(f"  Verbalized AUROC: {verb_auroc:.4f} (delta: {auroc - verb_auroc:+.4f})")

        # Plot
        plot_curves(results, f"{args.fig_dir}/uc1_selective_{target}.pdf", target_names[target])

        # Store (strip curves for JSON)
        all_results[target] = {
            method: {k: v for k, v in data.items() if k != "curve"}
            for method, data in results.items()
        }
        all_results[target]["n_samples"] = len(samples)
        all_results[target]["base_accuracy"] = float(np.mean([s["is_correct"] for s in samples]))

    out_path = f"{args.output_dir}/uc1_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
