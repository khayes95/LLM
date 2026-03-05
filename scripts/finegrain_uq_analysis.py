#!/usr/bin/env python3
"""Analysis script for FineGRAIN x UQ experiments.

Combines results from:
  - Part 1: Prompt variant ablation (human-labeled, data/finegrain_uq/ablation_*)
  - Part 2: New model scoring (judge-labeled, data/finegrain_uq/newmodels/)
  - Existing: Combined variant results (data/finegrain_uq/results.json)

Produces:
  - Prompt variant comparison table
  - Selective prediction curves
  - Per-failure-mode heatmap data
  - Model ranking correlation
  - Summary figures

Usage:
    python scripts/finegrain_uq_analysis.py
    python scripts/finegrain_uq_analysis.py --output_dir data/finegrain_uq/analysis
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# DATA LOADING
# ============================================================

def load_scored_jsonl(path):
    """Load a scored JSONL file."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def load_results_json(path):
    """Load a results.json file."""
    with open(path) as f:
        return json.load(f)


# ============================================================
# ANALYSIS 1: PROMPT VARIANT COMPARISON
# ============================================================

def analyze_prompt_variants(base_dir):
    """Compare AUROC across prompt variants."""
    print("=" * 70)
    print("ANALYSIS 1: Prompt Variant Comparison")
    print("=" * 70)

    variants = {}

    # Check for existing combined results
    combined_path = base_dir / "results.json"
    if combined_path.exists():
        r = load_results_json(combined_path)
        variants["combined"] = {
            "auroc": r["overall"]["overall_auroc"],
            "accuracy": r["overall"]["overall_accuracy"],
            "best_f1": r["overall"]["overall_best_f1"],
            "n_samples": r["overall"]["overall_n_samples"],
        }
        if "bootstrap_ci" in r:
            variants["combined"]["ci_low"] = r["bootstrap_ci"]["auroc_ci_low"]
            variants["combined"]["ci_high"] = r["bootstrap_ci"]["auroc_ci_high"]

    # Check ablation directories
    for variant in ["baseline", "finegrain_direct", "finegrain_qa"]:
        ablation_path = base_dir / f"ablation_{variant}" / "results.json"
        if ablation_path.exists():
            r = load_results_json(ablation_path)
            variants[variant] = {
                "auroc": r["overall"]["overall_auroc"],
                "accuracy": r["overall"]["overall_accuracy"],
                "best_f1": r["overall"]["overall_best_f1"],
                "n_samples": r["overall"]["overall_n_samples"],
            }
            if "bootstrap_ci" in r:
                variants[variant]["ci_low"] = r["bootstrap_ci"]["auroc_ci_low"]
                variants[variant]["ci_high"] = r["bootstrap_ci"]["auroc_ci_high"]

    if not variants:
        print("  No prompt variant results found.")
        return {}

    print(f"\n{'Variant':<25} {'AUROC':<10} {'95% CI':<20} {'Accuracy':<10} {'Best F1':<10}")
    print("-" * 75)
    for name, v in sorted(variants.items(), key=lambda x: -x[1]["auroc"]):
        ci = f"[{v.get('ci_low', '?'):.3f}, {v.get('ci_high', '?'):.3f}]" if 'ci_low' in v else "N/A"
        print(f"{name:<25} {v['auroc']:<10.4f} {ci:<20} {v['accuracy']:<10.4f} {v['best_f1']:<10.4f}")

    best = max(variants.items(), key=lambda x: x[1]["auroc"])
    print(f"\nBest variant: {best[0]} (AUROC={best[1]['auroc']:.4f})")

    return variants


# ============================================================
# ANALYSIS 2: SELECTIVE PREDICTION CURVES
# ============================================================

def analyze_selective_prediction(base_dir, output_dir):
    """Generate selective prediction curves from human-labeled data."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Selective Prediction Curves")
    print("=" * 70)

    # Find the best variant's scored samples
    scored_path = base_dir / "scored_samples.jsonl"
    if not scored_path.exists():
        print("  No scored_samples.jsonl found, skipping.")
        return {}

    samples = load_scored_jsonl(scored_path)
    labels = np.array([s["human_label"] for s in samples])
    scores = np.array([s["p_compliant"] for s in samples])
    failure_scores = 1.0 - scores

    # Find optimal threshold
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (failure_scores >= t).astype(int)
        if labels.sum() > 0:
            t_f1 = f1_score(labels, preds, zero_division=0)
            if t_f1 > best_f1:
                best_f1, best_thresh = t_f1, t

    # Compute selective prediction curve
    confidence = np.abs(failure_scores - 0.5)
    sorted_idx = np.argsort(-confidence)

    coverages = np.arange(0.05, 1.01, 0.05)
    curve_data = []

    print(f"\n{'Coverage':<12} {'Accuracy':<12} {'N':<8} {'Baseline':<12}")
    print("-" * 44)
    for cov in coverages:
        n = int(len(labels) * cov)
        if n == 0:
            continue
        sel_idx = sorted_idx[:n]
        sel_preds = (failure_scores[sel_idx] >= best_thresh).astype(int)
        sel_labels = labels[sel_idx]
        acc = accuracy_score(sel_labels, sel_preds)

        # Random baseline (expected accuracy = max(failure_rate, 1-failure_rate))
        baseline = max(labels.mean(), 1 - labels.mean())

        curve_data.append({
            "coverage": float(cov),
            "accuracy": float(acc),
            "n_samples": n,
            "baseline": float(baseline),
        })

        if cov in [0.10, 0.25, 0.50, 0.75, 1.0]:
            print(f"{cov:<12.0%} {acc:<12.4f} {n:<8} {baseline:<12.4f}")

    # Plot if matplotlib available
    if HAS_MATPLOTLIB and curve_data:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        covs = [d["coverage"] for d in curve_data]
        accs = [d["accuracy"] for d in curve_data]
        ax.plot(covs, accs, "o-", label="UQ Model", linewidth=2)
        ax.axhline(y=0.674, color="r", linestyle="--", label="FineGRAIN pipeline (67.4%)")
        ax.axhline(y=curve_data[-1]["baseline"], color="gray", linestyle=":",
                   label=f"Random baseline ({curve_data[-1]['baseline']:.1%})")
        ax.set_xlabel("Coverage (fraction of samples evaluated)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Selective Prediction: UQ Model on FineGRAIN")
        ax.legend()
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = output_dir / "fig_selective_prediction.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"\n  Saved figure: {fig_path}")

    return {"curve": curve_data, "threshold": best_thresh}


# ============================================================
# ANALYSIS 3: PER-FAILURE-MODE BREAKDOWN
# ============================================================

def analyze_failure_modes(base_dir, output_dir):
    """Analyze per-failure-mode performance."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Per-Failure-Mode Breakdown")
    print("=" * 70)

    # Load human-labeled results
    results_path = base_dir / "results.json"
    if not results_path.exists():
        print("  No results.json found, skipping.")
        return {}

    results = load_results_json(results_path)
    fm_data = results.get("per_failure_mode", {})

    if not fm_data:
        print("  No per-failure-mode data found.")
        return {}

    # Sort by AUROC
    sorted_fms = sorted(fm_data.items(),
                        key=lambda x: x[1].get("auroc", 0) if not np.isnan(x[1].get("auroc", float("nan"))) else -1,
                        reverse=True)

    print(f"\n{'Failure Mode':<50} {'AUROC':<8} {'N':<6} {'Fail%':<8}")
    print("-" * 72)
    for fm, metrics in sorted_fms:
        auroc = metrics.get("auroc", float("nan"))
        auroc_str = f"{auroc:.3f}" if not np.isnan(auroc) else "N/A"
        n = metrics.get("n_samples", 0)
        fail_rate = metrics.get("actual_failure_rate", 0)
        print(f"{fm:<50} {auroc_str:<8} {n:<6} {fail_rate:<8.1%}")

    # Categorize
    strong = [(fm, m) for fm, m in sorted_fms
              if not np.isnan(m.get("auroc", float("nan"))) and m["auroc"] >= 0.70]
    weak = [(fm, m) for fm, m in sorted_fms
            if not np.isnan(m.get("auroc", float("nan"))) and m["auroc"] < 0.60]

    print(f"\nStrong failure modes (AUROC >= 0.70): {len(strong)}")
    for fm, m in strong:
        print(f"  {fm}: {m['auroc']:.3f}")

    print(f"\nWeak failure modes (AUROC < 0.60): {len(weak)}")
    for fm, m in weak:
        print(f"  {fm}: {m['auroc']:.3f}")

    # Plot heatmap if matplotlib available
    if HAS_MATPLOTLIB and sorted_fms:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fms = [fm for fm, _ in sorted_fms]
        aurocs = [m.get("auroc", 0) for _, m in sorted_fms]
        # Replace NaN with 0.5 for display
        aurocs = [a if not np.isnan(a) else 0.5 for a in aurocs]

        colors = ["#d73027" if a < 0.6 else "#fee08b" if a < 0.7 else "#1a9850" for a in aurocs]
        ax.barh(range(len(fms)), aurocs, color=colors)
        ax.set_yticks(range(len(fms)))
        ax.set_yticklabels(fms, fontsize=8)
        ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5, label="Random")
        ax.axvline(x=0.7, color="black", linestyle="--", alpha=0.5, label="Target")
        ax.set_xlabel("AUROC")
        ax.set_title("UQ Model AUROC by Failure Mode")
        ax.legend()
        ax.invert_yaxis()
        fig.tight_layout()
        fig_path = output_dir / "fig_failure_mode_auroc.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"\n  Saved figure: {fig_path}")

    return {"per_failure_mode": dict(sorted_fms)}


# ============================================================
# ANALYSIS 4: MODEL RANKING (from newmodels results)
# ============================================================

def analyze_model_ranking(base_dir, output_dir):
    """Compare UQ model ranking with LLM judge ranking."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: T2I Model Ranking Correlation")
    print("=" * 70)

    newmodels_path = base_dir / "newmodels" / "results.json"
    if not newmodels_path.exists():
        print("  No newmodels results found, skipping.")
        return {}

    results = load_results_json(newmodels_path)

    model_data = {}
    for model_name, metrics in results.items():
        if model_name.startswith("_"):
            continue
        model_data[model_name] = {
            "uq_mean_failure": 1 - metrics.get("mean_p_compliant", 0.5),
            "judge_failure_rate": metrics.get("failure_rate_judge", 0.5),
            "auroc": metrics.get("auroc", float("nan")),
            "kappa": metrics.get("cohens_kappa", float("nan")),
            "n_samples": metrics.get("n_samples", 0),
        }

    if len(model_data) < 3:
        print("  Need at least 3 models for ranking, skipping.")
        return {}

    # Sort by judge failure rate
    sorted_models = sorted(model_data.items(), key=lambda x: x[1]["judge_failure_rate"])

    print(f"\n{'Model':<25} {'Judge Fail%':<14} {'UQ Fail%':<12} {'AUROC':<8} {'Kappa':<8} {'N':<6}")
    print("-" * 73)
    for m, d in sorted_models:
        auroc_str = f"{d['auroc']:.3f}" if not np.isnan(d["auroc"]) else "N/A"
        kappa_str = f"{d['kappa']:.3f}" if not np.isnan(d["kappa"]) else "N/A"
        print(f"{m:<25} {d['judge_failure_rate']:<14.1%} {d['uq_mean_failure']:<12.1%} "
              f"{auroc_str:<8} {kappa_str:<8} {d['n_samples']:<6}")

    # Rank correlation
    ordered = sorted(model_data.keys())
    uq_vals = [model_data[m]["uq_mean_failure"] for m in ordered]
    judge_vals = [model_data[m]["judge_failure_rate"] for m in ordered]
    rho, p_rho = spearmanr(uq_vals, judge_vals)
    tau, p_tau = kendalltau(uq_vals, judge_vals)

    print(f"\nRanking correlation:")
    print(f"  Spearman rho: {rho:.4f} (p={p_rho:.4f})")
    print(f"  Kendall tau:  {tau:.4f} (p={p_tau:.4f})")

    # Plot scatter if matplotlib available
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for m in ordered:
            ax.scatter(model_data[m]["judge_failure_rate"],
                       model_data[m]["uq_mean_failure"], s=60)
            ax.annotate(m, (model_data[m]["judge_failure_rate"],
                            model_data[m]["uq_mean_failure"]),
                        fontsize=7, ha="left", va="bottom")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect agreement")
        ax.set_xlabel("LLM Judge Failure Rate")
        ax.set_ylabel("UQ Model Mean P(failure)")
        ax.set_title(f"T2I Model Ranking: UQ vs LLM Judge (rho={rho:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = output_dir / "fig_model_ranking.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"\n  Saved figure: {fig_path}")

    return {
        "spearman_rho": rho, "spearman_p": p_rho,
        "kendall_tau": tau, "kendall_p": p_tau,
        "per_model": model_data,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FineGRAIN x UQ Analysis")
    parser.add_argument("--base_dir", default="data/finegrain_uq",
                        help="Base directory with all FineGRAIN UQ results")
    parser.add_argument("--output_dir", default="data/finegrain_uq/analysis",
                        help="Output directory for analysis results and figures")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Analysis 1: Prompt variants
    variants = analyze_prompt_variants(base_dir)
    if variants:
        all_results["prompt_variants"] = variants

    # Analysis 2: Selective prediction
    selective = analyze_selective_prediction(base_dir, output_dir)
    if selective:
        all_results["selective_prediction"] = selective

    # Analysis 3: Failure modes
    failure_modes = analyze_failure_modes(base_dir, output_dir)
    if failure_modes:
        all_results["failure_modes"] = failure_modes

    # Analysis 4: Model ranking
    ranking = analyze_model_ranking(base_dir, output_dir)
    if ranking:
        all_results["model_ranking"] = ranking

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if variants:
        best_var = max(variants.items(), key=lambda x: x[1]["auroc"])
        print(f"Best prompt variant: {best_var[0]} (AUROC={best_var[1]['auroc']:.4f})")

    if selective and selective.get("curve"):
        # Find accuracy at 50% coverage
        for pt in selective["curve"]:
            if abs(pt["coverage"] - 0.50) < 0.03:
                print(f"Selective acc @ 50% coverage: {pt['accuracy']:.4f}")
                break

    if ranking:
        print(f"Model ranking Spearman rho: {ranking['spearman_rho']:.4f}")

    # Save
    results_path = output_dir / "analysis_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
