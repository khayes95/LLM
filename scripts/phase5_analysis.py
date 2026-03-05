#!/usr/bin/env python3
"""Phase 5 analysis: reliability diagrams, confidence histograms, bootstrap CIs,
selective prediction curves, and use case summary.

All CPU-only. Reads from scored data in data/use_cases/scored_test_only_v2/.
Outputs figures to figures/paper/ and JSON results to data/use_cases/results_test_only_v2/.

Usage:
    python scripts/phase5_analysis.py
    python scripts/phase5_analysis.py --smoke_test
    python scripts/phase5_analysis.py --tasks reliability,histograms,bootstrap,selective,summary
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression

SCORED_DIR = Path("data/use_cases/scored_test_only_v2")
RESULTS_DIR = Path("data/use_cases/results_test_only_v2")
FIG_DIR = Path("figures/paper")

TARGETS = ["gpt5mini", "gpt52", "qwen35"]
TARGET_LABELS = {"gpt5mini": "GPT-5-mini", "gpt52": "GPT-5.2", "qwen35": "Qwen3.5"}

METHODS = [
    ("p_correct", "Calibrator (ours)"),
    ("verbalized_confidence", "Verbalized (raw)"),
    ("p_platt_verbalized", "Verbalized (Platt)"),
    ("p_isotonic_verbalized", "Verbalized (Isotonic)"),
    ("p_length_baseline", "Response length"),
    ("p_combined_baseline", "Combined (verb+len)"),
    ("p_zeroshot", "Zero-shot base model"),
]

def _override_paths(scored_dir, results_dir, fig_dir):
    global SCORED_DIR, RESULTS_DIR, FIG_DIR
    SCORED_DIR = scored_dir
    RESULTS_DIR = results_dir
    FIG_DIR = fig_dir


METHOD_COLORS = {
    "Calibrator (ours)": "#d62728",
    "Verbalized (raw)": "#1f77b4",
    "Verbalized (Platt)": "#2ca02c",
    "Verbalized (Isotonic)": "#ff7f0e",
    "Response length": "#9467bd",
    "Combined (verb+len)": "#8c564b",
    "Zero-shot base model": "#7f7f7f",
}


def load_scored(target):
    path = SCORED_DIR / f"{target}_scored.jsonl"
    if not path.exists():
        return []
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_all_scored():
    all_data = {}
    for t in TARGETS:
        d = load_scored(t)
        if d:
            all_data[t] = d
    return all_data


# ============================================================
# 1. RELIABILITY DIAGRAMS
# ============================================================

def plot_reliability_diagrams(all_data, n_bins=10):
    """Plot reliability diagrams (calibration curves) for all targets × key methods."""
    key_methods = [
        ("p_correct", "Calibrator (ours)"),
        ("verbalized_confidence", "Verbalized (raw)"),
        ("p_isotonic_verbalized", "Verbalized (Isotonic)"),
    ]

    targets_with_data = [t for t in TARGETS if t in all_data]
    n_targets = len(targets_with_data)
    if n_targets == 0:
        print("  No data for reliability diagrams.")
        return

    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4.5), squeeze=False)

    for col, target in enumerate(targets_with_data):
        ax = axes[0, col]
        data = all_data[target]

        for field, label in key_methods:
            valid = [d for d in data if d.get(field) is not None and d.get("is_correct") is not None]
            if len(valid) < 20:
                continue

            preds = np.array([d[field] for d in valid])
            labels = np.array([float(d["is_correct"]) for d in valid])

            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = []
            bin_accs = []
            bin_counts = []

            for i in range(n_bins):
                mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
                if i == n_bins - 1:
                    mask = (preds >= bin_edges[i]) & (preds <= bin_edges[i + 1])
                if mask.sum() > 0:
                    bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                    bin_accs.append(labels[mask].mean())
                    bin_counts.append(int(mask.sum()))

            color = METHOD_COLORS.get(label, "#333333")
            ax.plot(bin_centers, bin_accs, 'o-', color=color, label=label,
                    markersize=5, linewidth=1.5)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1, label="Perfect")
        ax.set_xlabel("Predicted P(correct)", fontsize=11)
        if col == 0:
            ax.set_ylabel("Empirical P(correct)", fontsize=11)
        ax.set_title(TARGET_LABELS.get(target, target), fontsize=12, fontweight='bold')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if col == n_targets - 1:
            ax.legend(fontsize=8, loc='lower right')

    fig.suptitle("Reliability Diagrams (Calibration)", fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "reliability_diagrams.pdf"
    fig.savefig(out, bbox_inches='tight', dpi=150)
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# 2. CONFIDENCE HISTOGRAMS
# ============================================================

def plot_confidence_histograms(all_data, n_bins=30):
    """Plot histograms of P(correct) for correct vs incorrect predictions."""
    targets_with_data = [t for t in TARGETS if t in all_data]
    n_targets = len(targets_with_data)
    if n_targets == 0:
        return

    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4), squeeze=False)

    for col, target in enumerate(targets_with_data):
        ax = axes[0, col]
        data = all_data[target]

        valid = [d for d in data if d.get("p_correct") is not None]
        correct_preds = [d["p_correct"] for d in valid if d["is_correct"]]
        incorrect_preds = [d["p_correct"] for d in valid if not d["is_correct"]]

        bins = np.linspace(0, 1, n_bins + 1)
        ax.hist(correct_preds, bins=bins, alpha=0.6, color="#2ca02c",
                label=f"Correct (n={len(correct_preds)})", density=True)
        ax.hist(incorrect_preds, bins=bins, alpha=0.6, color="#d62728",
                label=f"Incorrect (n={len(incorrect_preds)})", density=True)

        ax.set_xlabel("P(correct) from calibrator", fontsize=11)
        if col == 0:
            ax.set_ylabel("Density", fontsize=11)
        ax.set_title(TARGET_LABELS.get(target, target), fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Confidence Distributions: Correct vs Incorrect", fontsize=13,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "confidence_histograms.pdf"
    fig.savefig(out, bbox_inches='tight', dpi=150)
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# 3. BOOTSTRAP CONFIDENCE INTERVALS (with visualization)
# ============================================================

def bootstrap_auroc(y_true, y_score, n_bootstrap=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n < 10 or len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": n}

    auroc = roc_auc_score(y_true, y_score)
    aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_score[idx]))

    aurocs = np.array(aurocs)
    return {
        "auroc": float(auroc),
        "ci_low": float(np.percentile(aurocs, 2.5)),
        "ci_high": float(np.percentile(aurocs, 97.5)),
        "n": n,
    }


def _bootstrap_worker(args):
    name, y, scores, n_boot = args
    result = bootstrap_auroc(y, scores, n_bootstrap=n_boot)
    result["method"] = name
    return result


def compute_bootstrap_cis(all_data, n_bootstrap=2000):
    """Compute bootstrap CIs for all methods × targets, then plot."""
    all_results = {}
    n_workers = min(len(METHODS), int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count())))

    for target in TARGETS:
        if target not in all_data:
            continue
        data = all_data[target]

        tasks = []
        for field, label in METHODS:
            valid = [d for d in data if d.get(field) is not None and d.get("is_correct") is not None]
            if len(valid) < 20:
                continue
            y = np.array([d["is_correct"] for d in valid])
            scores = np.array([d[field] for d in valid])
            tasks.append((label, y, scores, n_bootstrap))

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_bootstrap_worker, tasks))

        results.sort(key=lambda x: x["auroc"], reverse=True)
        all_results[target] = results

        print(f"\n  {TARGET_LABELS.get(target, target)}:")
        print(f"    {'Method':<30s} {'AUROC':>8s} {'95% CI':>18s}")
        print(f"    {'—'*30} {'—'*8} {'—'*18}")
        for r in results:
            ci_str = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
            marker = " <--" if r["method"] == "Calibrator (ours)" else ""
            print(f"    {r['method']:<30s} {r['auroc']:>8.4f} {ci_str:>18s}{marker}")

    # Combined across all targets
    combined_data = []
    for t in TARGETS:
        if t in all_data:
            combined_data.extend(all_data[t])

    tasks = []
    for field, label in METHODS:
        valid = [d for d in combined_data if d.get(field) is not None and d.get("is_correct") is not None]
        if len(valid) < 20:
            continue
        y = np.array([d["is_correct"] for d in valid])
        scores = np.array([d[field] for d in valid])
        tasks.append((label, y, scores, n_bootstrap))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_bootstrap_worker, tasks))
    results.sort(key=lambda x: x["auroc"], reverse=True)
    all_results["combined"] = results

    # Save JSON
    out_json = RESULTS_DIR / "bootstrap_ci_full.json"
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_json}")

    # Plot: horizontal bar chart with CI error bars
    _plot_bootstrap_bars(all_results)

    return all_results


def _plot_bootstrap_bars(all_results):
    """Plot bootstrap CI comparison as horizontal bar chart."""
    # Use combined results for the main comparison figure
    results = all_results.get("combined", [])
    if not results:
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    names = [r["method"] for r in results]
    aurocs = [r["auroc"] for r in results]
    ci_lows = [r["auroc"] - r["ci_low"] for r in results]
    ci_highs = [r["ci_high"] - r["auroc"] for r in results]
    colors = [METHOD_COLORS.get(n, "#333333") for n in names]

    y_pos = range(len(names))
    ax.barh(y_pos, aurocs, xerr=[ci_lows, ci_highs], color=colors,
            capsize=4, edgecolor='white', linewidth=0.5, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("AUROC", fontsize=11)
    ax.set_title("Method Comparison (all targets, 95% CI)", fontsize=12, fontweight='bold')
    ax.set_xlim(0.4, 1.0)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()

    # Annotate values
    for i, r in enumerate(results):
        ax.text(r["auroc"] + ci_highs[i] + 0.005, i,
                f'{r["auroc"]:.3f}', va='center', fontsize=9)

    fig.tight_layout()
    out = FIG_DIR / "bootstrap_ci_comparison.pdf"
    fig.savefig(out, bbox_inches='tight', dpi=150)
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# 4. SELECTIVE PREDICTION CURVES
# ============================================================

def plot_selective_prediction(all_data):
    """Plot coverage vs accuracy curves for selective prediction."""
    targets_with_data = [t for t in TARGETS if t in all_data]
    n_targets = len(targets_with_data)
    if n_targets == 0:
        return

    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4.5), squeeze=False)

    key_methods = [
        ("p_correct", "Calibrator (ours)"),
        ("verbalized_confidence", "Verbalized (raw)"),
        ("p_isotonic_verbalized", "Verbalized (Isotonic)"),
    ]

    selective_results = {}

    for col, target in enumerate(targets_with_data):
        ax = axes[0, col]
        data = all_data[target]
        target_results = {}

        for field, label in key_methods:
            valid = [d for d in data if d.get(field) is not None and d.get("is_correct") is not None]
            if len(valid) < 20:
                continue

            preds = np.array([d[field] for d in valid])
            labels = np.array([float(d["is_correct"]) for d in valid])

            # Sort by descending confidence
            sorted_idx = np.argsort(-preds)
            sorted_labels = labels[sorted_idx]

            coverages = np.arange(1, len(sorted_labels) + 1) / len(sorted_labels)
            accuracies = np.cumsum(sorted_labels) / np.arange(1, len(sorted_labels) + 1)

            color = METHOD_COLORS.get(label, "#333333")
            ax.plot(coverages, accuracies, color=color, label=label, linewidth=1.5)

            # Key metrics
            base_acc = labels.mean()
            # Coverage at 90% accuracy
            cov_90 = 0.0
            for c, a in zip(coverages, accuracies):
                if a >= 0.90:
                    cov_90 = c
            # Coverage at 95% accuracy
            cov_95 = 0.0
            for c, a in zip(coverages, accuracies):
                if a >= 0.95:
                    cov_95 = c

            # AURC (area under risk-coverage curve)
            risks = 1 - accuracies
            aurc = float(np.trapezoid(risks, coverages))

            target_results[label] = {
                "coverage_at_90_acc": float(cov_90),
                "coverage_at_95_acc": float(cov_95),
                "aurc": aurc,
                "base_accuracy": float(base_acc),
                "n": len(valid),
            }

        # Random baseline
        base_acc = np.mean([d["is_correct"] for d in data if d.get("is_correct") is not None])
        ax.axhline(y=base_acc, color='gray', linestyle=':', alpha=0.5, label=f'Random ({base_acc:.2f})')
        ax.axhline(y=0.90, color='black', linestyle='--', alpha=0.3)

        ax.set_xlabel("Coverage", fontsize=11)
        if col == 0:
            ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(TARGET_LABELS.get(target, target), fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.02)
        ax.set_ylim(max(0, base_acc - 0.15), 1.02)
        ax.grid(True, alpha=0.3)
        if col == n_targets - 1:
            ax.legend(fontsize=8, loc='lower left')

        selective_results[target] = target_results

    fig.suptitle("Selective Prediction: Coverage vs Accuracy", fontsize=13,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "selective_prediction.pdf"
    fig.savefig(out, bbox_inches='tight', dpi=150)
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # Save metrics
    out_json = RESULTS_DIR / "selective_prediction_metrics.json"
    with open(out_json, 'w') as f:
        json.dump(selective_results, f, indent=2)
    print(f"  Saved: {out_json}")

    # Print summary
    for target, methods in selective_results.items():
        print(f"\n  {TARGET_LABELS.get(target, target)}:")
        for method, metrics in methods.items():
            print(f"    {method:<30s} Cov@90%={metrics['coverage_at_90_acc']:.1%}  "
                  f"Cov@95%={metrics['coverage_at_95_acc']:.1%}  AURC={metrics['aurc']:.3f}")


# ============================================================
# 5. USE CASE SUMMARY
# ============================================================

def generate_use_case_summary():
    """Generate a summary table/figure of all use case results."""
    uc_files = sorted(RESULTS_DIR.glob("uc*_results.json"))
    if not uc_files:
        print("  No use case results found.")
        return

    uc_data = {}
    for f in uc_files:
        name = f.stem.replace("_results", "")
        with open(f) as fh:
            uc_data[name] = json.load(fh)

    # Build summary table
    summary_rows = []

    uc_definitions = {
        "uc1": ("Selective Prediction", "AURC↓"),
        "uc2": ("Model Routing", "Cost Savings"),
        "uc3": ("Hallucination Detection", "Best F1"),
        "uc4": ("Difficulty Estimation", "Rank Corr"),
        "uc5": ("UQ Reward Model", "Pairwise Acc"),
        "uc8": ("Deployment Monitoring", "Alert F1"),
        "uc9": ("Annotation Prioritization", "AUEDR"),
        "uc_a": ("DPO Reward Signal", "Pair Acc"),
        "uc_b": ("Best-of-N Selection", "Accuracy"),
        "uc_c": ("Data Curation", "Acc@50%"),
        "uc_d": ("Agent Step Scoring", "Partial Corr"),
    }

    print("\n  Use Case Summary:")
    print(f"    {'UC':<8s} {'Name':<28s} {'Metric':<14s} {'GPT-5-mini':>12s} {'GPT-5.2':>12s} {'Qwen3.5':>12s}")
    print(f"    {'—'*8} {'—'*28} {'—'*14} {'—'*12} {'—'*12} {'—'*12}")

    for uc_id, (name, metric) in uc_definitions.items():
        if uc_id not in uc_data:
            continue
        d = uc_data[uc_id]
        vals = {}
        for target_key in ["gpt5mini", "gpt52", "qwen35", "GPT-5-mini", "GPT-5.2", "Qwen3.5"]:
            if target_key in d:
                v = d[target_key]
                if isinstance(v, dict):
                    # Try to extract the primary metric
                    for k in ["aurc", "cost_savings", "best_f1", "rank_correlation",
                              "pairwise_accuracy", "alert_f1", "auedr", "pair_accuracy",
                              "accuracy", "acc_at_50", "partial_correlation"]:
                        if k in v:
                            vals[target_key] = v[k]
                            break
                    else:
                        vals[target_key] = str(v)[:10]
                else:
                    vals[target_key] = v

        def fmt(v):
            if isinstance(v, float):
                return f"{v:.3f}"
            return str(v)[:10] if v else "—"

        g5m = fmt(vals.get("gpt5mini", vals.get("GPT-5-mini")))
        g52 = fmt(vals.get("gpt52", vals.get("GPT-5.2")))
        q35 = fmt(vals.get("qwen35", vals.get("Qwen3.5")))

        print(f"    {uc_id:<8s} {name:<28s} {metric:<14s} {g5m:>12s} {g52:>12s} {q35:>12s}")
        summary_rows.append({
            "uc": uc_id, "name": name, "metric": metric,
            "gpt5mini": vals.get("gpt5mini", vals.get("GPT-5-mini")),
            "gpt52": vals.get("gpt52", vals.get("GPT-5.2")),
            "qwen35": vals.get("qwen35", vals.get("Qwen3.5")),
        })

    # Save summary
    out_json = RESULTS_DIR / "use_case_summary.json"
    with open(out_json, 'w') as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\n  Saved: {out_json}")


# ============================================================
# 6. PER-BENCHMARK BREAKDOWN
# ============================================================

def plot_per_benchmark_breakdown(all_data):
    """Plot per-benchmark AUROC for the calibrator across all targets."""
    # Collect per-benchmark AUROCs
    bench_results = defaultdict(lambda: {})

    for target in TARGETS:
        if target not in all_data:
            continue
        data = all_data[target]

        by_bench = defaultdict(list)
        for d in data:
            if d.get("p_correct") is not None and d.get("is_correct") is not None:
                by_bench[d["benchmark"]].append(d)

        for bench, samples in by_bench.items():
            if len(samples) < 10:
                continue
            y = np.array([d["is_correct"] for d in samples])
            preds = np.array([d["p_correct"] for d in samples])
            if len(np.unique(y)) < 2:
                continue
            auroc = roc_auc_score(y, preds)
            bench_results[bench][target] = {"auroc": auroc, "n": len(samples)}

    if not bench_results:
        print("  No per-benchmark data.")
        return

    # Sort benchmarks by average AUROC
    bench_avg = {b: np.mean([v["auroc"] for v in d.values()]) for b, d in bench_results.items()}
    sorted_benchmarks = sorted(bench_avg, key=bench_avg.get, reverse=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_benchmarks) * 0.35)))

    bar_width = 0.25
    target_colors = {"gpt5mini": "#1f77b4", "gpt52": "#ff7f0e", "qwen35": "#2ca02c"}
    targets_present = [t for t in TARGETS if t in all_data]

    for i, target in enumerate(targets_present):
        y_pos = np.arange(len(sorted_benchmarks)) + i * bar_width
        vals = [bench_results[b].get(target, {}).get("auroc", 0) for b in sorted_benchmarks]
        ax.barh(y_pos, vals, height=bar_width, color=target_colors.get(target, "#999"),
                label=TARGET_LABELS.get(target, target), edgecolor='white', linewidth=0.5)

    ax.set_yticks(np.arange(len(sorted_benchmarks)) + bar_width * (len(targets_present) - 1) / 2)
    ax.set_yticklabels(sorted_benchmarks, fontsize=9)
    ax.set_xlabel("AUROC", fontsize=11)
    ax.set_title("Per-Benchmark Calibrator AUROC", fontsize=12, fontweight='bold')
    ax.set_xlim(0.4, 1.0)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()

    fig.tight_layout()
    out = FIG_DIR / "per_benchmark_breakdown.pdf"
    fig.savefig(out, bbox_inches='tight', dpi=150)
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # Save data
    out_json = RESULTS_DIR / "per_benchmark_auroc.json"
    serializable = {b: {t: d.get(t) for t in TARGETS} for b, d in bench_results.items()}
    with open(out_json, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {out_json}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5 analysis for UQ paper")
    parser.add_argument("--scored_dir", type=str, default="data/use_cases/scored_test_only_v2")
    parser.add_argument("--output_dir", type=str, default="data/use_cases/results_test_only_v2")
    parser.add_argument("--fig_dir", type=str, default="figures/paper")
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated: reliability,histograms,bootstrap,selective,summary,benchmark")
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    # Override module-level defaults with CLI args
    _override_paths(Path(args.scored_dir), Path(args.output_dir), Path(args.fig_dir))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks.split(",") if args.tasks != "all" else [
        "reliability", "histograms", "bootstrap", "selective", "summary", "benchmark"
    ]

    print("=" * 60)
    print("PHASE 5 ANALYSIS")
    print("=" * 60)
    print(f"Scored data: {SCORED_DIR}")
    print(f"Output:      {RESULTS_DIR}")
    print(f"Figures:     {FIG_DIR}")
    print(f"Tasks:       {', '.join(tasks)}")

    all_data = load_all_scored()
    if args.smoke_test:
        all_data = {t: d[:200] for t, d in all_data.items()}

    total = sum(len(d) for d in all_data.values())
    print(f"Loaded {total} scored samples across {len(all_data)} targets\n")

    if "reliability" in tasks:
        print("[1/6] Reliability diagrams...")
        plot_reliability_diagrams(all_data)

    if "histograms" in tasks:
        print("[2/6] Confidence histograms...")
        plot_confidence_histograms(all_data)

    if "bootstrap" in tasks:
        n_boot = 100 if args.smoke_test else args.n_bootstrap
        print(f"[3/6] Bootstrap CIs (n={n_boot})...")
        compute_bootstrap_cis(all_data, n_bootstrap=n_boot)

    if "selective" in tasks:
        print("[4/6] Selective prediction curves...")
        plot_selective_prediction(all_data)

    if "summary" in tasks:
        print("[5/6] Use case summary...")
        generate_use_case_summary()

    if "benchmark" in tasks:
        print("[6/6] Per-benchmark breakdown...")
        plot_per_benchmark_breakdown(all_data)

    print("\n" + "=" * 60)
    print("PHASE 5 ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
