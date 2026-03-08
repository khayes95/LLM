#!/usr/bin/env python3
"""Generate paper-ready figure for production deployment use cases (UC-E/F/G).

Creates a 3-panel figure showing:
  (a) UC-E: Error detection precision-recall (calibrator vs baselines)
  (b) UC-F: Coverage vs safety Pareto frontier
  (c) UC-G: Human review efficiency (UQ-guided vs random)

Uses academic styling consistent with existing paper figures.

Usage:
    python scripts/generate_production_uc_figure.py
    python scripts/generate_production_uc_figure.py --output_dir figures/paper
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Academic style (matching generate_paper_figures.py)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.edgecolor'] = '0.8'

COLORS = {
    'blue': '#4472C4',
    'orange': '#ED7D31',
    'gray': '#7F7F7F',
    'green': '#70AD47',
    'red': '#C55A5A',
    'purple': '#7030A0',
    'teal': '#2E8B8B',
    'light_blue': '#9DC3E6',
    'light_red': '#F4B4B4',
}

MODEL_NAMES = {
    "gpt5mini": "GPT-5-mini",
    "gpt52": "GPT-5.2",
    "qwen35": "Qwen3.5",
}

# Use GPT-5.2 as the representative model (cross-model, best results)
REPRESENTATIVE = "gpt52"
RESULTS_DIR = Path("data/use_cases/results_test_only_v2")


def load_results(uc_name):
    path = RESULTS_DIR / f"uc_{uc_name}_results.json"
    with open(path) as f:
        return json.load(f)


def panel_a_error_detection(ax):
    """UC-E: Precision-Recall curves for error detection — calibrator vs baselines."""
    data = load_results("e")

    # Average across all 3 models for robustness
    methods_to_plot = {
        "calibrator": ("Calibrator (ours)", COLORS['blue'], 'o', '-', 2.5),
        "verbalized": ("Verbalized conf.", COLORS['orange'], 's', '--', 1.8),
        "combined_baseline": ("Combined baseline", COLORS['gray'], '^', ':', 1.5),
        "length_baseline": ("Length baseline", COLORS['light_blue'], 'D', ':', 1.2),
    }

    for method_key, (label, color, marker, ls, lw) in methods_to_plot.items():
        # Gather trigger analysis across models
        all_precisions = {}
        all_recalls = {}

        for model in ["gpt5mini", "gpt52", "qwen35"]:
            if model not in data:
                continue
            # Use trigger_analysis data (precision/recall at each threshold)
            ta = data[model]["trigger_analysis"]
            for r in ta:
                t = r["threshold"]
                if t not in all_precisions:
                    all_precisions[t] = []
                    all_recalls[t] = []

            # For the method comparison, get AUPRC
            if method_key == "calibrator":
                for r in ta:
                    t = r["threshold"]
                    all_precisions[t].append(r["precision"])
                    all_recalls[t].append(r["recall"])

        if method_key == "calibrator":
            # Plot calibrator precision/recall from trigger analysis (averaged)
            thresholds = sorted(all_precisions.keys())
            precs = [np.mean(all_precisions[t]) for t in thresholds]
            recs = [np.mean(all_recalls[t]) for t in thresholds]
            f1s = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precs, recs)]

            ax.plot(recs, precs, marker=marker, linestyle=ls, color=color,
                    linewidth=lw, markersize=5, label=label, zorder=5)
        else:
            # For baselines, compute from per-model trigger analysis using that score
            # We need to re-derive from the scored data or use the summary AUPRC
            # Use the AUPRC values to annotate
            pass

    # Plot F1 contours
    for f1_val in [0.6, 0.7, 0.8, 0.9]:
        rec_range = np.linspace(0.01, 1.0, 100)
        prec_from_f1 = f1_val * rec_range / (2 * rec_range - f1_val)
        valid = (prec_from_f1 > 0) & (prec_from_f1 <= 1)
        ax.plot(rec_range[valid], prec_from_f1[valid], ':', color='#CCCCCC',
                linewidth=0.7, alpha=0.6)
        # Label
        idx = np.argmin(np.abs(rec_range[valid] - 0.95))
        if idx < len(rec_range[valid]):
            ax.text(rec_range[valid][idx], prec_from_f1[valid][idx] + 0.01,
                    f'F1={f1_val}', fontsize=7, color='#999999', ha='center')

    # Add AUPRC comparison as inset text box
    auprc_text = "AUPRC (avg):\n"
    for model in ["gpt5mini", "gpt52", "qwen35"]:
        if model not in data:
            continue
        mc = data[model]["method_comparison"]
        cal_val = mc.get("calibrator", {}).get("auprc_error_detection", 0)
        verb_val = mc.get("verbalized", {}).get("auprc_error_detection", 0)

    # Average AUPRC across models
    cal_auprc_avg = np.mean([
        data[m]["method_comparison"]["calibrator"]["auprc_error_detection"]
        for m in data if "method_comparison" in data[m]
    ])
    verb_auprc_avg = np.mean([
        data[m]["method_comparison"]["verbalized"]["auprc_error_detection"]
        for m in data if "method_comparison" in data[m]
    ])
    comb_auprc_avg = np.mean([
        data[m]["method_comparison"]["combined_baseline"]["auprc_error_detection"]
        for m in data if "method_comparison" in data[m]
    ])

    text = (f"AUPRC:\n"
            f"  Cal: {cal_auprc_avg:.3f}\n"
            f"  Verb: {verb_auprc_avg:.3f}\n"
            f"  Comb: {comb_auprc_avg:.3f}")
    ax.text(0.02, 0.35, text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.9))

    ax.set_xlabel("Recall (fraction of errors caught)", fontsize=11)
    ax.set_ylabel("Precision (flagged are truly wrong)", fontsize=11)
    ax.set_title("(a) Error Detection for Clarification", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.55, 1.02)


def panel_b_coverage_safety(ax):
    """UC-F: Coverage vs accuracy Pareto frontier — calibrator vs baselines."""
    data = load_results("f")

    methods_to_plot = {
        "calibrator": ("Calibrator (ours)", COLORS['blue'], 'o', '-', 2.5),
        "verbalized": ("Verbalized conf.", COLORS['orange'], 's', '--', 1.8),
        "combined_baseline": ("Combined baseline", COLORS['gray'], '^', ':', 1.5),
        "length_baseline": ("Length baseline", COLORS['light_blue'], 'D', ':', 1.2),
    }

    for method_key, (label, color, marker, ls, lw) in methods_to_plot.items():
        # Average pareto points across models
        all_coverages = {}

        for model in ["gpt5mini", "gpt52", "qwen35"]:
            if model not in data:
                continue
            mc = data[model]["method_comparison"]
            if method_key not in mc:
                continue
            points = mc[method_key].get("pareto_points", [])
            for p in points:
                t = p["threshold"]
                if t not in all_coverages:
                    all_coverages[t] = {"cov": [], "acc": []}
                all_coverages[t]["cov"].append(p["coverage"])
                all_coverages[t]["acc"].append(p["exec_accuracy"])

        if not all_coverages:
            continue

        thresholds = sorted(all_coverages.keys())
        covs = [np.mean(all_coverages[t]["cov"]) for t in thresholds]
        accs = [np.mean(all_coverages[t]["acc"]) for t in thresholds]

        ax.plot(covs, accs, marker=marker, linestyle=ls, color=color,
                linewidth=lw, markersize=5, label=label, zorder=4 if method_key == "calibrator" else 3)

    # Reference lines
    avg_base_acc = np.mean([data[m]["base_accuracy"] for m in data])
    ax.axhline(y=avg_base_acc, color='black', linestyle=':', alpha=0.4,
               label=f'No gating ({avg_base_acc:.2f})')
    ax.axhline(y=0.95, color=COLORS['red'], linestyle=':', alpha=0.5,
               label='95% accuracy target')

    # Shade the "safe zone"
    ax.fill_between([0, 1.1], 0.95, 1.02, alpha=0.05, color=COLORS['green'])

    # Annotate calibrator's coverage at 95%
    avg_cov_95 = np.mean([
        data[m]["method_comparison"]["calibrator"]["coverage_at_95pct_accuracy"]
        for m in data
    ])
    ax.annotate(f'{avg_cov_95:.0%} coverage\nat 95% acc',
                xy=(avg_cov_95, 0.95), xytext=(avg_cov_95 - 0.15, 0.88),
                fontsize=9, color=COLORS['blue'],
                arrowprops=dict(arrowstyle='->', color=COLORS['blue'], lw=1.5),
                fontweight='bold')

    ax.set_xlabel("Coverage (fraction auto-executed)", fontsize=11)
    ax.set_ylabel("Accuracy of auto-executed actions", fontsize=11)
    ax.set_title("(b) Confidence-Gated Actions", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(max(0, avg_base_acc - 0.15), 1.02)


def panel_c_review_efficiency(ax):
    """UC-G: Human review efficiency — UQ-guided vs random."""
    data = load_results("g")

    # Average review efficiency across models
    uq_points = {}  # pct_reviewed -> pct_errors_caught

    for model in ["gpt5mini", "gpt52", "qwen35"]:
        if model not in data:
            continue
        re = data[model]["review_efficiency"]
        for r in re:
            pct = round(r["pct_reviewed"], 3)
            if pct not in uq_points:
                uq_points[pct] = []
            uq_points[pct].append(r["errors_caught_pct"])

    pcts = sorted(uq_points.keys())
    avg_caught = [np.mean(uq_points[p]) for p in pcts]

    # UQ-guided curve
    ax.plot(pcts, avg_caught, 'o-', color=COLORS['blue'], linewidth=2.5,
            markersize=4, label='UQ-guided review', zorder=5)

    # Random baseline (diagonal)
    ax.plot([0, 1], [0, 1], ':', color=COLORS['gray'], linewidth=2,
            label='Random review')

    # Shade the efficiency gap
    random_at_pcts = pcts  # random = diagonal
    ax.fill_between(pcts, random_at_pcts, avg_caught,
                    alpha=0.12, color=COLORS['blue'])

    # Annotate efficiency at 20%
    for p, c in zip(pcts, avg_caught):
        if abs(p - 0.20) < 0.02:
            efficiency = c / p if p > 0 else 0
            ax.annotate(f'{c:.0%} errors caught\nat {p:.0%} review\n({efficiency:.1f}x random)',
                        xy=(p, c), xytext=(p + 0.15, c - 0.08),
                        fontsize=9, color=COLORS['blue'],
                        arrowprops=dict(arrowstyle='->', color=COLORS['blue'], lw=1.5),
                        fontweight='bold')
            break

    # Annotate the 95% accuracy workload reduction
    avg_workload = np.mean([
        d["workload_reduction"].get("target_0.95", {}).get("workload_reduction", 0)
        for d in data.values()
        if "workload_reduction" in d
    ])
    ax.text(0.55, 0.15, f'Avg {avg_workload:.0%} less review\nto reach 95% accuracy',
            transform=ax.transAxes, fontsize=10, color=COLORS['blue'],
            fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=COLORS['blue'], alpha=0.9))

    # Three-tier summary in corner
    avg_green_vol = np.mean([
        data[m]["three_tier"]["tiers"]["green"]["pct_volume"]
        for m in data
    ])
    avg_green_acc = np.mean([
        data[m]["three_tier"]["tiers"]["green"]["accuracy"]
        for m in data
        if data[m]["three_tier"]["tiers"]["green"]["accuracy"] is not None
    ])
    avg_red_err_share = np.mean([
        data[m]["three_tier"]["tiers"]["red"]["error_share"]
        for m in data
    ])

    tier_text = (f"Three-tier routing:\n"
                 f"  Green: {avg_green_vol:.0%} auto-sent, {avg_green_acc:.1%} acc\n"
                 f"  Red: captures {avg_red_err_share:.0%} of errors")
    ax.text(0.98, 0.55, tier_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', ha='right', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.9))

    ax.set_xlabel("Fraction of queries reviewed by human", fontsize=11)
    ax.set_ylabel("Fraction of errors caught", fontsize=11)
    ax.set_title("(c) Human Review Efficiency", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figure for production deployment use cases")
    parser.add_argument("--output_dir", default="figures/paper",
                        help="Output directory for figures")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Main 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    panel_a_error_detection(axes[0])
    panel_b_coverage_safety(axes[1])
    panel_c_review_efficiency(axes[2])

    fig.suptitle("Production Deployment Use Cases: UQ-Enabled Workflows",
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        out_path = f"{args.output_dir}/fig_production_use_cases.{ext}"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {out_path}")
    plt.close()

    # --- Per-model supplementary figure ---
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    data_e = load_results("e")
    data_f = load_results("f")
    data_g = load_results("g")

    for col, model in enumerate(["gpt5mini", "gpt52", "qwen35"]):
        model_label = MODEL_NAMES[model]

        # Row 0: UC-E precision/recall/F1
        ax = axes[0, col]
        if model in data_e:
            ta = data_e[model]["trigger_analysis"]
            thresholds = [r["threshold"] for r in ta]
            ax.plot(thresholds, [r["precision"] for r in ta], 'o-',
                    color=COLORS['blue'], linewidth=2, label='Precision')
            ax.plot(thresholds, [r["recall"] for r in ta], 's-',
                    color=COLORS['orange'], linewidth=2, label='Recall')
            ax.plot(thresholds, [r["f1"] for r in ta], '^-',
                    color=COLORS['red'], linewidth=2.5, label='F1')
            best_f1_idx = int(np.argmax([r["f1"] for r in ta]))
            ax.axvline(x=thresholds[best_f1_idx], color=COLORS['red'],
                       alpha=0.3, linestyle='--')
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"{model_label}" if col == 1 else model_label, fontsize=12)
        if col == 0:
            ax.set_ylabel("UC-E: Error Detection\nScore", fontsize=11)
        ax.legend(fontsize=8)
        ax.set_ylim(-0.02, 1.05)

        # Row 1: UC-F coverage vs safety
        ax = axes[1, col]
        if model in data_f:
            ga = data_f[model]["gated_analysis"]
            covs = [r["coverage"] for r in ga if r["n_executed"] > 0]
            accs = [r["exec_accuracy"] for r in ga if r["n_executed"] > 0]
            ax.plot(covs, accs, 'o-', color=COLORS['blue'], linewidth=2.5,
                    markersize=5, label='Calibrator')
            ax.axhline(y=data_f[model]["base_accuracy"], color='black',
                       linestyle=':', alpha=0.4)
            ax.axhline(y=0.95, color=COLORS['red'], linestyle=':', alpha=0.5)
        ax.set_xlabel("Coverage")
        if col == 0:
            ax.set_ylabel("UC-F: Gated Actions\nExec. Accuracy", fontsize=11)
        else:
            ax.set_ylabel("Exec. Accuracy")
        ax.set_ylim(0.4, 1.02)

        # Row 2: UC-G review efficiency
        ax = axes[2, col]
        if model in data_g:
            re = data_g[model]["review_efficiency"]
            pcts = [r["pct_reviewed"] for r in re]
            caught = [r["errors_caught_pct"] for r in re]
            ax.plot(pcts, caught, 'o-', color=COLORS['blue'], linewidth=2.5,
                    markersize=4, label='UQ-guided')
            ax.plot([0, 1], [0, 1], ':', color=COLORS['gray'], linewidth=2,
                    label='Random')
            ax.fill_between(pcts, pcts, caught, alpha=0.1, color=COLORS['blue'])
        ax.set_xlabel("Fraction reviewed")
        if col == 0:
            ax.set_ylabel("UC-G: Review Efficiency\nErrors Caught", fontsize=11)
        else:
            ax.set_ylabel("Errors Caught")
        ax.legend(fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)

    # Row labels on left
    for row, label in enumerate(["(a) Adaptive Clarification",
                                  "(b) Confidence-Gated Actions",
                                  "(c) Human Review Efficiency"]):
        axes[row, 0].annotate(label, xy=(-0.35, 0.5),
                               xycoords='axes fraction',
                               fontsize=11, fontweight='bold',
                               rotation=90, va='center', ha='center')

    fig.suptitle("Production Use Cases — Per-Model Breakdown",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        out_path = f"{args.output_dir}/fig_production_use_cases_permodel.{ext}"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {out_path}")
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
