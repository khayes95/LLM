#!/usr/bin/env python3
"""Generate paper-ready figures from cross-model JSON results.

Data-driven: loads actual results from data/cross_model/*.json.
No hardcoded numbers — all values come from saved evaluation results.

Usage:
    python scripts/generate_paper_figures.py
    python scripts/generate_paper_figures.py --output_dir figures/paper
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Academic style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.dpi'] = 150
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
}

CROSS_MODEL_DIR = Path("data/cross_model")


def load_result(filename):
    path = CROSS_MODEL_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def fig_cross_model_transfer_matrix(output_dir):
    """Figure: Cross-model transfer grouped bar chart — compare 3 calibrators + VLM judge."""
    targets = ["GPT-5.2", "Qwen3-VL-30B", "Qwen3.5-397B"]

    calibrators = {
        "v3 (mini only)": {
            "GPT-5.2": "text_v3_on_gpt52.json",
            "Qwen3-VL-30B": "text_v3_on_qwen3vl_updated.json",
            "Qwen3.5-397B": "text_v3_on_qwen35.json",
        },
        "GPT-5.2 only": {
            "GPT-5.2": "text_gpt52cal_on_gpt52.json",
            "Qwen3-VL-30B": "text_gpt52cal_on_qwen3vl.json",
            "Qwen3.5-397B": "text_gpt52cal_on_qwen35.json",
        },
        "Combined": {
            "GPT-5.2": "text_combined_on_gpt52.json",
            "Qwen3-VL-30B": "text_combined_on_qwen3vl.json",
            "Qwen3.5-397B": "text_combined_on_qwen35.json",
        },
    }

    vlm_files = {
        "GPT-5-mini": "vlm_judge_vsr_fixed_on_gpt5mini.json",
        "GPT-5.2": "vlm_judge_vsr_fixed_on_gpt52.json",
        "Qwen3-VL-30B": "vlm_judge_vsr_fixed_on_qwen3vl_updated.json",
        "Qwen3.5-397B": "vlm_judge_vsr_fixed_on_qwen35.json",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Text calibrators (grouped bars) ---
    cal_colors = [COLORS['gray'], COLORS['orange'], COLORS['blue']]
    x = np.arange(len(targets))
    width = 0.25
    for i, (cal_name, cal_files) in enumerate(calibrators.items()):
        aurocs = []
        for target in targets:
            fname = cal_files.get(target)
            r = load_result(fname) if fname else None
            aurocs.append(r["auroc"] if r else 0)
        bars = ax1.bar(x + i * width - width, aurocs, width, label=cal_name,
                       color=cal_colors[i], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, aurocs):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax1.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(targets, rotation=15, ha='right')
    ax1.set_ylim(0.4, 0.95)
    ax1.set_ylabel('AUROC')
    ax1.set_title('(a) Text Calibrator Comparison')
    ax1.legend(fontsize=8, loc='upper left')

    # --- VLM judge ---
    vlm_names, vlm_aurocs = [], []
    for name, fname in vlm_files.items():
        r = load_result(fname)
        if r:
            vlm_names.append(name)
            vlm_aurocs.append(r["auroc"])

    if vlm_names:
        x2 = np.arange(len(vlm_names))
        bars = ax2.bar(x2, vlm_aurocs, color=COLORS['purple'], width=0.5, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, vlm_aurocs):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(vlm_names, rotation=15, ha='right')
        ax2.set_ylim(0.4, 0.9)
        ax2.set_ylabel('AUROC')
        ax2.set_title('(b) VLM Judge (Qwen3-VL-8B)')

    plt.suptitle('Cross-Model Transfer: Train on Model A, Predict on Model B', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_cross_model_transfer.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig_cross_model_transfer.pdf")


def fig_per_benchmark_heatmap(output_dir):
    """Figure: Per-benchmark AUROC heatmap across models and judges."""
    result_files = {
        "v3→GPT5.2": "text_v3_on_gpt52.json",
        "Combined→GPT5.2": "text_combined_on_gpt52.json",
        "Combined→Qwen3.5": "text_combined_on_qwen35.json",
        "Combined→Qwen3VL": "text_combined_on_qwen3vl.json",
        "VLM→GPT5.2": "vlm_judge_vsr_fixed_on_gpt52.json",
        "VLM→GPT5-mini": "vlm_judge_vsr_fixed_on_gpt5mini.json",
    }

    # Collect benchmarks and values
    all_benchmarks = set()
    data = {}
    for label, fname in result_files.items():
        r = load_result(fname)
        if not r or "per_benchmark" not in r:
            continue
        data[label] = {}
        for bench, metrics in r["per_benchmark"].items():
            auroc = metrics.get("auroc")
            n = metrics.get("n_samples", 0)
            if auroc is not None and n >= 10:
                data[label][bench] = auroc
                all_benchmarks.add(bench)

    if not data:
        print("  Skipping heatmap: no data")
        return

    # Sort benchmarks by average AUROC
    bench_avg = {}
    for bench in all_benchmarks:
        vals = [data[l].get(bench) for l in data if data[l].get(bench) is not None]
        if vals:
            bench_avg[bench] = np.mean(vals)
    benchmarks = sorted(bench_avg.keys(), key=lambda b: bench_avg[b], reverse=True)
    labels = list(data.keys())

    # Build matrix
    matrix = np.full((len(benchmarks), len(labels)), np.nan)
    for j, label in enumerate(labels):
        for i, bench in enumerate(benchmarks):
            val = data[label].get(bench)
            if val is not None:
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(7, max(4, len(benchmarks) * 0.35)))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.35, vmax=0.9, aspect='auto')

    # Annotate cells
    for i in range(len(benchmarks)):
        for j in range(len(labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 or val > 0.8 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(benchmarks)))
    ax.set_yticklabels(benchmarks, fontsize=9)

    plt.colorbar(im, ax=ax, label='AUROC', shrink=0.8)
    ax.set_title('Per-Benchmark AUROC Across Models')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_per_benchmark_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig_per_benchmark_heatmap.pdf")


def fig_calibration_comparison(output_dir):
    """Figure: ECE and Brier score comparison across all evaluations."""
    all_files = {
        "Text→GPT5.2": "text_v3_on_gpt52.json",
        "Text→Qwen3.5": "text_v3_on_qwen35.json",
        "Text→Qwen3-VL": "text_v3_on_qwen3vl_updated.json",
        "VLM→GPT5.2": "vlm_judge_vsr_fixed_on_gpt52.json",
        "VLM→GPT5-mini": "vlm_judge_vsr_fixed_on_gpt5mini.json",
        "VLM→Qwen3-VL": "vlm_judge_vsr_fixed_on_qwen3vl_updated.json",
    }

    names, eces, briers, aurocs = [], [], [], []
    for label, fname in all_files.items():
        r = load_result(fname)
        if not r:
            continue
        names.append(label)
        eces.append(r.get("ece", 0))
        briers.append(r.get("brier", 0))
        aurocs.append(r.get("auroc", 0))

    if not names:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ECE comparison
    x = np.arange(len(names))
    colors = [COLORS['blue'] if 'Text' in n else COLORS['purple'] for n in names]
    bars = ax1.bar(x, eces, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, eces):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax1.axhline(y=0.1, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='Target ECE < 0.10')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax1.set_ylabel('ECE (lower = better)')
    ax1.set_title('(a) Expected Calibration Error')
    ax1.legend(fontsize=8)

    # AUROC vs Brier scatter
    for i, (name, auroc, brier) in enumerate(zip(names, aurocs, briers)):
        color = COLORS['blue'] if 'Text' in name else COLORS['purple']
        marker = 'o' if 'Text' in name else 's'
        ax2.scatter(auroc, brier, c=color, marker=marker, s=60, edgecolors='black', linewidths=0.5, zorder=5)
        ax2.annotate(name, (auroc, brier), textcoords='offset points',
                     xytext=(5, 5), fontsize=7)

    ax2.set_xlabel('AUROC (higher = better)')
    ax2.set_ylabel('Brier Score (lower = better)')
    ax2.set_title('(b) Discrimination vs Calibration')

    text_patch = mpatches.Patch(color=COLORS['blue'], label='Text Calibrator')
    vlm_patch = mpatches.Patch(color=COLORS['purple'], label='VLM Judge')
    ax2.legend(handles=[text_patch, vlm_patch], fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_calibration.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig_calibration.pdf")


def fig_verbalized_vs_calibrator(output_dir):
    """Figure: Verbalized confidence baseline vs trained calibrator."""
    # Load verbalized confidence data from prediction files
    runs_dir = Path("runs")
    prefixes = {
        "GPT-5.2": ("gpt52_high_", "text_v3_on_gpt52.json"),
        "Qwen3.5-397B": ("qwen35_397b_", "text_v3_on_qwen35.json"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_data = []

    for model_name, (prefix, result_file) in prefixes.items():
        # Extract verbalized confidence AUROC
        all_confs, all_labels = [], []
        for d in sorted(runs_dir.iterdir()):
            if not d.name.startswith(prefix):
                continue
            pred_file = d / "predictions.jsonl"
            if not pred_file.exists() or pred_file.stat().st_size == 0:
                continue
            with open(pred_file) as f:
                for line in f:
                    try:
                        pred = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    score = pred.get("score", {})
                    correct = score.get("correct", -1) if isinstance(score, dict) else score
                    if correct not in (0, 1):
                        continue
                    conf = _extract_confidence(pred)
                    if conf is not None:
                        all_confs.append(conf)
                        all_labels.append(float(correct == 1))

        from sklearn.metrics import roc_auc_score
        verb_auroc = None
        if len(all_confs) >= 10 and len(set(all_labels)) > 1:
            verb_auroc = roc_auc_score(all_labels, all_confs)

        # Load calibrator AUROC
        cal_auroc = None
        r = load_result(result_file)
        if r:
            cal_auroc = r.get("auroc")

        bar_data.append((model_name, verb_auroc, cal_auroc))

    if not bar_data:
        print("  Skipping verbalized vs calibrator: no data")
        return

    x = np.arange(len(bar_data))
    width = 0.3

    for i, (name, verb, cal) in enumerate(bar_data):
        if verb is not None:
            b1 = ax.bar(x[i] - width/2, verb, width, color=COLORS['gray'],
                        edgecolor='black', linewidth=0.5)
            ax.text(x[i] - width/2, verb + 0.01, f'{verb:.3f}', ha='center', fontsize=9)
        if cal is not None:
            b2 = ax.bar(x[i] + width/2, cal, width, color=COLORS['blue'],
                        edgecolor='black', linewidth=0.5)
            ax.text(x[i] + width/2, cal + 0.01, f'{cal:.3f}', ha='center', fontsize=9)

    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in bar_data])
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.4, 0.85)
    ax.set_title('Verbalized Confidence vs Trained Calibrator')

    verb_patch = mpatches.Patch(color=COLORS['gray'], edgecolor='black', linewidth=0.5,
                                label='Verbalized (self-reported)')
    cal_patch = mpatches.Patch(color=COLORS['blue'], edgecolor='black', linewidth=0.5,
                               label='Trained Calibrator')
    ax.legend(handles=[verb_patch, cal_patch], fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_verbalized_vs_calibrator.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig_verbalized_vs_calibrator.pdf")


def _extract_confidence(pred):
    """Extract verbalized confidence from prediction record."""
    response_text = pred.get("response_text", "")
    if isinstance(response_text, str):
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and "confidence" in parsed:
                conf = parsed["confidence"]
                if isinstance(conf, (int, float)) and 0 <= conf <= 1:
                    return float(conf)
        except (json.JSONDecodeError, ValueError):
            pass
    prediction = pred.get("prediction", {})
    if isinstance(prediction, dict) and "confidence" in prediction:
        conf = prediction["confidence"]
        if isinstance(conf, (int, float)) and 0 <= conf <= 1:
            return float(conf)
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from cross-model results")
    parser.add_argument("--output_dir", default="figures/paper",
                        help="Output directory for figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures from {CROSS_MODEL_DIR}")
    print(f"Output: {output_dir}\n")

    fig_cross_model_transfer_matrix(output_dir)
    fig_per_benchmark_heatmap(output_dir)
    fig_calibration_comparison(output_dir)
    fig_verbalized_vs_calibrator(output_dir)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
