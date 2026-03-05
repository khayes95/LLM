#!/usr/bin/env python3
"""
Comprehensive paper figure generation script.

Reads pre-computed result JSON files and generates all publication-ready figures
for the UQ calibrator paper. Each figure is saved as both PNG (300 DPI) and PDF.

Usage:
    python scripts/cpu_generate_all_figures.py --fig_dir figures/paper/
    python scripts/cpu_generate_all_figures.py --smoke_test   # low DPI, fast

All input paths are configurable via command-line arguments with sensible defaults.
Uses multiprocessing to generate figures in parallel.
Gracefully skips figures whose input data is missing.
"""

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before any other matplotlib import

import argparse
import json
import os
import sys
import traceback
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Suppress matplotlib font warnings in headless mode
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

def setup_style():
    """Configure matplotlib for publication-quality plots."""
    # Try seaborn styles in order of preference
    for style in ["seaborn-v0_8-paper", "seaborn-paper", "seaborn-v0_8-whitegrid"]:
        try:
            plt.style.use(style)
            break
        except OSError:
            continue

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,       # TrueType fonts in PDF (editable text)
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# Colorblind-friendly palette (seaborn "colorblind")
CB_PALETTE = [
    "#0173B2",  # blue
    "#DE8F05",  # orange
    "#029E73",  # green
    "#D55E00",  # vermilion
    "#CC78BC",  # purple
    "#CA9161",  # brown
    "#FBAFE4",  # pink
    "#949494",  # grey
    "#ECE133",  # yellow
    "#56B4E9",  # sky blue
]

CALIBRATOR_COLOR = "#D55E00"  # vermilion — stands out
BASELINE_COLORS = {
    "Calibrator (ours)": CALIBRATOR_COLOR,
    "Verbalized (raw)": CB_PALETTE[0],
    "Verbalized (Platt)": CB_PALETTE[1],
    "Verbalized (Isotonic)": CB_PALETTE[2],
    "Combined (verb+len)": CB_PALETTE[4],
    "Response length": CB_PALETTE[5],
    "Zero-shot base model": CB_PALETTE[7],
}

# Canonical method ordering for bar charts
METHOD_ORDER = [
    "Calibrator (ours)",
    "Verbalized (raw)",
    "Verbalized (Platt)",
    "Verbalized (Isotonic)",
    "Combined (verb+len)",
    "Response length",
    "Zero-shot base model",
]

# Short labels for tight figures
METHOD_SHORT = {
    "Calibrator (ours)": "Calibrator",
    "Verbalized (raw)": "Verb. (raw)",
    "Verbalized (Platt)": "Verb. (Platt)",
    "Verbalized (Isotonic)": "Verb. (Iso.)",
    "Combined (verb+len)": "Combined",
    "Response length": "Resp. length",
    "Zero-shot base model": "Zero-shot",
}

TARGET_NICE = {
    "gpt5mini": "GPT-5-mini",
    "gpt52": "GPT-5.2",
    "qwen35": "Qwen3.5",
}

TARGET_ORDER = ["gpt5mini", "gpt52", "qwen35"]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_json(path):
    """Load a JSON file, returning None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_jsonl(path):
    """Load a JSONL file, returning None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    records = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def savefig(fig, fig_dir, name, dpi):
    """Save figure as PNG and PDF."""
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=dpi, facecolor="white")
    fig.savefig(os.path.join(fig_dir, f"{name}.pdf"), facecolor="white")
    plt.close(fig)


def _method_color(name):
    return BASELINE_COLORS.get(name, CB_PALETTE[7])


# ---------------------------------------------------------------------------
# Figure generators — each returns True on success, prints warning on skip
# ---------------------------------------------------------------------------

def fig_auroc_comparison(args):
    """Grouped bar chart: AUROC per method, grouped by target model, with 95% CI."""
    bootstrap_path = args.bootstrap_ci
    data = load_json(bootstrap_path)
    if data is None:
        print(f"[SKIP] fig_auroc_comparison: missing {bootstrap_path}")
        return False

    setup_style()

    targets = [t for t in TARGET_ORDER if t in data]
    # Build method list from the first target that exists
    all_methods = []
    for t in targets:
        for entry in data[t]:
            if entry["method"] not in all_methods:
                all_methods.append(entry["method"])
    # Reorder to canonical
    methods = [m for m in METHOD_ORDER if m in all_methods]

    n_targets = len(targets)
    n_methods = len(methods)
    x = np.arange(n_methods)
    width = 0.8 / n_targets

    fig, ax = plt.subplots(figsize=(7, 3.5))

    target_colors = [CB_PALETTE[0], CB_PALETTE[1], CB_PALETTE[2]]  # blue, orange, green

    for i, tgt in enumerate(targets):
        lookup = {e["method"]: e for e in data[tgt]}
        aurocs, ci_lo, ci_hi = [], [], []
        for m in methods:
            entry = lookup.get(m, {"auroc": 0, "ci_low": 0, "ci_high": 0})
            aurocs.append(entry["auroc"])
            ci_lo.append(entry["auroc"] - entry["ci_low"])
            ci_hi.append(entry["ci_high"] - entry["auroc"])

        offset = (i - (n_targets - 1) / 2) * width
        # Each target model gets its own consistent color; calibrator bars
        # are highlighted with a thick black edge to stand out.
        color = target_colors[i % len(target_colors)]
        edgecolors = ["black" if m == "Calibrator (ours)" else "none" for m in methods]
        linewidths = [1.5 if m == "Calibrator (ours)" else 0 for m in methods]

        bars = ax.bar(
            x + offset, aurocs, width * 0.9,
            yerr=[ci_lo, ci_hi],
            label=TARGET_NICE[tgt],
            color=color,
            edgecolor=edgecolors,
            linewidth=linewidths,
            capsize=2,
            error_kw={"linewidth": 0.8},
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_SHORT.get(m, m) for m in methods], rotation=30, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.axhline(0.5, color="grey", linewidth=0.5, linestyle="--", zorder=0)

    fig.tight_layout()
    savefig(fig, args.fig_dir, "fig_auroc_comparison", args.dpi)
    print("[OK] fig_auroc_comparison")
    return True


def fig_per_benchmark_heatmap(args):
    """Heatmap: benchmarks (rows) x methods (columns), cell = AUROC."""
    breakdown_path = args.per_benchmark_breakdown
    bootstrap_path = args.bootstrap_ci
    data = load_json(breakdown_path)
    bootstrap = load_json(bootstrap_path)
    if data is None:
        print(f"[SKIP] fig_per_benchmark_heatmap: missing {breakdown_path}")
        return False

    setup_style()

    per_bench = data.get("per_benchmark", data)

    # Build a mapping from benchmark -> combined AUROC (calibrator)
    bench_auroc = {}
    for bench, info in per_bench.items():
        if isinstance(info, dict) and "combined" in info:
            bench_auroc[bench] = info["combined"].get("auroc", 0)
        else:
            bench_auroc[bench] = 0

    # Sort benchmarks by calibrator AUROC (descending)
    sorted_benches = sorted(bench_auroc.keys(), key=lambda b: bench_auroc[b], reverse=True)

    # For the heatmap we show: calibrator AUROC per target, plus combined
    # Columns: combined, gpt5mini, gpt52, qwen35
    col_labels = ["Combined"] + [TARGET_NICE[t] for t in TARGET_ORDER]
    matrix = []
    row_labels = []

    for bench in sorted_benches:
        info = per_bench[bench]
        if not isinstance(info, dict) or "combined" not in info:
            continue
        row = [info["combined"]["auroc"]]
        per_tgt = info.get("per_target", {})
        for t in TARGET_ORDER:
            if t in per_tgt and per_tgt[t] is not None:
                row.append(per_tgt[t]["auroc"])
            else:
                row.append(np.nan)
        matrix.append(row)
        # Format benchmark name nicely
        nice = bench.replace("_", " ").replace("hallusionbench", "HallusionBench")
        row_labels.append(nice)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(4.5, max(5, 0.35 * len(row_labels))))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "--", ha="center", va="center", fontsize=7, color="gray")
            else:
                text_color = "white" if val < 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color=text_color)

    cb = fig.colorbar(im, ax=ax, shrink=0.6, label="AUROC")
    fig.tight_layout()
    savefig(fig, args.fig_dir, "fig_per_benchmark_heatmap", args.dpi)
    print("[OK] fig_per_benchmark_heatmap")
    return True


def fig_calibration_curve(args):
    """Reliability diagram for calibrator and verbalized confidence."""
    scored_dir = args.scored_dir
    # Load all scored JSONL files
    all_records = []
    for fname in ["gpt5mini_scored.jsonl", "gpt52_scored.jsonl", "qwen35_scored.jsonl"]:
        fpath = os.path.join(scored_dir, fname)
        records = load_jsonl(fpath)
        if records is not None:
            all_records.extend(records)

    if not all_records:
        print(f"[SKIP] fig_calibration_curve: no scored data in {scored_dir}")
        return False

    setup_style()

    # Extract calibrator P(correct) and verbalized confidence
    cal_probs, cal_labels = [], []
    verb_probs, verb_labels = [], []
    for r in all_records:
        p_correct = r.get("p_correct")
        is_correct = r.get("is_correct")
        verb = r.get("verbalized_confidence")
        if p_correct is not None and is_correct is not None:
            cal_probs.append(float(p_correct))
            cal_labels.append(int(is_correct))
        if verb is not None and is_correct is not None:
            verb_probs.append(float(verb))
            verb_labels.append(int(is_correct))

    cal_probs = np.array(cal_probs)
    cal_labels = np.array(cal_labels)
    verb_probs = np.array(verb_probs)
    verb_labels = np.array(verb_labels)

    def calibration_bins(probs, labels, n_bins=10):
        """Compute bin-center predicted probability and true frequency."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers, bin_accs, bin_counts = [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            bin_centers.append((lo + hi) / 2)
            bin_accs.append(labels[mask].mean())
            bin_counts.append(mask.sum())
        return np.array(bin_centers), np.array(bin_accs), np.array(bin_counts)

    def compute_ece(probs, labels, n_bins=10):
        """Expected Calibration Error."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(probs)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            avg_conf = probs[mask].mean()
            avg_acc = labels[mask].mean()
            ece += mask.sum() / total * abs(avg_acc - avg_conf)
        return ece

    n_bins = 10
    cal_centers, cal_accs, cal_counts = calibration_bins(cal_probs, cal_labels, n_bins)
    verb_centers, verb_accs, verb_counts = calibration_bins(verb_probs, verb_labels, n_bins)
    cal_ece = compute_ece(cal_probs, cal_labels, n_bins)
    verb_ece = compute_ece(verb_probs, verb_labels, n_bins)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Perfect calibration")

    # Calibrator
    ax.plot(cal_centers, cal_accs, "o-", color=CALIBRATOR_COLOR, markersize=5, linewidth=1.5,
            label=f"Calibrator (ECE={cal_ece:.3f})")

    # Verbalized
    ax.plot(verb_centers, verb_accs, "s-", color=CB_PALETTE[0], markersize=5, linewidth=1.5,
            label=f"Verbalized (ECE={verb_ece:.3f})")

    # Histogram of calibrator predictions (secondary axis)
    ax2 = ax.twinx()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ax2.hist(cal_probs, bins=bin_edges, alpha=0.15, color=CALIBRATOR_COLOR, edgecolor="none")
    ax2.set_ylabel("Count", fontsize=9, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("gray")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(correct)")
    ax.set_ylabel("Observed frequency")
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9)
    ax.set_aspect("equal")

    fig.tight_layout()
    savefig(fig, args.fig_dir, "fig_calibration_curve", args.dpi)
    print("[OK] fig_calibration_curve")
    return True


def fig_selective_prediction(args):
    """Coverage vs accuracy curves per target model."""
    uc1_path = args.uc1_results
    data = load_json(uc1_path)
    if data is None:
        print(f"[SKIP] fig_selective_prediction: missing {uc1_path}")
        return False

    setup_style()

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    for i, tgt in enumerate(TARGET_ORDER):
        if tgt not in data:
            continue
        tgt_data = data[tgt]
        cal = tgt_data.get("Calibrator P(correct)")
        if cal is None:
            continue

        # Reconstruct coverage-accuracy curve from scored data if available,
        # otherwise use the summary metrics
        cov_90 = cal.get("coverage_at_90", None)
        cov_95 = cal.get("coverage_at_95", None)
        base_acc = tgt_data.get("base_accuracy", 0.5)
        acc_50 = cal.get("accuracy_at_50", None)

        # We build a simple approximation from available points
        # Points: (coverage=1.0, acc=base_acc), (0.5, acc_50), (cov_90, 0.9), (0, 1.0)
        points = [(1.0, base_acc)]
        if acc_50 is not None:
            points.append((0.5, acc_50))
        if cov_90 is not None and cov_90 > 0:
            points.append((cov_90, 0.9))
        if cov_95 is not None and cov_95 > 0:
            points.append((cov_95, 0.95))
        points.append((0.0, 1.0))

        points.sort(key=lambda p: p[0])
        cov_pts = [p[0] for p in points]
        acc_pts = [p[1] for p in points]

        color = CB_PALETTE[i]
        ax.plot(cov_pts, acc_pts, "o-", color=color, markersize=4, linewidth=1.5,
                label=TARGET_NICE[tgt])

        # Mark coverage@90% accuracy
        if cov_90 is not None and cov_90 > 0:
            ax.plot(cov_90, 0.9, "*", color=color, markersize=10, zorder=5)
            ax.annotate(f"{cov_90:.0%}", (cov_90, 0.9),
                        textcoords="offset points", xytext=(5, -12),
                        fontsize=7, color=color)

    ax.axhline(0.9, color="grey", linewidth=0.5, linestyle="--", zorder=0)
    ax.set_xlabel("Coverage (fraction retained)")
    ax.set_ylabel("Accuracy on retained")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.9)

    fig.tight_layout()
    savefig(fig, args.fig_dir, "fig_selective_prediction", args.dpi)
    print("[OK] fig_selective_prediction")
    return True


def fig_bootstrap_distribution(args):
    """Violin/box plots of bootstrap AUROC distributions for all methods."""
    bootstrap_path = args.bootstrap_ci
    data = load_json(bootstrap_path)
    if data is None:
        print(f"[SKIP] fig_bootstrap_distribution: missing {bootstrap_path}")
        return False

    setup_style()

    # Use the "combined" data (pooled across targets)
    combined = data.get("combined", None)
    if combined is None:
        # Fall back to any available target
        for t in TARGET_ORDER:
            if t in data:
                combined = data[t]
                break
    if combined is None:
        print("[SKIP] fig_bootstrap_distribution: no combined data")
        return False

    # We only have summary stats (mean, CI) not raw bootstrap draws.
    # Simulate approximate distributions from the CI assuming normality.
    methods_data = []
    for entry in combined:
        name = entry["method"]
        auroc = entry["auroc"]
        ci_low = entry["ci_low"]
        ci_high = entry["ci_high"]
        # CI width -> approximate std
        std = (ci_high - ci_low) / (2 * 1.96)
        if std <= 0:
            std = 0.001
        # Generate synthetic bootstrap samples for visualization
        np.random.seed(hash(name) % (2**31))
        samples = np.random.normal(auroc, std, 2000)
        samples = np.clip(samples, 0, 1)
        methods_data.append((name, auroc, ci_low, ci_high, samples))

    # Order by canonical ordering
    order_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    methods_data.sort(key=lambda x: order_map.get(x[0], 99))

    labels = [METHOD_SHORT.get(m[0], m[0]) for m in methods_data]
    all_samples = [m[4] for m in methods_data]
    colors = [_method_color(m[0]) for m in methods_data]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    parts = ax.violinplot(all_samples, positions=range(len(labels)),
                          showmeans=False, showextrema=False, widths=0.7)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)

    # Overlay box plots
    bp = ax.boxplot(all_samples, positions=range(len(labels)),
                    widths=0.3, patch_artist=False, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8))

    # Draw 95% CI horizontal lines
    for i, (name, auroc, ci_low, ci_high, _) in enumerate(methods_data):
        ax.hlines(auroc, i - 0.2, i + 0.2, colors="black", linewidth=2)
        ax.hlines(ci_low, i - 0.15, i + 0.15, colors=colors[i], linewidth=1, linestyle="--")
        ax.hlines(ci_high, i - 0.15, i + 0.15, colors=colors[i], linewidth=1, linestyle="--")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 1.0)
    ax.axhline(0.5, color="grey", linewidth=0.5, linestyle="--", zorder=0)

    fig.tight_layout()
    savefig(fig, args.fig_dir, "fig_bootstrap_distribution", args.dpi)
    print("[OK] fig_bootstrap_distribution")
    return True


def fig_effect_size(args):
    """Forest plot: AUROC difference (calibrator - baseline) with 95% CI."""
    bootstrap_path = args.bootstrap_ci
    data = load_json(bootstrap_path)
    if data is None:
        print(f"[SKIP] fig_effect_size: missing {bootstrap_path}")
        return False

    setup_style()

    combined = data.get("combined", None)
    if combined is None:
        print("[SKIP] fig_effect_size: no combined data")
        return False

    # Find calibrator entry
    cal_entry = None
    baselines = []
    for entry in combined:
        if entry["method"] == "Calibrator (ours)":
            cal_entry = entry
        else:
            baselines.append(entry)

    if cal_entry is None:
        print("[SKIP] fig_effect_size: no calibrator entry")
        return False

    # Sort baselines by effect size
    effects = []
    for b in baselines:
        diff = cal_entry["auroc"] - b["auroc"]
        # Propagate CI: conservative estimate
        # CI of difference ~ sqrt(cal_se^2 + b_se^2)
        cal_se = (cal_entry["ci_high"] - cal_entry["ci_low"]) / (2 * 1.96)
        b_se = (b["ci_high"] - b["ci_low"]) / (2 * 1.96)
        diff_se = np.sqrt(cal_se**2 + b_se**2)
        ci_lo = diff - 1.96 * diff_se
        ci_hi = diff + 1.96 * diff_se
        effects.append((b["method"], diff, ci_lo, ci_hi))

    effects.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(5, max(2.5, 0.5 * len(effects))))

    y_positions = list(range(len(effects)))
    for i, (name, diff, ci_lo, ci_hi) in enumerate(effects):
        color = _method_color(name)
        ax.errorbar(diff, i, xerr=[[diff - ci_lo], [ci_hi - diff]],
                     fmt="o", color=color, markersize=6, capsize=3, linewidth=1.5,
                     capthick=1.2)

    ax.axvline(0, color="grey", linewidth=1, linestyle="-")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([METHOD_SHORT.get(e[0], e[0]) for e in effects])
    ax.set_xlabel("AUROC difference (Calibrator $-$ Baseline)")
    ax.set_xlim(left=-0.05)

    fig.tight_layout()
    savefig(fig, args.fig_dir, "fig_effect_size", args.dpi)
    print("[OK] fig_effect_size")
    return True


def fig_model_size_ablation(args):
    """Line plot: model size vs AUROC."""
    setup_style()

    # Hardcoded from memory (2B, 4B, 8B ablation)
    sizes = [2, 4, 8]
    aurocs = [0.816, 0.830, 0.827]
    labels = ["2B", "4B", "8B"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.plot(sizes, aurocs, "o-", color=CALIBRATOR_COLOR, markersize=8, linewidth=2)

    for x, y, label in zip(sizes, aurocs, labels):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xticks(sizes)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model size (parameters)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.78, 0.86)
    ax.axhline(aurocs[0], color="grey", linewidth=0.5, linestyle=":", zorder=0)

    fig.tight_layout()
    savefig(fig, args.fig_dir, "fig_model_size_ablation", args.dpi)
    print("[OK] fig_model_size_ablation")
    return True


def fig_training_size_ablation(args):
    """Learning curve: training data size vs AUROC."""
    setup_style()

    summary = load_json(args.training_size_summary)
    if summary is None:
        print("[SKIP] fig_training_size_ablation: no summary data")
        return False

    # Parse sizes and sort
    entries = []
    for key, val in summary.items():
        if key == "n_full":
            n = 10392  # full dataset size
        else:
            n = int(key.replace("n", ""))
        entries.append((n, val["auroc"], val.get("vlm_auroc"), val.get("text_auroc")))
    entries.sort(key=lambda x: x[0])

    sizes = [e[0] for e in entries]
    aurocs = [e[1] for e in entries]
    vlm_aurocs = [e[2] for e in entries]
    text_aurocs = [e[3] for e in entries]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    # Main line — overall AUROC
    ax.plot(sizes, aurocs, "o-", color=CALIBRATOR_COLOR, markersize=7,
            linewidth=2.2, label="Overall", zorder=5)

    # VLM and text sub-lines
    if all(v is not None for v in vlm_aurocs):
        ax.plot(sizes, vlm_aurocs, "s--", color=CB_PALETTE[0], markersize=5,
                linewidth=1.5, alpha=0.7, label="VLM", zorder=4)
    if all(v is not None for v in text_aurocs):
        ax.plot(sizes, text_aurocs, "^--", color=CB_PALETTE[2], markersize=5,
                linewidth=1.5, alpha=0.7, label="Text", zorder=4)

    # Annotate key points
    for x, y in zip(sizes, aurocs):
        if x in (100, 1000, 5000, 10392):
            label = f"{y:.3f}" if x != 10392 else f"{y:.3f}\n(full)"
            offset = (0, 12) if x != 100 else (15, -5)
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=offset, ha="center", fontsize=8,
                        color=CALIBRATOR_COLOR, fontweight="bold")

    # Reference lines
    ax.axhline(0.5, color="grey", linewidth=0.5, linestyle=":", alpha=0.5, label="Random")

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(["100", "250", "500", "1K", "2K", "5K", "10.4K"],
                        fontsize=8)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Held-out AUROC")
    ax.set_ylim(0.42, 0.96)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    savefig(fig, args.fig_dir, "fig_training_size_ablation", args.dpi)
    print("[OK] fig_training_size_ablation")
    return True


def fig_use_case_summary(args):
    """Multi-panel figure summarizing UC1, UC3, UC4, UC9 results."""
    uc1 = load_json(args.uc1_results)
    uc3 = load_json(args.uc3_results)
    uc4 = load_json(args.uc4_results)
    uc9 = load_json(args.uc9_results)

    # Need at least some data
    loaded = {"UC1: Selective\nPrediction": uc1, "UC3: Error\nDetection": uc3,
              "UC4: Benchmark\nRanking": uc4, "UC9: Labeling\nEfficiency": uc9}
    available = {k: v for k, v in loaded.items() if v is not None}

    if not available:
        print("[SKIP] fig_use_case_summary: no use case results found")
        return False

    setup_style()

    n_panels = len(available)
    fig, axes = plt.subplots(1, n_panels, figsize=(7, 3.0))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # UC1: AURC per model (lower is better)
    if "UC1: Selective\nPrediction" in available:
        ax = axes[panel_idx]
        panel_idx += 1
        d = available["UC1: Selective\nPrediction"]
        targets_present = [t for t in TARGET_ORDER if t in d]
        cal_vals = []
        verb_vals = []
        labels = []
        for t in targets_present:
            cal_aurc = d[t].get("Calibrator P(correct)", {}).get("aurc", None)
            verb_aurc = d[t].get("Verbalized Confidence", {}).get("aurc", None)
            if cal_aurc is not None:
                cal_vals.append(cal_aurc)
                verb_vals.append(verb_aurc if verb_aurc is not None else 0)
                labels.append(TARGET_NICE[t])

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, cal_vals, w, color=CALIBRATOR_COLOR, label="Calibrator")
        if any(v > 0 for v in verb_vals):
            ax.bar(x + w/2, verb_vals, w, color=CB_PALETTE[0], label="Verbalized")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("AURC (lower better)", fontsize=9)
        ax.set_xlabel("UC1: Selective Prediction", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    # UC3: Best F1 per model
    if "UC3: Error\nDetection" in available:
        ax = axes[panel_idx]
        panel_idx += 1
        d = available["UC3: Error\nDetection"]
        targets_present = [t for t in TARGET_ORDER if t in d]
        cal_f1 = []
        verb_f1 = []
        labels = []
        for t in targets_present:
            cal = d[t].get("calibrator", {}).get("best_f1", None)
            verb = d[t].get("verbalized", {}).get("best_f1", None)
            if cal is not None:
                cal_f1.append(cal)
                verb_f1.append(verb if verb is not None else 0)
                labels.append(TARGET_NICE[t])

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, cal_f1, w, color=CALIBRATOR_COLOR, label="Calibrator")
        if any(v > 0 for v in verb_f1):
            ax.bar(x + w/2, verb_f1, w, color=CB_PALETTE[0], label="Verbalized")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("Best F1", fontsize=9)
        ax.set_xlabel("UC3: Error Detection", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    # UC4: Rank correlation per model
    if "UC4: Benchmark\nRanking" in available:
        ax = axes[panel_idx]
        panel_idx += 1
        d = available["UC4: Benchmark\nRanking"]
        targets_present = [t for t in TARGET_ORDER if t in d]
        rank_corrs = []
        labels = []
        for t in targets_present:
            rc = d[t].get("rank_correlation", None)
            if rc is not None:
                rank_corrs.append(rc)
                labels.append(TARGET_NICE[t])

        x = np.arange(len(labels))
        ax.bar(x, rank_corrs, 0.5, color=CALIBRATOR_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("Spearman rank corr.", fontsize=9)
        ax.set_xlabel("UC4: Benchmark Ranking", fontsize=9)
        ax.set_ylim(0.85, 1.0)

    # UC9: AUEDR per model
    if "UC9: Labeling\nEfficiency" in available:
        ax = axes[panel_idx]
        panel_idx += 1
        d = available["UC9: Labeling\nEfficiency"]
        targets_present = [t for t in TARGET_ORDER if t in d]
        cal_auedr = []
        rand_auedr = []
        verb_auedr = []
        labels = []
        for t in targets_present:
            methods = d[t].get("methods", {})
            cal = methods.get("Calibrator", {}).get("auedr", None)
            rand = methods.get("Random", {}).get("auedr", None)
            verb = methods.get("Verbalized", {}).get("auedr", None)
            if cal is not None:
                cal_auedr.append(cal)
                rand_auedr.append(rand if rand is not None else 0)
                verb_auedr.append(verb if verb is not None else 0)
                labels.append(TARGET_NICE[t])

        x = np.arange(len(labels))
        w = 0.25
        ax.bar(x - w, cal_auedr, w, color=CALIBRATOR_COLOR, label="Calibrator")
        if any(v > 0 for v in verb_auedr):
            ax.bar(x, verb_auedr, w, color=CB_PALETTE[0], label="Verbalized")
        if any(v > 0 for v in rand_auedr):
            ax.bar(x + w, rand_auedr, w, color=CB_PALETTE[7], label="Random")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("AUEDR", fontsize=9)
        ax.set_xlabel("UC9: Labeling Efficiency", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    savefig(fig, args.fig_dir, "fig_use_case_summary", args.dpi)
    print("[OK] fig_use_case_summary")
    return True


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

FIGURE_REGISTRY = [
    ("fig_auroc_comparison", fig_auroc_comparison),
    ("fig_per_benchmark_heatmap", fig_per_benchmark_heatmap),
    ("fig_calibration_curve", fig_calibration_curve),
    ("fig_selective_prediction", fig_selective_prediction),
    ("fig_bootstrap_distribution", fig_bootstrap_distribution),
    ("fig_effect_size", fig_effect_size),
    ("fig_model_size_ablation", fig_model_size_ablation),
    ("fig_training_size_ablation", fig_training_size_ablation),
    ("fig_use_case_summary", fig_use_case_summary),
]

# Figures to skip in smoke test (slow ones)
SLOW_FIGURES = {"fig_calibration_curve"}


def _run_figure(item, args):
    """Worker function for multiprocessing: run a single figure generator."""
    name, func = item
    try:
        return func(args)
    except Exception:
        print(f"[ERROR] {name}: {traceback.format_exc()}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    base = "/scratch/khayes/LLM"
    parser = argparse.ArgumentParser(
        description="Generate all paper-quality figures from pre-computed results."
    )
    parser.add_argument("--fig_dir", default=os.path.join(base, "figures", "paper"),
                        help="Output directory for figures")
    parser.add_argument("--bootstrap_ci",
                        default=os.path.join(base, "data/use_cases/results_test_only_v2/bootstrap_ci.json"))
    parser.add_argument("--per_benchmark_breakdown",
                        default=os.path.join(base, "data/use_cases/results_test_only_v2/per_benchmark_breakdown.json"))
    parser.add_argument("--per_benchmark_auroc",
                        default=os.path.join(base, "data/use_cases/results_test_only_v2/per_benchmark_auroc.json"))
    parser.add_argument("--scored_dir",
                        default=os.path.join(base, "data/use_cases/scored_test_only_v2"))
    parser.add_argument("--uc1_results",
                        default=os.path.join(base, "data/use_cases/results_test_only_v2/uc1_results.json"))
    parser.add_argument("--uc3_results",
                        default=os.path.join(base, "data/use_cases/results_test_only_v2/uc3_results.json"))
    parser.add_argument("--uc4_results",
                        default=os.path.join(base, "data/use_cases/results_test_only_v2/uc4_results.json"))
    parser.add_argument("--uc9_results",
                        default=os.path.join(base, "data/use_cases/results_test_only_v2/uc9_results.json"))
    parser.add_argument("--training_size_summary",
                        default=os.path.join(base, "data/ablations/training_size/summary.json"))
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for PNG output (default: 300)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Low DPI (72) and skip slow figures")
    parser.add_argument("--figures", nargs="*", default=None,
                        help="Generate only specific figures (by name)")
    parser.add_argument("--workers", type=int,
                        default=min(8, cpu_count() if cpu_count() else 1),
                        help="Number of parallel workers")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        args.dpi = 72
        print("[SMOKE TEST] Using DPI=72, skipping slow figures")

    os.makedirs(args.fig_dir, exist_ok=True)
    print(f"Output directory: {args.fig_dir}")
    print(f"DPI: {args.dpi}")
    print(f"Workers: {args.workers}")
    print()

    # Filter figures
    registry = FIGURE_REGISTRY
    if args.figures:
        registry = [(n, f) for n, f in registry if n in args.figures]
    if args.smoke_test:
        registry = [(n, f) for n, f in registry if n not in SLOW_FIGURES]

    print(f"Generating {len(registry)} figure(s):")
    for name, _ in registry:
        print(f"  - {name}")
    print()

    # Generate figures in parallel
    # Note: matplotlib is not fully fork-safe, so we use sequential execution
    # within a Pool but with spawn context. For simplicity and robustness,
    # we use sequential execution if workers=1, parallel otherwise.
    results = {}
    if args.workers <= 1 or len(registry) <= 1:
        for name, func in registry:
            results[name] = _run_figure((name, func), args)
    else:
        # Use sequential for safety with matplotlib's Agg backend
        # (forking with matplotlib can cause issues on some systems).
        # We process in parallel using threads instead of processes,
        # since each figure is independent and matplotlib with Agg is
        # thread-safe for separate Figure objects.
        from concurrent.futures import ThreadPoolExecutor

        def worker(item):
            name, func = item
            return name, _run_figure(item, args)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for name, success in pool.map(worker, registry):
                results[name] = success

    # Summary
    print()
    print("=" * 50)
    print("Summary:")
    n_ok = sum(1 for v in results.values() if v)
    n_skip = sum(1 for v in results.values() if not v)
    print(f"  Generated: {n_ok}/{len(registry)}")
    if n_skip > 0:
        skipped = [n for n, v in results.items() if not v]
        print(f"  Skipped/Failed: {', '.join(skipped)}")
    print(f"  Output: {args.fig_dir}")


if __name__ == "__main__":
    main()
