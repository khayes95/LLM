#!/usr/bin/env python3
"""Generate publication-quality reliability diagram from pre-computed bin data.

Reads: data/use_cases/results_test_only_v3/reliability_diagram_data.json
Saves: figures/paper/fig_reliability_diagram.pdf
       overleaf/figures/fig_reliability_diagram.pdf
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# ---------- Style (matches cpu_generate_all_figures.py) ----------
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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CALIBRATOR_COLOR = "#D55E00"  # vermilion

# ---------- Load data ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(ROOT, "data/use_cases/results_test_only_v3/reliability_diagram_data.json")
with open(data_path) as f:
    data = json.load(f)

bins = data["bins"]
ece = data["ece"]
n_bins = data["n_bins"]
total = data["total_samples"]

mean_pred = np.array([b["mean_predicted_probability"] for b in bins])
mean_acc = np.array([b["mean_actual_accuracy"] for b in bins])
counts = np.array([b["n_samples"] for b in bins])
bin_lowers = np.array([b["bin_lower"] for b in bins])
bin_uppers = np.array([b["bin_upper"] for b in bins])
bin_width = bin_uppers[0] - bin_lowers[0]

# ---------- Plot ----------
fig, ax1 = plt.subplots(figsize=(3.5, 3.5))

# Perfect calibration diagonal
ax1.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Perfect calibration", zorder=1)

# Calibration gap shading (subtle)
for mp, ma in zip(mean_pred, mean_acc):
    ax1.plot([mp, mp], [mp, ma], color=CALIBRATOR_COLOR, alpha=0.25, linewidth=1.5, zorder=2)

# Calibrator reliability curve
ax1.plot(mean_pred, mean_acc, "o-", color=CALIBRATOR_COLOR, markersize=5, linewidth=1.8,
         label=f"Pinocchio (ECE = {ece:.3f})", zorder=5)

# Histogram on secondary axis
ax2 = ax1.twinx()
bin_edges = np.linspace(0, 1, n_bins + 1)
ax2.bar(bin_lowers + bin_width / 2, counts, width=bin_width * 0.85,
        alpha=0.15, color=CALIBRATOR_COLOR, edgecolor="none", zorder=0)
ax2.set_ylabel("Sample count", fontsize=9, color="gray")
ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color("gray")
ax2.spines["right"].set_alpha(0.5)
# Set y-limit so histogram doesn't dominate
ax2.set_ylim(0, max(counts) * 2.5)

# Labels & formatting
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel("Predicted P(correct)")
ax1.set_ylabel("Observed accuracy")
ax1.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.9)
ax1.set_aspect("equal")

# ECE annotation in the lower-right area
ax1.text(0.95, 0.05, f"$n$ = {total:,}\nECE = {ece:.3f}",
         transform=ax1.transAxes, fontsize=8.5,
         verticalalignment="bottom", horizontalalignment="right",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.9))

fig.tight_layout()

# ---------- Save ----------
out_dirs = [
    os.path.join(ROOT, "figures/paper"),
    os.path.join(ROOT, "overleaf/figures"),
]
for d in out_dirs:
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "fig_reliability_diagram.pdf")
    fig.savefig(path, facecolor="white")
    print(f"Saved: {path}")

# Also save PNG for quick viewing
png_path = os.path.join(ROOT, "figures/paper/fig_reliability_diagram.png")
fig.savefig(png_path, dpi=300, facecolor="white")
print(f"Saved: {png_path}")

plt.close(fig)
print("Done.")
