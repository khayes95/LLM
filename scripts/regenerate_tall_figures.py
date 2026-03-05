#!/usr/bin/env python3
"""Regenerate UC3, UC7, UC9 figures with taller stacked layouts for report readability.

The original scripts produce 1x3 wide figures (18x6) that look tiny when embedded
in a portrait PDF report. This script regenerates them as taller layouts.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score
)

SCORED_DIR = Path("data/use_cases/scored_test_only_v2")
FIG_DIR = Path("figures/use_cases_v2")

DOMAIN_MAP = {
    "factual": ["simpleqa", "gpqa", "chembench", "hallusionbench"],
    "reasoning": ["bbeh", "arc_agi", "multichallenge"],
    "math": ["omnimath", "mathvista", "mathvision", "mathverse"],
    "knowledge": ["mmlu_pro", "mmmu", "hle"],
    "multimodal": ["charxiv", "mmstar", "mmvet", "realworldqa", "vizwiz"],
    "coding": ["prbench"],
    "long_context": ["livebench", "oolong", "longbench"],
}


def get_domain(benchmark):
    for domain, benches in DOMAIN_MAP.items():
        if benchmark in benches:
            return domain
    return "other"


def load_scored(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def load_all():
    all_scored = {}
    for target in ["gpt5mini", "gpt52", "qwen35"]:
        path = SCORED_DIR / f"{target}_scored.jsonl"
        if path.exists():
            all_scored[target] = load_scored(path)
    return all_scored


# =====================================================================
# UC3: Error Detection - 2x1 vertical layout
# =====================================================================
def regen_uc3(all_scored):
    """Regenerate UC3 as 2-row layout: PR curves on top, domain bars on bottom."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # Top: Precision-Recall curves (all 3 models)
    ax = axes[0]
    model_names = {"gpt5mini": "GPT-5-mini", "gpt52": "GPT-5.2", "qwen35": "Qwen3.5"}
    colors = {"gpt5mini": "#4472C4", "gpt52": "#ED7D31", "qwen35": "#70AD47"}

    for target, samples in all_scored.items():
        labels = np.array([s["is_correct"] for s in samples])
        scores = np.array([s["p_correct"] for s in samples])
        error_labels = 1 - labels
        error_scores = 1 - scores
        if len(set(error_labels)) <= 1:
            continue
        auroc = roc_auc_score(error_labels, error_scores)
        prec, rec, _ = precision_recall_curve(error_labels, error_scores)
        ax.plot(rec, prec, color=colors[target], linewidth=2,
                label=f"{model_names[target]} (AUROC={auroc:.3f})")

    ax.set_xlabel("Recall (fraction of errors caught)", fontsize=11)
    ax.set_ylabel("Precision (fraction of flags that are real errors)", fontsize=11)
    ax.set_title("Error Detection: Precision-Recall", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)

    # Bottom: Per-domain AUROC comparison (calibrator vs verbalized)
    ax = axes[1]
    first_target = list(all_scored.keys())[0]
    samples = all_scored[first_target]

    domain_cal = defaultdict(lambda: {"labels": [], "scores": []})
    domain_verb = defaultdict(lambda: {"labels": [], "scores": []})

    for s in samples:
        d = get_domain(s["benchmark"])
        domain_cal[d]["labels"].append(s["is_correct"])
        domain_cal[d]["scores"].append(s["p_correct"])
        if "verbalized_confidence" in s and s["verbalized_confidence"] is not None:
            domain_verb[d]["labels"].append(s["is_correct"])
            domain_verb[d]["scores"].append(s["verbalized_confidence"])

    domains = sorted(domain_cal.keys())
    cal_aurocs = []
    verb_aurocs = []
    for d in domains:
        lbls = np.array(domain_cal[d]["labels"])
        scrs = np.array(domain_cal[d]["scores"])
        if len(set(lbls)) > 1:
            cal_aurocs.append(roc_auc_score(lbls, scrs))
        else:
            cal_aurocs.append(0)

        if d in domain_verb and len(set(domain_verb[d]["labels"])) > 1:
            verb_aurocs.append(roc_auc_score(
                domain_verb[d]["labels"], domain_verb[d]["scores"]))
        else:
            verb_aurocs.append(0)

    x = np.arange(len(domains))
    w = 0.35
    ax.bar(x - w/2, cal_aurocs, w, color="#4472C4", label="Calibrator", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, verb_aurocs, w, color="#ED7D31", label="Verbalized", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("AUROC (error detection)", fontsize=11)
    ax.set_title(f"Per-Domain Error Detection - {model_names[first_target]}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIG_DIR / "uc3_error_detection.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =====================================================================
# UC7: Self-Improvement - 2x1 vertical layout
# =====================================================================
def regen_uc7(all_scored):
    """Regenerate UC7 as 2-row layout: heatmap on top, domain weaknesses on bottom."""
    model_names = {"gpt5mini": "GPT-5-mini", "gpt52": "GPT-5.2", "qwen35": "Qwen3.5"}
    p_threshold = 0.7

    # Compute overconfident error rates
    all_oc = {}
    for target, samples in all_scored.items():
        bench_data = defaultdict(lambda: {"wrong": 0, "oc_errors": 0, "total": 0})
        for s in samples:
            b = s["benchmark"]
            bench_data[b]["total"] += 1
            if not s["is_correct"]:
                bench_data[b]["wrong"] += 1
                if s["p_correct"] > p_threshold:
                    bench_data[b]["oc_errors"] += 1
        all_oc[target] = {}
        for b, d in bench_data.items():
            if d["total"] >= 5 and d["wrong"] > 0:
                all_oc[target][b] = d["oc_errors"] / d["wrong"]

    fig, axes = plt.subplots(2, 1, figsize=(8, 12))

    # Top: Overconfident error rate heatmap
    ax = axes[0]
    models = sorted(all_oc.keys())
    benchmarks = set()
    for m in models:
        benchmarks.update(all_oc[m].keys())
    benchmarks = sorted(benchmarks)

    data = np.full((len(benchmarks), len(models)), np.nan)
    for j, m in enumerate(models):
        for i, b in enumerate(benchmarks):
            if b in all_oc[m]:
                data[i, j] = all_oc[m][b]

    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([model_names.get(m, m) for m in models], fontsize=11)
    ax.set_yticks(range(len(benchmarks)))
    ax.set_yticklabels(benchmarks, fontsize=9)
    ax.set_title("Overconfident Error Rate by Benchmark\n(P(correct) > 0.7 but answer wrong, as fraction of all errors)",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="OC Error Rate")

    # Annotate cells
    for i in range(len(benchmarks)):
        for j in range(len(models)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=8, color=color)

    # Bottom: Domain-level weakness comparison across models
    ax = axes[1]
    domain_oc = defaultdict(lambda: defaultdict(list))
    for target in models:
        for bench, rate in all_oc[target].items():
            domain_oc[get_domain(bench)][target].append(rate)

    domains = sorted(domain_oc.keys())
    x = np.arange(len(domains))
    width = 0.25
    model_colors = {"gpt52": "#4472C4", "gpt5mini": "#ED7D31", "qwen35": "#70AD47"}

    for idx, m in enumerate(models):
        means = [np.mean(domain_oc[d].get(m, [0])) for d in domains]
        ax.bar(x + idx * width - width, means, width, color=model_colors.get(m, "gray"),
               label=model_names.get(m, m), edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Mean Overconfident Error Rate", fontsize=11)
    ax.set_title("Weakness by Domain (All Models)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIG_DIR / "uc7_self_improvement.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =====================================================================
# UC9: Annotation Prioritization - 2x2 layout (3 models + legend)
# =====================================================================
def regen_uc9(all_scored):
    """Regenerate UC9 as 2x2 layout instead of 1x3."""
    model_names = {"gpt5mini": "GPT-5-mini", "gpt52": "GPT-5.2", "qwen35": "Qwen3.5"}
    budgets = np.linspace(0.01, 1.0, 100)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0]]

    for idx, (target, samples) in enumerate(all_scored.items()):
        if idx >= 3:
            break
        ax = axes_flat[idx]
        labels = np.array([s["is_correct"] for s in samples])
        scores = np.array([s["p_correct"] for s in samples])
        n = len(labels)
        total_errors = int((1 - labels).sum())

        if total_errors == 0:
            continue

        # Calibrator: sort by ascending P(correct)
        cal_order = np.argsort(scores)
        cal_edr = []
        for b in budgets:
            k = max(1, int(n * b))
            found = (1 - labels[cal_order[:k]]).sum()
            cal_edr.append(found / total_errors)

        # Verbalized baseline
        verb_scores = []
        verb_labels = []
        for s in samples:
            vc = s.get("verbalized_confidence")
            if vc is not None:
                verb_scores.append(vc)
                verb_labels.append(s["is_correct"])

        verb_edr = None
        if verb_scores:
            verb_labels_arr = np.array(verb_labels)
            verb_scores_arr = np.array(verb_scores)
            verb_total_errors = int((1 - verb_labels_arr).sum())
            if verb_total_errors > 0:
                verb_order = np.argsort(verb_scores_arr)
                verb_edr = []
                for b in budgets:
                    k = max(1, int(len(verb_labels_arr) * b))
                    found = (1 - verb_labels_arr[verb_order[:k]]).sum()
                    verb_edr.append(found / verb_total_errors)

        # Oracle: sort by correctness (errors first)
        oracle_order = np.argsort(labels)
        oracle_edr = []
        for b in budgets:
            k = max(1, int(n * b))
            found = (1 - labels[oracle_order[:k]]).sum()
            oracle_edr.append(found / total_errors)

        # Random baseline
        random_edr = budgets.tolist()

        # Compute AUEDRs
        cal_auedr = float(np.trapz(cal_edr, budgets))
        random_auedr = float(np.trapz(random_edr, budgets))
        oracle_auedr = float(np.trapz(oracle_edr, budgets))

        ax.plot(budgets * 100, oracle_edr, "g--", linewidth=1.5, alpha=0.7,
                label=f"Oracle (AUEDR={oracle_auedr:.3f})")
        ax.plot(budgets * 100, cal_edr, "b-", linewidth=2,
                label=f"Calibrator (AUEDR={cal_auedr:.3f})")
        if verb_edr:
            verb_auedr = float(np.trapz(verb_edr, budgets))
            ax.plot(budgets * 100, verb_edr, "r--", linewidth=1.5, alpha=0.7,
                    label=f"Verbalized (AUEDR={verb_auedr:.3f})")
        ax.plot(budgets * 100, random_edr, "k:", linewidth=1, alpha=0.5,
                label=f"Random (AUEDR={random_auedr:.3f})")

        dist_label = "in-dist" if target == "gpt5mini" else "cross-model"
        ax.set_title(f"{model_names[target]} ({dist_label})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Annotation Budget (%)", fontsize=10)
        ax.set_ylabel("Error Discovery Rate", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.05)

    # Bottom-right: summary text
    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        "Annotation Prioritization\n"
        "─────────────────────────\n\n"
        "Sort predictions by ascending\n"
        "P(correct). Annotators find\n"
        "errors ~2x faster than random.\n\n"
        "AUEDR (higher = better):\n"
    )
    for target, samples in all_scored.items():
        labels = np.array([s["is_correct"] for s in samples])
        scores = np.array([s["p_correct"] for s in samples])
        n = len(labels)
        total_errors = int((1 - labels).sum())
        if total_errors > 0:
            cal_order = np.argsort(scores)
            cal_edr = []
            for b in budgets:
                k = max(1, int(n * b))
                found = (1 - labels[cal_order[:k]]).sum()
                cal_edr.append(found / total_errors)
            auedr = float(np.trapz(cal_edr, budgets))
            summary += f"  {model_names[target]}: {auedr:.3f}\n"

    ax.text(0.1, 0.5, summary, fontsize=11, family="monospace",
            verticalalignment="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f7ff", edgecolor="#2b6cb0"))

    plt.tight_layout()
    out = FIG_DIR / "uc9_annotation_prioritization.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    print("Regenerating UC figures with taller layouts...")
    all_scored = load_all()
    if not all_scored:
        print("ERROR: No scored data found")
        sys.exit(1)

    print(f"Loaded {sum(len(v) for v in all_scored.values())} samples from {len(all_scored)} models")

    regen_uc3(all_scored)
    regen_uc7(all_scored)
    regen_uc9(all_scored)

    print("\nDone! All figures saved to", FIG_DIR)
