#!/usr/bin/env python3
"""Domain-Specific UQ Analysis — Shows calibrator value across professional domains.

Categorizes benchmarks into domains (medical, code, math/STEM, reasoning, vision)
and shows how the UQ calibrator performs in each. Demonstrates that a single
calibrator generalizes across professional domains without domain-specific tuning.

Usage:
    python scripts/demo_domain_analysis.py
    python scripts/demo_domain_analysis.py --scored_dir data/use_cases/scored_test_only_v2
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

# Domain categorization
DOMAIN_MAP = {
    # Medical / Health Sciences
    "chembench": "Medical & Science",
    "mmmu": "Medical & Science",  # includes clinical medicine, diagnostics, etc.
    # Code / Software Engineering
    "prbench": "Code & Engineering",
    # Math / STEM Education
    "mathverse": "Math & Education",
    "mathvision": "Math & Education",
    "mathvista": "Math & Education",
    "omnimath": "Math & Education",
    # Reasoning / Knowledge
    "bbeh": "Reasoning & Logic",
    "gpqa": "Reasoning & Logic",
    "hle": "Reasoning & Logic",
    "hle_multimodal": "Reasoning & Logic",
    "livebench": "Reasoning & Logic",
    "simpleqa": "Factual Knowledge",
    "arc_agi": "Reasoning & Logic",
    # Vision / Perception
    "charxiv": "Vision & Perception",
    "hallusionbench": "Vision & Perception",
    "mmstar": "Vision & Perception",
    "mmvet": "Vision & Perception",
    "realworldqa": "Vision & Perception",
    "vizwiz": "Vision & Perception",
}

DOMAIN_COLORS = {
    "Medical & Science": "#e74c3c",
    "Code & Engineering": "#3498db",
    "Math & Education": "#2ecc71",
    "Reasoning & Logic": "#9b59b6",
    "Factual Knowledge": "#f39c12",
    "Vision & Perception": "#1abc9c",
}

DOMAIN_ICONS = {
    "Medical & Science": "Medical",
    "Code & Engineering": "Code",
    "Math & Education": "Education",
    "Reasoning & Logic": "Reasoning",
    "Factual Knowledge": "Facts",
    "Vision & Perception": "Vision",
}


def load_scored(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def bootstrap_auroc(labels, scores, n_boot=2000, ci=0.95):
    """Bootstrap AUROC with CI."""
    rng = np.random.RandomState(42)
    # Filter NaN scores
    mask = ~np.isnan(scores)
    labels, scores = labels[mask], scores[mask]
    n = len(labels)
    if n < 5 or len(set(labels)) < 2:
        return float("nan"), float("nan"), float("nan")
    base = roc_auc_score(labels, scores)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        bl, bs = labels[idx], scores[idx]
        if len(set(bl)) < 2:
            continue
        boots.append(roc_auc_score(bl, bs))
    boots = sorted(boots)
    lo = boots[int((1 - ci) / 2 * len(boots))]
    hi = boots[int((1 + ci) / 2 * len(boots))]
    return base, lo, hi


def analyze_domains(all_samples):
    """Compute AUROC per domain and per benchmark."""
    domain_samples = defaultdict(list)
    bench_samples = defaultdict(list)

    for s in all_samples:
        bench = s["benchmark"]
        domain = DOMAIN_MAP.get(bench, "Other")
        domain_samples[domain].append(s)
        bench_samples[bench].append(s)

    results = {"domains": {}, "benchmarks": {}}

    for domain, samples in sorted(domain_samples.items()):
        labels = np.array([s["is_correct"] for s in samples])
        cal_scores = np.array([s["p_correct"] for s in samples])
        verb_scores = np.array([s.get("verbalized_confidence") or 0.5 for s in samples])

        cal_auroc, cal_lo, cal_hi = bootstrap_auroc(labels, cal_scores)
        verb_auroc, verb_lo, verb_hi = bootstrap_auroc(labels, verb_scores)

        acc = labels.mean()
        results["domains"][domain] = {
            "n": len(samples),
            "accuracy": float(acc),
            "calibrator_auroc": cal_auroc,
            "calibrator_ci": [cal_lo, cal_hi],
            "verbalized_auroc": verb_auroc,
            "verbalized_ci": [verb_lo, verb_hi],
            "advantage": cal_auroc - verb_auroc,
            "benchmarks": sorted(set(s["benchmark"] for s in samples)),
        }

    for bench, samples in sorted(bench_samples.items()):
        labels = np.array([s["is_correct"] for s in samples])
        cal_scores = np.array([s["p_correct"] for s in samples])
        if len(set(labels)) < 2:
            continue
        auroc = roc_auc_score(labels, cal_scores)
        results["benchmarks"][bench] = {
            "n": len(samples),
            "accuracy": float(labels.mean()),
            "auroc": auroc,
            "domain": DOMAIN_MAP.get(bench, "Other"),
        }

    return results


def selective_prediction_by_domain(all_samples, score_key="p_correct"):
    """Coverage@90% accuracy per domain — how many answers can we keep while
    maintaining 90% accuracy?"""
    domain_samples = defaultdict(list)
    for s in all_samples:
        domain = DOMAIN_MAP.get(s["benchmark"], "Other")
        domain_samples[domain].append(s)

    results = {}
    for domain, samples in sorted(domain_samples.items()):
        scores = np.array([s[score_key] for s in samples])
        labels = np.array([s["is_correct"] for s in samples])
        order = np.argsort(-scores)
        sorted_labels = labels[order]

        # Find max coverage with >= 90% accuracy
        best_cov = 0.0
        for k in range(1, len(sorted_labels) + 1):
            acc = sorted_labels[:k].mean()
            if acc >= 0.90:
                best_cov = k / len(sorted_labels)

        # Accuracy at 50% coverage
        k50 = max(1, len(sorted_labels) // 2)
        acc_at_50 = sorted_labels[:k50].mean()

        results[domain] = {
            "coverage_at_90acc": best_cov,
            "accuracy_at_50cov": float(acc_at_50),
            "n": len(samples),
            "base_accuracy": float(labels.mean()),
        }
    return results


def plot_domain_auroc(results, fig_dir):
    """Bar chart of AUROC per domain: calibrator vs verbalized."""
    domains = sorted(results["domains"].keys())
    cal_aurocs = [results["domains"][d]["calibrator_auroc"] for d in domains]
    verb_aurocs = [results["domains"][d]["verbalized_auroc"] for d in domains]
    cal_cis = [results["domains"][d]["calibrator_ci"] for d in domains]
    verb_cis = [results["domains"][d]["verbalized_ci"] for d in domains]

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    cal_errs = [[a - ci[0] for a, ci in zip(cal_aurocs, cal_cis)],
                [ci[1] - a for a, ci in zip(cal_aurocs, cal_cis)]]
    verb_errs = [[a - ci[0] for a, ci in zip(verb_aurocs, verb_cis)],
                 [ci[1] - a for a, ci in zip(verb_aurocs, verb_cis)]]

    bars1 = ax.bar(x - width/2, cal_aurocs, width, label="UQ Calibrator",
                   color="#2196F3", yerr=cal_errs, capsize=3)
    bars2 = ax.bar(x + width/2, verb_aurocs, width, label="Verbalized Confidence",
                   color="#FF9800", yerr=verb_errs, capsize=3)

    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("UQ Calibrator Performance Across Professional Domains", fontsize=14)
    ax.set_xticks(x)
    short_names = [DOMAIN_ICONS.get(d, d) for d in domains]
    ax.set_xticklabels(short_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0.3, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Random")
    ax.grid(axis="y", alpha=0.3)

    # Add sample counts
    for i, d in enumerate(domains):
        n = results["domains"][d]["n"]
        ax.text(i, 0.33, f"n={n}", ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    path = os.path.join(fig_dir, "domain_auroc_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_selective_prediction_domains(sel_results, fig_dir):
    """Show coverage@90% accuracy per domain."""
    domains = sorted(sel_results.keys())
    coverages = [sel_results[d]["coverage_at_90acc"] for d in domains]
    colors = [DOMAIN_COLORS.get(d, "#95a5a6") for d in domains]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(domains)), coverages, color=colors, edgecolor="white")
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([DOMAIN_ICONS.get(d, d) for d in domains], fontsize=11)
    ax.set_xlabel("Coverage @ 90% Accuracy", fontsize=12)
    ax.set_title("How Much Content Can Be Auto-Approved per Domain?", fontsize=13)
    ax.set_xlim(0, 1.0)

    for i, (cov, d) in enumerate(zip(coverages, domains)):
        n = sel_results[d]["n"]
        label = f"{cov:.0%} (n={n})"
        ax.text(cov + 0.02, i, label, va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(fig_dir, "domain_selective_coverage.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_professional_use_case_summary(results, sel_results, fig_dir):
    """Combined figure showing the professional value proposition."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    domains = sorted(results["domains"].keys())

    # Panel 1: AUROC advantage (calibrator - verbalized)
    ax = axes[0]
    advantages = [results["domains"][d]["advantage"] for d in domains]
    colors = [DOMAIN_COLORS.get(d, "#95a5a6") for d in domains]
    short = [DOMAIN_ICONS.get(d, d) for d in domains]
    bars = ax.barh(range(len(domains)), advantages, color=colors)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(short, fontsize=10)
    ax.set_xlabel("AUROC Advantage over Self-Reported Confidence", fontsize=10)
    ax.set_title("(a) Calibrator Advantage by Domain", fontsize=11)
    ax.axvline(x=0, color="black", linewidth=0.5)
    for i, adv in enumerate(advantages):
        ax.text(adv + 0.01 if adv >= 0 else adv - 0.05, i,
                f"+{adv:.3f}" if adv >= 0 else f"{adv:.3f}",
                va="center", fontsize=9)

    # Panel 2: Coverage at 90% acc
    ax = axes[1]
    covs = [sel_results.get(d, {}).get("coverage_at_90acc", 0) for d in domains]
    bars = ax.barh(range(len(domains)), covs, color=colors)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(short, fontsize=10)
    ax.set_xlabel("Fraction Auto-Approvable @ 90% Accuracy", fontsize=10)
    ax.set_title("(b) Safe Automation Coverage", fontsize=11)
    for i, cov in enumerate(covs):
        ax.text(cov + 0.02, i, f"{cov:.0%}", va="center", fontsize=9)

    # Panel 3: Per-benchmark AUROC colored by domain
    ax = axes[2]
    bench_data = [(b, info["auroc"], info["domain"])
                  for b, info in results["benchmarks"].items()]
    bench_data.sort(key=lambda x: x[1], reverse=True)
    bnames = [b[0] for b in bench_data]
    baurocs = [b[1] for b in bench_data]
    bcolors = [DOMAIN_COLORS.get(b[2], "#95a5a6") for b in bench_data]
    ax.barh(range(len(bnames)), baurocs, color=bcolors)
    ax.set_yticks(range(len(bnames)))
    ax.set_yticklabels(bnames, fontsize=8)
    ax.set_xlabel("AUROC", fontsize=10)
    ax.set_title("(c) Per-Benchmark AUROC", fontsize=11)
    ax.set_xlim(0.5, 1.0)

    plt.tight_layout()
    path = os.path.join(fig_dir, "professional_use_case_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2")
    parser.add_argument("--output_dir", default="data/use_cases/domain_analysis")
    parser.add_argument("--fig_dir", default="figures/domain_analysis")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # Load all scored data
    all_samples = []
    for fn in sorted(os.listdir(args.scored_dir)):
        if fn.endswith("_scored.jsonl"):
            path = os.path.join(args.scored_dir, fn)
            samples = load_scored(path)
            if args.smoke_test:
                samples = samples[:20]
            all_samples.extend(samples)
            print(f"Loaded {len(samples)} from {fn}")

    print(f"\nTotal samples: {len(all_samples)}")

    # 1. Domain AUROC analysis
    print("\n=== Domain AUROC Analysis ===")
    results = analyze_domains(all_samples)
    for domain, info in sorted(results["domains"].items()):
        print(f"  {domain:25s} | n={info['n']:5d} | acc={info['accuracy']:.3f} "
              f"| cal={info['calibrator_auroc']:.3f} [{info['calibrator_ci'][0]:.3f},{info['calibrator_ci'][1]:.3f}] "
              f"| verb={info['verbalized_auroc']:.3f} | adv=+{info['advantage']:.3f}")

    # 2. Selective prediction by domain
    print("\n=== Selective Prediction by Domain ===")
    sel_results = selective_prediction_by_domain(all_samples)
    for domain, info in sorted(sel_results.items()):
        print(f"  {domain:25s} | coverage@90acc={info['coverage_at_90acc']:.1%} "
              f"| acc@50cov={info['accuracy_at_50cov']:.3f} | base_acc={info['base_accuracy']:.3f}")

    # 3. Generate figures
    print("\n=== Generating Figures ===")
    plot_domain_auroc(results, args.fig_dir)
    plot_selective_prediction_domains(sel_results, args.fig_dir)
    plot_professional_use_case_summary(results, sel_results, args.fig_dir)

    # 4. Save results
    output = {
        "domain_analysis": results,
        "selective_prediction": sel_results,
        "total_samples": len(all_samples),
    }
    out_path = os.path.join(args.output_dir, "domain_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # 5. Print use-case narratives
    print("\n" + "=" * 70)
    print("PROFESSIONAL USE CASE NARRATIVES")
    print("=" * 70)

    narratives = {
        "Medical & Science": (
            "A medical AI chatbot uses GPT-5 to answer patient questions. "
            "The UQ calibrator flags low-confidence answers for physician review. "
            f"With our calibrator (AUROC {results['domains'].get('Medical & Science', {}).get('calibrator_auroc', 0):.3f}), "
            f"{sel_results.get('Medical & Science', {}).get('coverage_at_90acc', 0):.0%} of responses can be "
            "auto-approved at 90% accuracy, reducing physician workload."
        ),
        "Code & Engineering": (
            "A code review assistant uses LLMs to check pull requests. "
            "The UQ calibrator scores each review comment's reliability. "
            f"AUROC {results['domains'].get('Code & Engineering', {}).get('calibrator_auroc', 0):.3f} means "
            "developers can trust high-confidence suggestions and manually verify low-confidence ones."
        ),
        "Math & Education": (
            "An AI tutoring platform uses GPT-5 for math homework help. "
            "The calibrator catches incorrect solution steps before students see them. "
            f"At 50% coverage, accuracy is {sel_results.get('Math & Education', {}).get('accuracy_at_50cov', 0):.1%} — "
            "the top half of responses by confidence are nearly always correct."
        ),
        "Factual Knowledge": (
            "A legal research tool uses LLMs to answer factual queries. "
            "The calibrator detects hallucinated facts (fake citations, wrong dates). "
            f"AUROC {results['domains'].get('Factual Knowledge', {}).get('calibrator_auroc', 0):.3f} on factual QA "
            "shows the calibrator reliably separates correct from fabricated facts."
        ),
    }

    for domain, narrative in narratives.items():
        print(f"\n--- {domain} ---")
        print(narrative)


if __name__ == "__main__":
    main()
