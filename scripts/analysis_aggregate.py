#!/usr/bin/env python3
"""Comprehensive analysis of cross-model results using aggregate data.

Produces:
  1. Per-benchmark breakdown table (text + CSV)
  2. Summary statistics table for paper
  3. Which benchmarks transfer best/worst
  4. ECE/Brier/AUROC comparison across all settings

No GPU needed — reads from data/cross_model/*.json

Usage:
    python scripts/analysis_aggregate.py [--output_dir figures/analysis]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


CROSS_MODEL_DIR = Path("data/cross_model")

# Canonical result files (skip interim/smoke/duplicates)
TEXT_RESULTS = {
    "v3→GPT5-mini (in-dist)": "text_v3_on_gpt52.json",  # Note: v3 trained on GPT-5-mini
    "v3→GPT5.2": "text_v3_on_gpt52_reeval.json",
    "v3→Qwen3-VL-30B": "text_v3_on_qwen3vl_updated.json",
    "v3→Qwen3.5-397B": "text_v3_on_qwen35.json",
    "Combined→GPT5.2": "text_combined_on_gpt52.json",
    "Combined→Qwen3-VL-30B": "text_combined_on_qwen3vl.json",
    "Combined→Qwen3.5-397B": "text_combined_on_qwen35.json",
    "GPT5.2-cal→GPT5.2": "text_gpt52cal_on_gpt52.json",
    "GPT5.2-cal→Qwen3-VL-30B": "text_gpt52cal_on_qwen3vl.json",
    "GPT5.2-cal→Qwen3.5-397B": "text_gpt52cal_on_qwen35.json",
}

VLM_RESULTS = {
    "VLM→GPT5-mini": "vlm_judge_vsr_fixed_on_gpt5mini.json",
    "VLM→GPT5.2": "vlm_judge_vsr_fixed_on_gpt52.json",
    "VLM→Qwen3-VL-30B": "vlm_judge_vsr_fixed_on_qwen3vl_updated.json",
}


def load_result(filename):
    path = CROSS_MODEL_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def summary_table(output_dir):
    """Print a comprehensive summary table of all results."""
    print("=" * 80)
    print("SUMMARY TABLE: All Cross-Model Results")
    print("=" * 80)

    header = f"{'Setting':<30} {'AUROC':>7} {'ECE':>7} {'Brier':>7} {'N':>6} {'Base%':>6}"
    print(header)
    print("-" * 80)

    rows = []

    print("\n--- Text Calibrator ---")
    for label, fname in TEXT_RESULTS.items():
        r = load_result(fname)
        if not r:
            continue
        auroc = r.get("auroc", 0)
        ece = r.get("ece", 0)
        brier = r.get("brier", 0)
        n = r.get("n_samples", 0)
        base = r.get("base_rate", 0)
        row = f"{label:<30} {auroc:>7.4f} {ece:>7.4f} {brier:>7.4f} {n:>6} {base:>6.1%}"
        print(row)
        rows.append({"setting": label, "type": "text", "auroc": auroc, "ece": ece,
                      "brier": brier, "n_samples": n, "base_rate": base})

    print("\n--- VLM Judge ---")
    for label, fname in VLM_RESULTS.items():
        r = load_result(fname)
        if not r:
            continue
        auroc = r.get("auroc", 0)
        ece = r.get("ece", 0)
        brier = r.get("brier", 0)
        n = r.get("n_samples", 0)
        base = r.get("base_rate", 0)
        row = f"{label:<30} {auroc:>7.4f} {ece:>7.4f} {brier:>7.4f} {n:>6} {base:>6.1%}"
        print(row)
        rows.append({"setting": label, "type": "vlm", "auroc": auroc, "ece": ece,
                      "brier": brier, "n_samples": n, "base_rate": base})

    # Save as JSON for other scripts
    with open(output_dir / "summary_table.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {output_dir}/summary_table.json")


def per_benchmark_analysis(output_dir):
    """Detailed per-benchmark breakdown showing which benchmarks transfer best/worst."""
    print("\n" + "=" * 80)
    print("PER-BENCHMARK ANALYSIS")
    print("=" * 80)

    # Collect all per-benchmark data
    all_results = {}
    all_results.update(TEXT_RESULTS)
    all_results.update(VLM_RESULTS)

    bench_data = defaultdict(dict)  # bench -> {setting: {auroc, n}}
    all_benchmarks = set()

    for label, fname in all_results.items():
        r = load_result(fname)
        if not r or "per_benchmark" not in r:
            continue
        for bench, metrics in r["per_benchmark"].items():
            auroc = metrics.get("auroc")
            n = metrics.get("n_samples", 0)
            if auroc is not None and n >= 10:
                bench_data[bench][label] = {"auroc": auroc, "n": n}
                all_benchmarks.add(bench)

    if not bench_data:
        print("No per-benchmark data available.")
        return

    # Compute average AUROC per benchmark (across all settings)
    bench_stats = {}
    for bench in all_benchmarks:
        aurocs = [bench_data[bench][s]["auroc"] for s in bench_data[bench]]
        bench_stats[bench] = {
            "mean_auroc": np.mean(aurocs),
            "std_auroc": np.std(aurocs),
            "min_auroc": np.min(aurocs),
            "max_auroc": np.max(aurocs),
            "n_settings": len(aurocs),
        }

    # Sort by mean AUROC
    sorted_benchmarks = sorted(bench_stats.keys(), key=lambda b: bench_stats[b]["mean_auroc"], reverse=True)

    print(f"\n{'Benchmark':<20} {'Mean':>7} {'Std':>6} {'Min':>6} {'Max':>6} {'#Settings':>9}")
    print("-" * 60)
    for bench in sorted_benchmarks:
        s = bench_stats[bench]
        print(f"{bench:<20} {s['mean_auroc']:>7.3f} {s['std_auroc']:>6.3f} "
              f"{s['min_auroc']:>6.3f} {s['max_auroc']:>6.3f} {s['n_settings']:>9}")

    print(f"\n--- Top 5 Easiest Benchmarks (best transfer) ---")
    for bench in sorted_benchmarks[:5]:
        s = bench_stats[bench]
        print(f"  {bench}: mean AUROC = {s['mean_auroc']:.3f}")

    print(f"\n--- Top 5 Hardest Benchmarks (worst transfer) ---")
    for bench in sorted_benchmarks[-5:]:
        s = bench_stats[bench]
        print(f"  {bench}: mean AUROC = {s['mean_auroc']:.3f}")

    # Save detailed per-benchmark data
    output = {
        "benchmark_stats": bench_stats,
        "sorted_by_mean_auroc": sorted_benchmarks,
        "per_setting": {bench: bench_data[bench] for bench in sorted_benchmarks},
    }
    with open(output_dir / "per_benchmark_analysis.json", "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved: {output_dir}/per_benchmark_analysis.json")


def calibrator_comparison(output_dir):
    """Compare the three text calibrator variants head-to-head."""
    print("\n" + "=" * 80)
    print("CALIBRATOR COMPARISON (Text)")
    print("=" * 80)

    targets = ["GPT5.2", "Qwen3-VL-30B", "Qwen3.5-397B"]
    calibrators = {
        "v3 (mini only)": ["text_v3_on_gpt52_reeval.json", "text_v3_on_qwen3vl_updated.json", "text_v3_on_qwen35.json"],
        "GPT-5.2 only": ["text_gpt52cal_on_gpt52.json", "text_gpt52cal_on_qwen3vl.json", "text_gpt52cal_on_qwen35.json"],
        "Combined": ["text_combined_on_gpt52.json", "text_combined_on_qwen3vl.json", "text_combined_on_qwen35.json"],
    }

    print(f"\n{'Calibrator':<20}", end="")
    for t in targets:
        print(f" {t:>15}", end="")
    print(f" {'Mean':>8}")
    print("-" * 70)

    comparison = {}
    for cal_name, files in calibrators.items():
        aurocs = []
        print(f"{cal_name:<20}", end="")
        for fname in files:
            r = load_result(fname)
            a = r["auroc"] if r else 0
            aurocs.append(a)
            print(f" {a:>15.4f}", end="")
        mean_a = np.mean(aurocs) if aurocs else 0
        print(f" {mean_a:>8.4f}")
        comparison[cal_name] = {"aurocs": aurocs, "mean": mean_a}

    # Winner per target
    print("\nBest calibrator per target:")
    for i, target in enumerate(targets):
        best_cal = max(calibrators.keys(), key=lambda c: load_result(calibrators[c][i]).get("auroc", 0) if load_result(calibrators[c][i]) else 0)
        best_auroc = load_result(calibrators[best_cal][i]).get("auroc", 0)
        print(f"  {target}: {best_cal} ({best_auroc:.4f})")

    with open(output_dir / "calibrator_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=float)
    print(f"\nSaved: {output_dir}/calibrator_comparison.json")


def latex_tables(output_dir):
    """Generate LaTeX-ready tables for the paper."""
    print("\n" + "=" * 80)
    print("LATEX TABLES")
    print("=" * 80)

    # Table 1: Main cross-model results
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-model transfer AUROC. Calibrator trained on Source model responses, evaluated on Target model responses.}")
    lines.append(r"\label{tab:cross_model}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{3}{c}{\textbf{Target Model}} \\")
    lines.append(r"\cmidrule(lr){2-4}")
    lines.append(r"\textbf{Calibrator} & GPT-5.2 & Qwen3-VL-30B & Qwen3.5-397B \\")
    lines.append(r"\midrule")

    cal_map = {
        r"v3 (GPT-5-mini)": ["text_v3_on_gpt52_reeval.json", "text_v3_on_qwen3vl_updated.json", "text_v3_on_qwen35.json"],
        r"GPT-5.2 only": ["text_gpt52cal_on_gpt52.json", "text_gpt52cal_on_qwen3vl.json", "text_gpt52cal_on_qwen35.json"],
        r"Combined": ["text_combined_on_gpt52.json", "text_combined_on_qwen3vl.json", "text_combined_on_qwen35.json"],
    }

    for cal_name, files in cal_map.items():
        vals = []
        for fname in files:
            r = load_result(fname)
            vals.append(f"{r['auroc']:.3f}" if r else "---")
        lines.append(f"{cal_name} & {' & '.join(vals)} \\\\")

    lines.append(r"\midrule")

    # VLM row
    vlm_files = ["vlm_judge_vsr_fixed_on_gpt52.json", "vlm_judge_vsr_fixed_on_qwen3vl_updated.json", None]
    vlm_vals = []
    for fname in vlm_files:
        if fname:
            r = load_result(fname)
            vlm_vals.append(f"{r['auroc']:.3f}" if r else "---")
        else:
            vlm_vals.append("---")
    lines.append(f"VLM Judge & {' & '.join(vlm_vals)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    print(latex)

    with open(output_dir / "table_cross_model.tex", "w") as f:
        f.write(latex)
    print(f"\nSaved: {output_dir}/table_cross_model.tex")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="figures/analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_table(output_dir)
    per_benchmark_analysis(output_dir)
    calibrator_comparison(output_dir)
    latex_tables(output_dir)

    print("\n" + "=" * 80)
    print("ALL ANALYSIS COMPLETE")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
