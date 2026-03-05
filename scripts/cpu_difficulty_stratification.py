#!/usr/bin/env python3
"""Compute AUROC broken down by question difficulty.

Difficulty is defined per question (unique `id`) as:
    difficulty = 1 - mean(is_correct) across all target models that answered it.

Stratification:
  1. Three difficulty bands: Easy (all correct), Medium (mixed), Hard (all wrong)
  2. Five quantile-based bands for finer granularity

Also computes:
  - Per-benchmark difficulty and stratified AUROC
  - Spearman correlation between question difficulty and calibrator confidence

Output: data/use_cases/results_test_only/difficulty_stratification.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def load_scored_data(scored_dir: str) -> list[dict]:
    """Load all scored JSONL files from the directory."""
    files = [
        "gpt5mini_scored.jsonl",
        "gpt52_scored.jsonl",
        "qwen35_scored.jsonl",
    ]
    records = []
    for fname in files:
        path = os.path.join(scored_dir, fname)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def compute_difficulty(records: list[dict]) -> dict[str, float]:
    """Compute difficulty for each unique question ID.

    difficulty = 1 - mean(is_correct) across all target models.
    """
    question_correct = defaultdict(list)
    for r in records:
        question_correct[r["id"]].append(r["is_correct"])

    difficulty = {}
    for qid, corrects in question_correct.items():
        difficulty[qid] = 1.0 - np.mean(corrects)
    return difficulty


def safe_auroc(y_true, y_score, label: str = "") -> float | None:
    """Compute AUROC, returning None if undefined (single class)."""
    # Convert to float, treating None as NaN
    y_true_clean = []
    y_score_clean = []
    for yt, ys in zip(y_true, y_score):
        if ys is None or yt is None:
            continue
        try:
            ys_f = float(ys)
            yt_f = float(yt)
        except (TypeError, ValueError):
            continue
        if np.isnan(ys_f) or np.isnan(yt_f):
            continue
        y_true_clean.append(yt_f)
        y_score_clean.append(ys_f)

    if len(y_true_clean) < 2:
        return None
    y_true_arr = np.array(y_true_clean)
    y_score_arr = np.array(y_score_clean)
    unique = np.unique(y_true_arr)
    if len(unique) < 2:
        return None
    try:
        return float(roc_auc_score(y_true_arr, y_score_arr))
    except ValueError:
        return None


def compute_band_metrics(records: list[dict], band_name: str) -> dict:
    """Compute metrics for a group of records."""
    if not records:
        return {"band": band_name, "n_samples": 0}

    y_true = [r["is_correct"] for r in records]
    n = len(y_true)
    base_acc = float(np.mean(y_true))

    # Calibrator
    auroc_calibrator = safe_auroc(y_true, [r["p_correct"] for r in records], "calibrator")

    # Baselines
    auroc_verbalized = safe_auroc(
        y_true,
        [r.get("verbalized_confidence", float("nan")) for r in records],
        "verbalized",
    )
    auroc_platt = safe_auroc(
        y_true,
        [r.get("p_platt_verbalized", float("nan")) for r in records],
        "platt",
    )
    auroc_combined = safe_auroc(
        y_true,
        [r.get("p_combined_baseline", float("nan")) for r in records],
        "combined",
    )
    auroc_zeroshot = safe_auroc(
        y_true,
        [r.get("p_zeroshot", float("nan")) for r in records],
        "zeroshot",
    )

    return {
        "band": band_name,
        "n_samples": n,
        "base_accuracy": round(base_acc, 4),
        "auroc_calibrator": round(auroc_calibrator, 4) if auroc_calibrator is not None else None,
        "auroc_verbalized": round(auroc_verbalized, 4) if auroc_verbalized is not None else None,
        "auroc_platt": round(auroc_platt, 4) if auroc_platt is not None else None,
        "auroc_combined": round(auroc_combined, 4) if auroc_combined is not None else None,
        "auroc_zeroshot": round(auroc_zeroshot, 4) if auroc_zeroshot is not None else None,
    }


def assign_3band(difficulty: float) -> str:
    """Assign to Easy / Medium / Hard."""
    if difficulty == 0.0:
        return "Easy (all correct)"
    elif difficulty == 1.0:
        return "Hard (all wrong)"
    else:
        return "Medium (mixed)"


def assign_quantile_band(difficulty: float, edges: list[float]) -> str:
    """Assign to one of 5 quantile bands given bin edges."""
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            # Last bin is inclusive on both sides
            if lo <= difficulty <= hi:
                return f"Q{i+1} [{lo:.2f}, {hi:.2f}]"
        else:
            if lo <= difficulty < hi:
                return f"Q{i+1} [{lo:.2f}, {hi:.2f})"
    # Fallback to last bin
    return f"Q{len(edges)-1} [{edges[-2]:.2f}, {edges[-1]:.2f}]"


def main():
    parser = argparse.ArgumentParser(description="Difficulty-stratified AUROC analysis")
    parser.add_argument(
        "--scored_dir",
        default="data/use_cases/scored_test_only",
        help="Directory with scored JSONL files",
    )
    parser.add_argument(
        "--output",
        default="data/use_cases/results_test_only/difficulty_stratification.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run on first 50 records per file for quick testing",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scored_dir = os.path.join(project_root, args.scored_dir) if not os.path.isabs(args.scored_dir) else args.scored_dir
    output_path = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output

    # --- Load data ---
    print(f"Loading scored data from {scored_dir}")
    records = load_scored_data(scored_dir)
    if not records:
        print("ERROR: No records loaded. Check scored_dir path.")
        sys.exit(1)

    if args.smoke_test:
        # Take first 50 from each target model
        by_model = defaultdict(list)
        for r in records:
            by_model[r["target_model"]].append(r)
        records = []
        for model, recs in by_model.items():
            records.extend(recs[:50])
        print(f"Smoke test: using {len(records)} records")

    print(f"Loaded {len(records)} total records")

    # --- Compute difficulty ---
    difficulty = compute_difficulty(records)
    print(f"Unique questions: {len(difficulty)}")

    # Attach difficulty to each record
    for r in records:
        r["difficulty"] = difficulty[r["id"]]

    # --- 3-band stratification ---
    print("\n=== 3-Band Difficulty Stratification ===")
    bands_3 = defaultdict(list)
    for r in records:
        band = assign_3band(r["difficulty"])
        bands_3[band].append(r)

    results_3band = []
    band_order = ["Easy (all correct)", "Medium (mixed)", "Hard (all wrong)"]
    for band_name in band_order:
        recs = bands_3.get(band_name, [])
        metrics = compute_band_metrics(recs, band_name)
        results_3band.append(metrics)
        auroc_str = f"{metrics['auroc_calibrator']:.4f}" if metrics["auroc_calibrator"] is not None else "N/A"
        print(
            f"  {band_name:25s}  n={metrics['n_samples']:5d}  "
            f"acc={metrics['base_accuracy']:.3f}  "
            f"AUROC(cal)={auroc_str}"
        )

    # --- 5-quantile stratification ---
    print("\n=== 5-Quantile Difficulty Stratification ===")
    all_difficulties = np.array([r["difficulty"] for r in records])
    # Compute quantile edges (0%, 20%, 40%, 60%, 80%, 100%)
    try:
        edges = list(np.quantile(all_difficulties, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    except Exception:
        edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Deduplicate edges to avoid empty bins (common with discrete difficulty values)
    unique_edges = sorted(set(edges))
    if len(unique_edges) < 3:
        # If difficulty is too concentrated, use fixed edges
        unique_edges = [0.0, 0.25, 0.5, 0.75, 1.0]
        print("  (Using fixed edges due to concentrated difficulty distribution)")

    # Assign to quantile bands
    bands_q = defaultdict(list)
    for r in records:
        band = assign_quantile_band(r["difficulty"], unique_edges)
        bands_q[band].append(r)

    results_5quantile = []
    for band_name in sorted(bands_q.keys()):
        recs = bands_q[band_name]
        metrics = compute_band_metrics(recs, band_name)
        results_5quantile.append(metrics)
        auroc_str = f"{metrics['auroc_calibrator']:.4f}" if metrics["auroc_calibrator"] is not None else "N/A"
        print(
            f"  {band_name:30s}  n={metrics['n_samples']:5d}  "
            f"acc={metrics['base_accuracy']:.3f}  "
            f"AUROC(cal)={auroc_str}"
        )

    # --- Per-benchmark difficulty analysis ---
    print("\n=== Per-Benchmark Difficulty ===")
    bench_records = defaultdict(list)
    for r in records:
        bench_records[r["benchmark"]].append(r)

    per_benchmark = []
    for bench in sorted(bench_records.keys()):
        recs = bench_records[bench]
        bench_difficulty = np.mean([r["difficulty"] for r in recs])
        bench_acc = np.mean([r["is_correct"] for r in recs])
        metrics = compute_band_metrics(recs, bench)
        metrics["mean_difficulty"] = round(float(bench_difficulty), 4)
        per_benchmark.append(metrics)

        auroc_str = f"{metrics['auroc_calibrator']:.4f}" if metrics["auroc_calibrator"] is not None else "N/A"
        print(
            f"  {bench:25s}  n={metrics['n_samples']:4d}  "
            f"diff={bench_difficulty:.3f}  "
            f"acc={bench_acc:.3f}  "
            f"AUROC(cal)={auroc_str}"
        )

    # Sort by difficulty
    per_benchmark.sort(key=lambda x: x.get("mean_difficulty", 0), reverse=True)

    # --- Spearman correlation: difficulty vs calibrator confidence ---
    print("\n=== Difficulty-Confidence Correlation ===")
    difficulties_arr = np.array([r["difficulty"] for r in records])
    confidences_arr = np.array([r["p_correct"] for r in records])

    # Remove any NaN
    mask = ~(np.isnan(difficulties_arr) | np.isnan(confidences_arr))
    if mask.sum() >= 3:
        rho, pval = stats.spearmanr(difficulties_arr[mask], confidences_arr[mask])
        print(f"  Spearman rho (difficulty vs p_correct): {rho:.4f}  (p={pval:.2e})")
        correlation = {
            "spearman_rho": round(float(rho), 4),
            "spearman_pvalue": float(pval),
            "n_samples": int(mask.sum()),
        }
    else:
        print("  Not enough samples for correlation")
        correlation = {"spearman_rho": None, "spearman_pvalue": None, "n_samples": int(mask.sum())}

    # Also compute per-question correlation (one point per question, not per record)
    question_data = defaultdict(lambda: {"difficulties": [], "confidences": []})
    for r in records:
        qid = r["id"]
        question_data[qid]["difficulties"].append(r["difficulty"])
        question_data[qid]["confidences"].append(r["p_correct"])

    q_difficulties = []
    q_confidences = []
    for qid, d in question_data.items():
        q_difficulties.append(np.mean(d["difficulties"]))
        q_confidences.append(np.mean(d["confidences"]))

    q_difficulties = np.array(q_difficulties)
    q_confidences = np.array(q_confidences)
    mask_q = ~(np.isnan(q_difficulties) | np.isnan(q_confidences))
    if mask_q.sum() >= 3:
        rho_q, pval_q = stats.spearmanr(q_difficulties[mask_q], q_confidences[mask_q])
        print(f"  Spearman rho (per-question):              {rho_q:.4f}  (p={pval_q:.2e})")
        correlation["per_question_spearman_rho"] = round(float(rho_q), 4)
        correlation["per_question_spearman_pvalue"] = float(pval_q)
        correlation["n_questions"] = int(mask_q.sum())
    else:
        correlation["per_question_spearman_rho"] = None
        correlation["per_question_spearman_pvalue"] = None
        correlation["n_questions"] = int(mask_q.sum())

    # --- Difficulty distribution summary ---
    difficulty_values = list(difficulty.values())
    dist_summary = {
        "n_questions": len(difficulty_values),
        "mean": round(float(np.mean(difficulty_values)), 4),
        "median": round(float(np.median(difficulty_values)), 4),
        "std": round(float(np.std(difficulty_values)), 4),
        "min": round(float(np.min(difficulty_values)), 4),
        "max": round(float(np.max(difficulty_values)), 4),
        "pct_easy": round(float(np.mean(np.array(difficulty_values) == 0.0)), 4),
        "pct_hard": round(float(np.mean(np.array(difficulty_values) == 1.0)), 4),
    }
    print(f"\n=== Difficulty Distribution ===")
    print(f"  Mean difficulty: {dist_summary['mean']:.3f} +/- {dist_summary['std']:.3f}")
    print(f"  Easy (all correct): {dist_summary['pct_easy']*100:.1f}%")
    print(f"  Hard (all wrong):   {dist_summary['pct_hard']*100:.1f}%")

    # --- Assemble output ---
    output = {
        "description": (
            "AUROC stratified by question difficulty. "
            "Difficulty = 1 - mean(is_correct) across target models per question."
        ),
        "n_total_records": len(records),
        "n_unique_questions": len(difficulty),
        "difficulty_distribution": dist_summary,
        "three_band_stratification": results_3band,
        "five_quantile_stratification": results_5quantile,
        "quantile_edges": [round(e, 4) for e in unique_edges],
        "per_benchmark_difficulty": per_benchmark,
        "difficulty_confidence_correlation": correlation,
    }

    # --- Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
