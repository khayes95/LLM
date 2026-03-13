#!/usr/bin/env python3
"""Cross-model transfer analysis for v3 scored test data."""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr

def load_scored(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def auroc(labels, scores):
    """Compute AUROC. Returns NaN if fewer than 2 classes."""
    labels = np.array(labels)
    scores = np.array(scores)
    if len(set(labels)) < 2:
        return float('nan')
    # Wilcoxon-Mann-Whitney
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    # Vectorized
    return np.mean(np.array([[float(p > n) + 0.5 * float(p == n) for n in neg] for p in pos]))

def main():
    base = "/scratch/khayes/LLM/data/use_cases/scored_test_only_v3"
    out_dir = "/scratch/khayes/LLM/data/use_cases/results_test_only_v3"
    os.makedirs(out_dir, exist_ok=True)

    files = {
        "gpt5mini": os.path.join(base, "gpt5mini_scored.jsonl"),
        "gpt52": os.path.join(base, "gpt52_scored.jsonl"),
        "qwen35": os.path.join(base, "qwen35_scored.jsonl"),
    }

    # Load all data
    all_data = {}
    for model, path in files.items():
        all_data[model] = load_scored(path)
        print(f"Loaded {len(all_data[model])} samples for {model}")

    models = list(all_data.keys())

    # =========================================================================
    # 1. Per-model, per-benchmark AUROC matrix
    # =========================================================================
    # Group by (model, benchmark)
    grouped = defaultdict(lambda: {"labels": [], "scores": []})
    for model, records in all_data.items():
        for r in records:
            key = (model, r["benchmark"])
            grouped[key]["labels"].append(r["is_correct"])
            grouped[key]["scores"].append(r["p_correct"])

    # Get all benchmarks
    all_benchmarks = sorted(set(b for _, b in grouped.keys()))
    print(f"\nBenchmarks ({len(all_benchmarks)}): {all_benchmarks}")

    auroc_matrix = {}  # model -> benchmark -> auroc
    sample_counts = {}  # model -> benchmark -> n
    for model in models:
        auroc_matrix[model] = {}
        sample_counts[model] = {}
        for bench in all_benchmarks:
            key = (model, bench)
            if key in grouped and len(grouped[key]["labels"]) >= 5:
                val = auroc(grouped[key]["labels"], grouped[key]["scores"])
                auroc_matrix[model][bench] = round(val, 4) if not np.isnan(val) else None
                sample_counts[model][bench] = len(grouped[key]["labels"])
            else:
                auroc_matrix[model][bench] = None
                sample_counts[model][bench] = len(grouped[key]["labels"]) if key in grouped else 0

    # Print matrix
    print("\n=== Per-Model, Per-Benchmark AUROC ===")
    header = f"{'Benchmark':<25}" + "".join(f"{m:>12}" for m in models) + f"{'  Range':>10}" + f"{'  N(min)':>8}"
    print(header)
    print("-" * len(header))

    # =========================================================================
    # 2 & 3. Cross-model consistency and divergence
    # =========================================================================
    bench_stats = []
    for bench in all_benchmarks:
        vals = []
        row = f"{bench:<25}"
        for model in models:
            v = auroc_matrix[model][bench]
            if v is not None:
                row += f"{v:>12.3f}"
                vals.append(v)
            else:
                row += f"{'N/A':>12}"

        if len(vals) >= 2:
            rng = max(vals) - min(vals)
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            min_n = min(sample_counts[m].get(bench, 0) for m in models if auroc_matrix[m][bench] is not None)
            row += f"{rng:>10.3f}" + f"{min_n:>8}"
            bench_stats.append({
                "benchmark": bench,
                "aurocs": {m: auroc_matrix[m][bench] for m in models},
                "mean": round(float(mean_v), 4),
                "std": round(float(std_v), 4),
                "range": round(float(rng), 4),
                "n_models": len(vals),
                "min_samples": min_n,
            })
        else:
            row += f"{'---':>10}" + f"{'---':>8}"
        print(row)

    # Sort by range for consistency / divergence
    bench_stats_valid = [b for b in bench_stats if b["n_models"] >= 2 and b["min_samples"] >= 10]
    bench_stats_valid.sort(key=lambda x: x["range"])

    print("\n=== Most Consistent Benchmarks (smallest AUROC range across models) ===")
    for b in bench_stats_valid[:5]:
        print(f"  {b['benchmark']:<25} range={b['range']:.3f}  mean={b['mean']:.3f}  aurocs={b['aurocs']}")

    print("\n=== Most Divergent Benchmarks (largest AUROC range across models) ===")
    for b in bench_stats_valid[-5:]:
        print(f"  {b['benchmark']:<25} range={b['range']:.3f}  mean={b['mean']:.3f}  aurocs={b['aurocs']}")

    # =========================================================================
    # 4. Score correlation for shared questions
    # =========================================================================
    # Build question_id -> {model: p_correct}
    question_scores = defaultdict(dict)
    question_labels = defaultdict(dict)
    for model, records in all_data.items():
        for r in records:
            qid = r["id"]
            question_scores[qid][model] = r["p_correct"]
            question_labels[qid][model] = r["is_correct"]

    # Find questions shared across model pairs
    print("\n=== Score Correlation for Shared Questions ===")
    model_pairs = [("gpt5mini", "gpt52"), ("gpt5mini", "qwen35"), ("gpt52", "qwen35")]
    correlation_results = {}
    for m1, m2 in model_pairs:
        shared_ids = [qid for qid in question_scores if m1 in question_scores[qid] and m2 in question_scores[qid]]
        if len(shared_ids) < 10:
            print(f"  {m1} vs {m2}: Only {len(shared_ids)} shared questions, skipping")
            correlation_results[f"{m1}_vs_{m2}"] = {"n_shared": len(shared_ids), "pearson_r": None, "spearman_r": None}
            continue

        s1 = np.array([question_scores[qid][m1] for qid in shared_ids])
        s2 = np.array([question_scores[qid][m2] for qid in shared_ids])
        l1 = np.array([question_labels[qid][m1] for qid in shared_ids])
        l2 = np.array([question_labels[qid][m2] for qid in shared_ids])

        pr, pp = pearsonr(s1, s2)
        sr, sp = spearmanr(s1, s2)

        # Agreement: both correct or both incorrect
        agree = np.mean(l1 == l2)
        # Calibrator agreement: both high or both low (threshold 0.5)
        cal_agree = np.mean((s1 > 0.5) == (s2 > 0.5))

        print(f"  {m1} vs {m2}: N={len(shared_ids)}")
        print(f"    Pearson r={pr:.3f} (p={pp:.2e}), Spearman rho={sr:.3f} (p={sp:.2e})")
        print(f"    Label agreement: {agree:.1%}, Calibrator prediction agreement (>0.5): {cal_agree:.1%}")

        correlation_results[f"{m1}_vs_{m2}"] = {
            "n_shared": len(shared_ids),
            "pearson_r": round(float(pr), 4),
            "pearson_p": float(pp),
            "spearman_rho": round(float(sr), 4),
            "spearman_p": float(sp),
            "label_agreement": round(float(agree), 4),
            "calibrator_prediction_agreement": round(float(cal_agree), 4),
        }

    # =========================================================================
    # 5. Benchmarks where calibrator works great on one model but poorly on another
    # =========================================================================
    print("\n=== Benchmarks with Large Model-Specific Performance Gaps ===")
    gap_analysis = []
    for b in bench_stats_valid:
        if b["range"] >= 0.10:  # At least 10 point gap
            best_model = max(b["aurocs"], key=lambda m: b["aurocs"][m] if b["aurocs"][m] is not None else -1)
            worst_model = min(b["aurocs"], key=lambda m: b["aurocs"][m] if b["aurocs"][m] is not None else 999)
            best_val = b["aurocs"][best_model]
            worst_val = b["aurocs"][worst_model]
            gap_analysis.append({
                "benchmark": b["benchmark"],
                "best_model": best_model,
                "best_auroc": best_val,
                "worst_model": worst_model,
                "worst_auroc": worst_val,
                "gap": round(best_val - worst_val, 4) if best_val and worst_val else None,
                "all_aurocs": b["aurocs"],
            })
            print(f"  {b['benchmark']:<25} best: {best_model}={best_val:.3f}, worst: {worst_model}={worst_val:.3f}, gap={best_val-worst_val:.3f}")

    if not gap_analysis:
        print("  No benchmarks with >= 0.10 AUROC gap found.")

    # =========================================================================
    # Per-model overall AUROC
    # =========================================================================
    print("\n=== Per-Model Overall AUROC ===")
    overall_aurocs = {}
    for model in models:
        all_labels = [r["is_correct"] for r in all_data[model]]
        all_scores = [r["p_correct"] for r in all_data[model]]
        a = auroc(all_labels, all_scores)
        overall_aurocs[model] = round(a, 4)
        n_correct = sum(all_labels)
        print(f"  {model}: AUROC={a:.4f}  N={len(all_labels)}  accuracy={n_correct/len(all_labels):.1%}")

    # =========================================================================
    # Per-benchmark accuracy by model (helps explain AUROC differences)
    # =========================================================================
    print("\n=== Per-Benchmark Accuracy by Model ===")
    acc_header = f"{'Benchmark':<25}" + "".join(f"{m:>12}" for m in models)
    print(acc_header)
    print("-" * len(acc_header))
    accuracy_matrix = {}
    for model in models:
        accuracy_matrix[model] = {}
        for bench in all_benchmarks:
            key = (model, bench)
            if key in grouped and len(grouped[key]["labels"]) >= 5:
                acc = np.mean(grouped[key]["labels"])
                accuracy_matrix[model][bench] = round(float(acc), 4)
            else:
                accuracy_matrix[model][bench] = None

    for bench in all_benchmarks:
        row = f"{bench:<25}"
        for model in models:
            v = accuracy_matrix[model][bench]
            if v is not None:
                row += f"{v:>12.1%}"
            else:
                row += f"{'N/A':>12}"
        print(row)

    # =========================================================================
    # Save results
    # =========================================================================
    results = {
        "description": "Cross-model transfer analysis for v3 scored test data (question-level split)",
        "overall_aurocs": overall_aurocs,
        "auroc_matrix": auroc_matrix,
        "sample_counts": sample_counts,
        "accuracy_matrix": accuracy_matrix,
        "benchmark_consistency": sorted(bench_stats_valid, key=lambda x: x["range"]),
        "most_consistent_top5": [b["benchmark"] for b in bench_stats_valid[:5]],
        "most_divergent_top5": [b["benchmark"] for b in bench_stats_valid[-5:]],
        "score_correlations": correlation_results,
        "large_gap_benchmarks": gap_analysis,
    }

    out_path = os.path.join(out_dir, "cross_model_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
