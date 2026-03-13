#!/usr/bin/env python3
"""Analyze v3 calibrator performance stratified by difficulty level and confidence."""

import json
import os
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score

SCORED_DIR = "data/use_cases/scored_test_only_v3"
OUTPUT_DIR = "data/use_cases/results_test_only_v3"

def load_all_samples():
    samples = []
    for fname in ["gpt5mini_scored.jsonl", "gpt52_scored.jsonl", "qwen35_scored.jsonl"]:
        path = os.path.join(SCORED_DIR, fname)
        with open(path) as f:
            for line in f:
                samples.append(json.loads(line))
    return samples

def safe_auroc(labels, scores):
    """Compute AUROC if both classes present, else return None."""
    labels = np.array(labels)
    scores = np.array(scores)
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))

def main():
    samples = load_all_samples()
    print(f"Loaded {len(samples)} samples")

    # --- 1. Compute per-benchmark accuracy ---
    bench_correct = defaultdict(list)
    for s in samples:
        bench_correct[s["benchmark"]].append(s["is_correct"])

    bench_accuracy = {b: np.mean(v) for b, v in bench_correct.items()}
    print(f"\nBenchmarks: {len(bench_accuracy)}")
    for b in sorted(bench_accuracy, key=bench_accuracy.get):
        print(f"  {b}: acc={bench_accuracy[b]:.3f} (n={len(bench_correct[b])})")

    # --- 2. Bin by difficulty tier ---
    easy_benchmarks = [b for b, a in bench_accuracy.items() if a > 0.65]
    medium_benchmarks = [b for b, a in bench_accuracy.items() if 0.35 <= a <= 0.65]
    hard_benchmarks = [b for b, a in bench_accuracy.items() if a < 0.35]

    print(f"\nEasy (>65% acc): {sorted(easy_benchmarks)}")
    print(f"Medium (35-65%): {sorted(medium_benchmarks)}")
    print(f"Hard (<35%):     {sorted(hard_benchmarks)}")

    tier_results = {}
    for tier_name, tier_benches in [("easy", easy_benchmarks), ("medium", medium_benchmarks), ("hard", hard_benchmarks)]:
        tier_samples = [s for s in samples if s["benchmark"] in tier_benches]
        labels = [s["is_correct"] for s in tier_samples]
        scores = [s["p_correct"] for s in tier_samples]
        auroc = safe_auroc(labels, scores)
        acc = np.mean(labels) if labels else None
        tier_results[tier_name] = {
            "benchmarks": sorted(tier_benches),
            "n_benchmarks": len(tier_benches),
            "n_samples": len(tier_samples),
            "accuracy": float(acc) if acc is not None else None,
            "auroc": auroc,
        }
        print(f"\n{tier_name.upper()} tier: {len(tier_samples)} samples, acc={acc:.3f}, AUROC={auroc}")

        # Per-benchmark AUROC within tier
        per_bench = {}
        for b in sorted(tier_benches):
            b_samples = [s for s in tier_samples if s["benchmark"] == b]
            b_labels = [s["is_correct"] for s in b_samples]
            b_scores = [s["p_correct"] for s in b_samples]
            b_auroc = safe_auroc(b_labels, b_scores)
            per_bench[b] = {
                "n": len(b_samples),
                "accuracy": float(np.mean(b_labels)),
                "auroc": b_auroc,
            }
            print(f"    {b}: n={len(b_samples)}, acc={np.mean(b_labels):.3f}, AUROC={b_auroc}")
        tier_results[tier_name]["per_benchmark"] = per_bench

    # --- 3. Confidence accuracy analysis ---
    print("\n--- Confidence Accuracy ---")
    high_conf = [s for s in samples if s["p_correct"] > 0.8 or s["p_correct"] < 0.2]
    med_conf = [s for s in samples if 0.3 <= s["p_correct"] <= 0.7]
    low_margin = [s for s in samples if 0.2 <= s["p_correct"] < 0.3 or 0.7 < s["p_correct"] <= 0.8]

    def confidence_accuracy(subset, label):
        """For high-conf predictions, check if the confident direction is correct."""
        if not subset:
            return {"n": 0}
        correct_direction = 0
        for s in subset:
            predicted_correct = s["p_correct"] >= 0.5
            actual_correct = s["is_correct"] == 1
            if predicted_correct == actual_correct:
                correct_direction += 1
        acc = correct_direction / len(subset)
        print(f"  {label}: n={len(subset)}, direction_accuracy={acc:.3f}")
        return {"n": len(subset), "direction_accuracy": float(acc)}

    conf_results = {
        "high_confidence_gt08_or_lt02": confidence_accuracy(high_conf, "High conf (>0.8 or <0.2)"),
        "medium_confidence_03_to_07": confidence_accuracy(med_conf, "Medium conf (0.3-0.7)"),
        "marginal_02_03_or_07_08": confidence_accuracy(low_margin, "Marginal (0.2-0.3 or 0.7-0.8)"),
    }

    # Also break high-conf into "confident correct" vs "confident incorrect"
    conf_correct = [s for s in samples if s["p_correct"] > 0.8]
    conf_incorrect = [s for s in samples if s["p_correct"] < 0.2]
    conf_results["confident_correct_gt08"] = {
        "n": len(conf_correct),
        "actual_accuracy": float(np.mean([s["is_correct"] for s in conf_correct])) if conf_correct else None,
    }
    conf_results["confident_incorrect_lt02"] = {
        "n": len(conf_incorrect),
        "actual_accuracy": float(np.mean([s["is_correct"] for s in conf_incorrect])) if conf_incorrect else None,
    }
    print(f"  Confident correct (>0.8): n={conf_results['confident_correct_gt08']['n']}, "
          f"actual_acc={conf_results['confident_correct_gt08']['actual_accuracy']:.3f}")
    print(f"  Confident incorrect (<0.2): n={conf_results['confident_incorrect_lt02']['n']}, "
          f"actual_acc={conf_results['confident_incorrect_lt02']['actual_accuracy']:.3f}")

    # --- 4. Reliability diagram (10 bins) ---
    print("\n--- Reliability Diagram ---")
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_data = []

    all_scores = np.array([s["p_correct"] for s in samples])
    all_labels = np.array([s["is_correct"] for s in samples])

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (all_scores >= lo) & (all_scores < hi)
        else:
            mask = (all_scores >= lo) & (all_scores <= hi)

        n_in_bin = int(mask.sum())
        if n_in_bin > 0:
            mean_predicted = float(all_scores[mask].mean())
            mean_actual = float(all_labels[mask].mean())
            gap = mean_predicted - mean_actual
        else:
            mean_predicted = None
            mean_actual = None
            gap = None

        bin_entry = {
            "bin_index": i,
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "n_samples": n_in_bin,
            "mean_predicted_probability": mean_predicted,
            "mean_actual_accuracy": mean_actual,
            "calibration_gap": gap,
        }
        bin_data.append(bin_entry)
        if n_in_bin > 0:
            print(f"  Bin [{lo:.1f}, {hi:.1f}): n={n_in_bin:4d}, pred={mean_predicted:.3f}, actual={mean_actual:.3f}, gap={gap:+.3f}")
        else:
            print(f"  Bin [{lo:.1f}, {hi:.1f}): n=0")

    # ECE
    ece = 0.0
    total = len(samples)
    for b in bin_data:
        if b["n_samples"] > 0:
            ece += (b["n_samples"] / total) * abs(b["calibration_gap"])
    print(f"\nECE = {ece:.4f}")

    # --- Save results ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Main analysis
    analysis = {
        "total_samples": len(samples),
        "n_benchmarks": len(bench_accuracy),
        "benchmark_accuracies": {b: float(v) for b, v in sorted(bench_accuracy.items())},
        "difficulty_tiers": tier_results,
        "confidence_accuracy": conf_results,
        "ece": float(ece),
    }

    out_path = os.path.join(OUTPUT_DIR, "difficulty_stratified_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis to {out_path}")

    # Reliability diagram data
    reliability = {
        "n_bins": n_bins,
        "total_samples": len(samples),
        "ece": float(ece),
        "bins": bin_data,
    }
    rel_path = os.path.join(OUTPUT_DIR, "reliability_diagram_data.json")
    with open(rel_path, "w") as f:
        json.dump(reliability, f, indent=2)
    print(f"Saved reliability diagram data to {rel_path}")


if __name__ == "__main__":
    main()
