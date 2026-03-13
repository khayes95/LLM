#!/usr/bin/env python3
"""Deep per-benchmark analysis of v3 test-only results."""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from scipy import stats

SCORED_DIR = "/scratch/khayes/LLM/data/use_cases/scored_test_only_v3"
OUTPUT_PATH = "/scratch/khayes/LLM/data/use_cases/results_test_only_v3/per_benchmark_deep_analysis.json"

def load_all_samples():
    """Load all scored samples from all 3 source model files."""
    samples = []
    for fname in ["gpt5mini_scored.jsonl", "gpt52_scored.jsonl", "qwen35_scored.jsonl"]:
        path = os.path.join(SCORED_DIR, fname)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                s = json.loads(line)
                samples.append(s)
    return samples

def auroc(labels, scores):
    """Compute AUROC. Returns None if only one class present."""
    labels = np.array(labels)
    scores = np.array(scores)
    if len(np.unique(labels)) < 2:
        return None
    # Manual AUROC via Wilcoxon-Mann-Whitney
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # Use scipy rankdata for ties
    from scipy.stats import rankdata
    all_scores = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    ranks = rankdata(all_scores)
    pos_rank_sum = ranks[:len(pos)].sum()
    u = pos_rank_sum - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))

def bootstrap_auroc(labels, scores, n_boot=2000, seed=42):
    """Bootstrap 95% CI for AUROC."""
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    scores = np.array(scores)
    n = len(labels)
    aurocs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        a = auroc(labels[idx], scores[idx])
        if a is not None:
            aurocs.append(a)
    if len(aurocs) < 100:
        return None, None
    return float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))

def analyze():
    print("Loading samples...")
    samples = load_all_samples()
    print(f"Total samples: {len(samples)}")

    # Group by benchmark
    by_bench = defaultdict(list)
    for s in samples:
        by_bench[s["benchmark"]].append(s)

    # Also group by (benchmark, target_model)
    by_bench_model = defaultdict(list)
    for s in samples:
        by_bench_model[(s["benchmark"], s["target_model"])].append(s)

    results = {}

    for bench in sorted(by_bench.keys()):
        samps = by_bench[bench]
        labels = [s["is_correct"] for s in samps]
        scores = [s["p_correct"] for s in samps]

        labels_arr = np.array(labels)
        scores_arr = np.array(scores)

        n = len(samps)
        n_correct = int(labels_arr.sum())
        n_incorrect = n - n_correct
        accuracy = n_correct / n if n > 0 else 0

        # AUROC
        auc = auroc(labels, scores)
        ci_lo, ci_hi = bootstrap_auroc(labels, scores)

        # Score distributions for correct vs incorrect
        correct_scores = scores_arr[labels_arr == 1]
        incorrect_scores = scores_arr[labels_arr == 0]

        score_stats = {
            "correct": {
                "n": int(len(correct_scores)),
                "mean": float(correct_scores.mean()) if len(correct_scores) > 0 else None,
                "std": float(correct_scores.std()) if len(correct_scores) > 0 else None,
                "median": float(np.median(correct_scores)) if len(correct_scores) > 0 else None,
                "q25": float(np.percentile(correct_scores, 25)) if len(correct_scores) > 0 else None,
                "q75": float(np.percentile(correct_scores, 75)) if len(correct_scores) > 0 else None,
            },
            "incorrect": {
                "n": int(len(incorrect_scores)),
                "mean": float(incorrect_scores.mean()) if len(incorrect_scores) > 0 else None,
                "std": float(incorrect_scores.std()) if len(incorrect_scores) > 0 else None,
                "median": float(np.median(incorrect_scores)) if len(incorrect_scores) > 0 else None,
                "q25": float(np.percentile(incorrect_scores, 25)) if len(incorrect_scores) > 0 else None,
                "q75": float(np.percentile(incorrect_scores, 75)) if len(incorrect_scores) > 0 else None,
            },
        }

        # Separation: Cohen's d between correct and incorrect score distributions
        if len(correct_scores) > 1 and len(incorrect_scores) > 1:
            pooled_std = np.sqrt(
                ((len(correct_scores)-1)*correct_scores.std()**2 +
                 (len(incorrect_scores)-1)*incorrect_scores.std()**2) /
                (len(correct_scores) + len(incorrect_scores) - 2)
            )
            cohens_d = (correct_scores.mean() - incorrect_scores.mean()) / pooled_std if pooled_std > 0 else None
        else:
            cohens_d = None

        # Per-model breakdown
        per_model = {}
        for model in ["gpt5mini", "gpt52", "qwen35"]:
            key = (bench, model)
            if key not in by_bench_model:
                continue
            msamps = by_bench_model[key]
            ml = [s["is_correct"] for s in msamps]
            ms = [s["p_correct"] for s in msamps]
            mauc = auroc(ml, ms)
            macc = sum(ml) / len(ml) if ml else 0
            per_model[model] = {
                "n": len(msamps),
                "accuracy": round(macc, 4),
                "auroc": round(mauc, 4) if mauc is not None else None,
            }

        # Has images?
        has_image_count = sum(1 for s in samps if s.get("has_image", False))
        modality = "VLM" if has_image_count > len(samps) * 0.5 else "Text"

        # Failure analysis: find misranked examples (high score but incorrect, or low score but correct)
        # "Confident wrong": incorrect but p_correct > 0.8
        confident_wrong = [s for s in samps if s["is_correct"] == 0 and s["p_correct"] > 0.8]
        # "Underconfident right": correct but p_correct < 0.2
        underconfident_right = [s for s in samps if s["is_correct"] == 1 and s["p_correct"] < 0.2]

        # Example IDs for worst failures
        confident_wrong_examples = sorted(confident_wrong, key=lambda x: -x["p_correct"])[:3]
        underconfident_right_examples = sorted(underconfident_right, key=lambda x: x["p_correct"])[:3]

        failure_analysis = {
            "n_confident_wrong": len(confident_wrong),
            "pct_confident_wrong": round(len(confident_wrong) / n_incorrect * 100, 1) if n_incorrect > 0 else 0,
            "n_underconfident_right": len(underconfident_right),
            "pct_underconfident_right": round(len(underconfident_right) / n_correct * 100, 1) if n_correct > 0 else 0,
            "confident_wrong_examples": [
                {"id": s["id"], "p_correct": round(s["p_correct"], 4),
                 "question_preview": s.get("question_preview", "")[:200],
                 "response_preview": s.get("response_preview", "")[:200]}
                for s in confident_wrong_examples
            ],
            "underconfident_right_examples": [
                {"id": s["id"], "p_correct": round(s["p_correct"], 4),
                 "question_preview": s.get("question_preview", "")[:200],
                 "response_preview": s.get("response_preview", "")[:200]}
                for s in underconfident_right_examples
            ],
        }

        results[bench] = {
            "n_samples": n,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "accuracy": round(accuracy, 4),
            "auroc": round(auc, 4) if auc is not None else None,
            "auroc_ci_95": [round(ci_lo, 4), round(ci_hi, 4)] if ci_lo is not None else None,
            "modality": modality,
            "cohens_d": round(cohens_d, 4) if cohens_d is not None else None,
            "score_distributions": score_stats,
            "per_model": per_model,
            "failure_analysis": failure_analysis,
        }

    # === Global statistics ===
    all_benchmarks = sorted(results.keys())
    aurocs = [(b, results[b]["auroc"]) for b in all_benchmarks if results[b]["auroc"] is not None]
    accuracies = [(b, results[b]["accuracy"]) for b in all_benchmarks]

    # Sort by AUROC
    aurocs_sorted = sorted(aurocs, key=lambda x: x[1])

    # Correlation between accuracy and AUROC
    acc_vals = []
    auc_vals = []
    for b in all_benchmarks:
        if results[b]["auroc"] is not None:
            acc_vals.append(results[b]["accuracy"])
            auc_vals.append(results[b]["auroc"])

    if len(acc_vals) > 3:
        pearson_r, pearson_p = stats.pearsonr(acc_vals, auc_vals)
        spearman_r, spearman_p = stats.spearmanr(acc_vals, auc_vals)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = None

    # Correlation between sample size and AUROC
    n_vals = [results[b]["n_samples"] for b in all_benchmarks if results[b]["auroc"] is not None]
    if len(n_vals) > 3:
        size_spearman_r, size_spearman_p = stats.spearmanr(n_vals, auc_vals)
    else:
        size_spearman_r = size_spearman_p = None

    # Mean/median AUROC
    auc_values = [v for _, v in aurocs]

    summary = {
        "total_samples": len(samples),
        "n_benchmarks": len(all_benchmarks),
        "mean_auroc": round(np.mean(auc_values), 4),
        "median_auroc": round(np.median(auc_values), 4),
        "std_auroc": round(np.std(auc_values), 4),
        "min_auroc": {"benchmark": aurocs_sorted[0][0], "auroc": round(aurocs_sorted[0][1], 4)},
        "max_auroc": {"benchmark": aurocs_sorted[-1][0], "auroc": round(aurocs_sorted[-1][1], 4)},
        "bottom_5": [{"benchmark": b, "auroc": round(a, 4)} for b, a in aurocs_sorted[:5]],
        "top_5": [{"benchmark": b, "auroc": round(a, 4)} for b, a in aurocs_sorted[-5:]],
        "accuracy_vs_auroc_correlation": {
            "pearson_r": round(pearson_r, 4) if pearson_r is not None else None,
            "pearson_p": round(pearson_p, 4) if pearson_p is not None else None,
            "spearman_r": round(spearman_r, 4) if spearman_r is not None else None,
            "spearman_p": round(spearman_p, 4) if spearman_p is not None else None,
            "interpretation": None,  # filled below
        },
        "sample_size_vs_auroc_correlation": {
            "spearman_r": round(size_spearman_r, 4) if size_spearman_r is not None else None,
            "spearman_p": round(size_spearman_p, 4) if size_spearman_p is not None else None,
        },
    }

    # Interpretation
    if pearson_r is not None:
        if pearson_r > 0.3:
            summary["accuracy_vs_auroc_correlation"]["interpretation"] = (
                f"Positive correlation (r={pearson_r:.3f}): calibrator works BETTER on easier benchmarks"
            )
        elif pearson_r < -0.3:
            summary["accuracy_vs_auroc_correlation"]["interpretation"] = (
                f"Negative correlation (r={pearson_r:.3f}): calibrator works BETTER on harder benchmarks"
            )
        else:
            summary["accuracy_vs_auroc_correlation"]["interpretation"] = (
                f"Weak correlation (r={pearson_r:.3f}): no strong relationship between difficulty and calibrator performance"
            )

    # === Worst benchmark deep dive ===
    worst_benchmarks_analysis = {}
    for bench, auc_val in aurocs_sorted[:5]:
        b_data = results[bench]
        # Analyze WHY it's hard
        reasons = []

        # Check class imbalance
        minority_pct = min(b_data["accuracy"], 1 - b_data["accuracy"])
        if minority_pct < 0.15:
            reasons.append(f"Severe class imbalance: {b_data['accuracy']:.1%} accuracy (minority class {minority_pct:.1%})")
        elif minority_pct < 0.30:
            reasons.append(f"Moderate class imbalance: {b_data['accuracy']:.1%} accuracy")

        # Check score separation
        if b_data["cohens_d"] is not None:
            if abs(b_data["cohens_d"]) < 0.5:
                reasons.append(f"Poor score separation (Cohen's d={b_data['cohens_d']:.2f}): correct and incorrect scores highly overlapping")
            elif abs(b_data["cohens_d"]) < 1.0:
                reasons.append(f"Moderate score separation (Cohen's d={b_data['cohens_d']:.2f})")

        # Check confident wrong rate
        fa = b_data["failure_analysis"]
        if fa["pct_confident_wrong"] > 15:
            reasons.append(f"High confident-wrong rate: {fa['pct_confident_wrong']:.1f}% of incorrect answers scored >0.8")
        if fa["pct_underconfident_right"] > 15:
            reasons.append(f"High underconfident-right rate: {fa['pct_underconfident_right']:.1f}% of correct answers scored <0.2")

        # Small sample
        if b_data["n_samples"] < 50:
            reasons.append(f"Very small sample size: {b_data['n_samples']} (AUROC unreliable)")
        elif b_data["n_samples"] < 100:
            reasons.append(f"Small sample size: {b_data['n_samples']}")

        worst_benchmarks_analysis[bench] = {
            "auroc": round(auc_val, 4),
            "n_samples": b_data["n_samples"],
            "accuracy": b_data["accuracy"],
            "cohens_d": b_data["cohens_d"],
            "likely_reasons": reasons,
        }

    output = {
        "summary": summary,
        "worst_benchmarks_deep_dive": worst_benchmarks_analysis,
        "per_benchmark": results,
    }

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'Benchmark':<20} {'N':>5} {'Acc':>7} {'AUROC':>7} {'CI_lo':>7} {'CI_hi':>7} {'d':>7} {'Mod':<5}")
    print(f"{'='*90}")
    for bench in sorted(all_benchmarks, key=lambda b: results[b]["auroc"] if results[b]["auroc"] is not None else 0):
        r = results[bench]
        ci = r["auroc_ci_95"]
        print(f"{bench:<20} {r['n_samples']:>5} {r['accuracy']:>7.3f} "
              f"{r['auroc']:>7.3f} "
              f"{ci[0] if ci else 'N/A':>7} {ci[1] if ci else 'N/A':>7} "
              f"{r['cohens_d'] if r['cohens_d'] is not None else 'N/A':>7} "
              f"{r['modality']:<5}")

    print(f"\n{'='*90}")
    print(f"Mean AUROC: {summary['mean_auroc']:.4f} | Median: {summary['median_auroc']:.4f} | Std: {summary['std_auroc']:.4f}")
    print(f"Accuracy vs AUROC: Pearson r={pearson_r:.3f} (p={pearson_p:.4f}), Spearman r={spearman_r:.3f} (p={spearman_p:.4f})")
    if size_spearman_r is not None:
        print(f"Sample size vs AUROC: Spearman r={size_spearman_r:.3f} (p={size_spearman_p:.4f})")

    print(f"\nWorst 5: {[f'{b} ({a:.3f})' for b,a in aurocs_sorted[:5]]}")
    print(f"Best 5:  {[f'{b} ({a:.3f})' for b,a in aurocs_sorted[-5:]]}")

    # Worst benchmark reasons
    print(f"\n{'='*90}")
    print("WORST BENCHMARK DEEP DIVE")
    print(f"{'='*90}")
    for bench, info in worst_benchmarks_analysis.items():
        print(f"\n--- {bench} (AUROC={info['auroc']:.3f}, N={info['n_samples']}, Acc={info['accuracy']:.3f}) ---")
        for reason in info["likely_reasons"]:
            print(f"  * {reason}")
        # Show failure examples
        fa = results[bench]["failure_analysis"]
        if fa["confident_wrong_examples"]:
            print(f"  Confident-wrong examples ({fa['n_confident_wrong']} total):")
            for ex in fa["confident_wrong_examples"][:2]:
                print(f"    ID: {ex['id']}, p={ex['p_correct']:.3f}")
                print(f"    Q: {ex['question_preview'][:100]}...")
        if fa["underconfident_right_examples"]:
            print(f"  Underconfident-right examples ({fa['n_underconfident_right']} total):")
            for ex in fa["underconfident_right_examples"][:2]:
                print(f"    ID: {ex['id']}, p={ex['p_correct']:.3f}")
                print(f"    Q: {ex['question_preview'][:100]}...")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    analyze()
