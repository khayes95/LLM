#!/usr/bin/env python3
"""
CPU Feature Analysis for UQ Scored Data

Extracts text features from scored UQ samples and performs error analysis
to understand when the calibrator succeeds vs fails.

Features extracted:
  - Text statistics (lengths, word counts, sentence counts)
  - Content markers (numbers, hedge words, confidence language, code, JSON, lists)
  - Question type classification (MCQ, math, open-ended)
  - Token counts and verbalized confidence from scored data

Error analysis:
  - Mann-Whitney U tests per feature (correct vs wrong calibrator predictions)
  - Logistic regression to predict calibrator errors from features
  - Per-benchmark feature aggregation correlated with benchmark AUROC
  - Pearson correlation matrix between features and key outputs

Usage:
  python scripts/cpu_feature_analysis.py [--scored_dir DIR] [--output FILE] [--smoke_test]
"""

import argparse
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "likely", "unlikely", "probably",
    "approximately", "roughly", "about", "around", "could be", "might be",
    "uncertain", "not sure", "i think",
]

CONFIDENCE_WORDS = [
    "definitely", "certainly", "clearly", "obviously", "undoubtedly",
    "without doubt", "absolutely", "precisely", "exactly", "sure",
]

CODE_MARKERS = ["```", "def ", "class ", "import ", "function"]

_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_NUMBER_RE = re.compile(r"\d+\.?\d*")
_MCQ_RE = re.compile(r"\([A-E]\)")
_MATH_RE = re.compile(r"(\$.*?\$|\\frac|\\sum|\\int|\\sqrt|\\mathbb|\\text\{)")
_LIST_RE = re.compile(r"(?:^|\n)\s*(?:\d+[\.\)]\s|[-*]\s|\u2022)")


def _count_phrases(text_lower: str, phrases: list[str]) -> int:
    """Count total occurrences of all phrases in text."""
    return sum(text_lower.count(p) for p in phrases)


def extract_features(sample: dict) -> dict:
    """Extract all text / metadata features from a single scored sample."""
    question = sample.get("question_preview", "") or ""
    response = sample.get("response_preview", "") or ""
    response_lower = response.lower()

    # Basic text stats
    question_length = len(question)
    response_length = len(response)
    word_count_q = len(question.split())
    word_count_r = len(response.split())
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(response) if s.strip()]
    sentence_count_r = max(len(sentences), 1) if response else 0
    num_count_r = len(_NUMBER_RE.findall(response))

    # Semantic markers
    hedge_word_count = _count_phrases(response_lower, HEDGE_WORDS)
    confidence_language_count = _count_phrases(response_lower, CONFIDENCE_WORDS)

    # Content type markers
    has_json = int("{" in response and "}" in response)
    has_code = int(any(marker in response for marker in CODE_MARKERS))
    has_list = int(bool(_LIST_RE.search(response)))

    # Question type
    if _MCQ_RE.search(question):
        question_type = "mcq"
    elif _MATH_RE.search(question) or _MATH_RE.search(response):
        question_type = "math"
    else:
        question_type = "open-ended"

    # Token counts from scored data (may be None)
    input_tokens = sample.get("input_tokens")
    output_tokens = sample.get("output_tokens")
    if input_tokens is None:
        input_tokens = 0
    if output_tokens is None:
        output_tokens = 0

    verbalized_confidence = sample.get("verbalized_confidence")
    if verbalized_confidence is None:
        verbalized_confidence = 0.0

    return {
        "question_length": question_length,
        "response_length": response_length,
        "word_count_q": word_count_q,
        "word_count_r": word_count_r,
        "sentence_count_r": sentence_count_r,
        "num_count_r": num_count_r,
        "hedge_word_count": hedge_word_count,
        "confidence_language_count": confidence_language_count,
        "has_json": has_json,
        "has_code": has_code,
        "has_list": has_list,
        "question_type": question_type,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "verbalized_confidence": verbalized_confidence,
        # Pass through keys needed for analysis
        "benchmark": sample.get("benchmark", "unknown"),
        "target_model": sample.get("target_model", "unknown"),
        "is_correct": sample.get("is_correct", 0),
        "p_correct": sample.get("p_correct", 0.5),
        "has_image": sample.get("has_image", False),
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_scored_data(scored_dir: str, max_samples: int = 0) -> list[dict]:
    """Load all scored JSONL files from the directory."""
    scored_dir = Path(scored_dir)
    all_samples = []
    for path in sorted(scored_dir.glob("*_scored.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append(json.loads(line))
        print(f"  Loaded {path.name}: {sum(1 for s in all_samples if True)} cumulative samples")
    if max_samples > 0:
        all_samples = all_samples[:max_samples]
    print(f"  Total samples: {len(all_samples)}")
    return all_samples


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "question_length", "response_length", "word_count_q", "word_count_r",
    "sentence_count_r", "num_count_r", "hedge_word_count",
    "confidence_language_count", "has_json", "has_code", "has_list",
    "input_tokens", "output_tokens", "verbalized_confidence",
]


def compute_feature_stats(features_list: list[dict]) -> dict:
    """Compute per-feature stats split by calibrator correct vs wrong."""
    from scipy.stats import mannwhitneyu

    # Classify calibrator correctness
    correct_feats = []
    wrong_feats = []
    for f in features_list:
        predicted_correct = f["p_correct"] > 0.5
        actually_correct = f["is_correct"] == 1
        if predicted_correct == actually_correct:
            correct_feats.append(f)
        else:
            wrong_feats.append(f)

    n_correct = len(correct_feats)
    n_wrong = len(wrong_feats)
    print(f"  Calibrator correct: {n_correct}, wrong: {n_wrong} "
          f"({n_correct / (n_correct + n_wrong) * 100:.1f}% accuracy)")

    stats = {}
    for feat_name in NUMERIC_FEATURES:
        all_vals = np.array([f[feat_name] for f in features_list], dtype=float)
        correct_vals = np.array([f[feat_name] for f in correct_feats], dtype=float)
        wrong_vals = np.array([f[feat_name] for f in wrong_feats], dtype=float)

        mw_stat, mw_p = np.nan, np.nan
        if len(correct_vals) > 0 and len(wrong_vals) > 0:
            try:
                mw_stat, mw_p = mannwhitneyu(correct_vals, wrong_vals, alternative="two-sided")
            except Exception:
                pass

        stats[feat_name] = {
            "mean": float(np.mean(all_vals)),
            "std": float(np.std(all_vals)),
            "median": float(np.median(all_vals)),
            "correct_mean": float(np.mean(correct_vals)) if len(correct_vals) > 0 else None,
            "correct_std": float(np.std(correct_vals)) if len(correct_vals) > 0 else None,
            "wrong_mean": float(np.mean(wrong_vals)) if len(wrong_vals) > 0 else None,
            "wrong_std": float(np.std(wrong_vals)) if len(wrong_vals) > 0 else None,
            "mann_whitney_U": float(mw_stat) if not np.isnan(mw_stat) else None,
            "mann_whitney_p": float(mw_p) if not np.isnan(mw_p) else None,
        }

    return stats, n_correct, n_wrong


def run_logistic_regression(features_list: list[dict]) -> dict:
    """Logistic regression predicting calibrator error from features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    X = np.array([[f[feat] for feat in NUMERIC_FEATURES] for f in features_list], dtype=float)
    # y = 1 if calibrator was wrong
    y = np.array([
        int((f["p_correct"] > 0.5) != (f["is_correct"] == 1))
        for f in features_list
    ], dtype=int)

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    # Cross-validated accuracy
    n_splits = min(5, max(2, len(y) // 20))
    cv_scores = cross_val_score(clf, X_scaled, y, cv=n_splits, scoring="accuracy")

    # Fit on all data for coefficients
    clf.fit(X_scaled, y)
    coefs = clf.coef_[0]

    # Sort features by absolute coefficient
    feat_importance = sorted(
        zip(NUMERIC_FEATURES, coefs.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return {
        "features": NUMERIC_FEATURES,
        "coefficients": {feat: round(c, 5) for feat, c in zip(NUMERIC_FEATURES, coefs.tolist())},
        "intercept": round(float(clf.intercept_[0]), 5),
        "cv_accuracy_mean": round(float(np.mean(cv_scores)), 4),
        "cv_accuracy_std": round(float(np.std(cv_scores)), 4),
        "n_folds": n_splits,
        "class_balance": {
            "calibrator_correct": int(np.sum(y == 0)),
            "calibrator_wrong": int(np.sum(y == 1)),
        },
        "top_error_predictors": [
            {"feature": feat, "coefficient": round(c, 5), "abs_importance": round(abs(c), 5)}
            for feat, c in feat_importance
        ],
    }


def compute_per_benchmark(features_list: list[dict]) -> dict:
    """Aggregate features per benchmark and compute per-benchmark AUROC."""
    from sklearn.metrics import roc_auc_score

    bench_groups = defaultdict(list)
    for f in features_list:
        bench_groups[f["benchmark"]].append(f)

    per_bench = {}
    for bench, samples in sorted(bench_groups.items()):
        n = len(samples)
        is_correct = [s["is_correct"] for s in samples]
        p_correct = [s["p_correct"] for s in samples]

        auroc = None
        if len(set(is_correct)) > 1:
            try:
                auroc = round(float(roc_auc_score(is_correct, p_correct)), 4)
            except Exception:
                pass

        entry = {
            "n": n,
            "auroc": auroc,
            "accuracy": round(sum(is_correct) / n, 4) if n > 0 else None,
        }

        # Mean feature values per benchmark
        for feat_name in NUMERIC_FEATURES:
            vals = [s[feat_name] for s in samples]
            entry[f"mean_{feat_name}"] = round(float(np.mean(vals)), 4) if vals else None

        # Question type distribution
        type_counts = defaultdict(int)
        for s in samples:
            type_counts[s["question_type"]] += 1
        entry["question_type_dist"] = dict(type_counts)

        per_bench[bench] = entry

    return per_bench


def compute_benchmark_feature_correlations(per_bench: dict) -> dict:
    """Correlate benchmark-level mean features with benchmark-level AUROC."""
    from scipy.stats import pearsonr

    # Only include benchmarks that have a valid AUROC
    valid = {b: v for b, v in per_bench.items() if v["auroc"] is not None}
    if len(valid) < 3:
        return {"note": "Too few benchmarks with valid AUROC for correlation"}

    aurocs = np.array([v["auroc"] for v in valid.values()])
    correlations = {}
    for feat_name in NUMERIC_FEATURES:
        key = f"mean_{feat_name}"
        vals = np.array([v.get(key, 0) or 0 for v in valid.values()], dtype=float)
        if np.std(vals) < 1e-12:
            continue
        try:
            r, p = pearsonr(vals, aurocs)
            correlations[f"auroc_vs_mean_{feat_name}"] = {
                "pearson_r": round(float(r), 4),
                "p_value": round(float(p), 5),
            }
        except Exception:
            pass

    return correlations


def compute_correlation_matrix(features_list: list[dict]) -> dict:
    """Pearson correlations between all numeric features and key targets."""
    from scipy.stats import pearsonr

    targets = {
        "p_correct": np.array([f["p_correct"] for f in features_list], dtype=float),
        "is_correct": np.array([f["is_correct"] for f in features_list], dtype=float),
        "calibrator_error": np.array([
            int((f["p_correct"] > 0.5) != (f["is_correct"] == 1))
            for f in features_list
        ], dtype=float),
    }

    correlations = {}
    for feat_name in NUMERIC_FEATURES:
        feat_vals = np.array([f[feat_name] for f in features_list], dtype=float)
        feat_vals = np.nan_to_num(feat_vals, nan=0.0)
        if np.std(feat_vals) < 1e-12:
            continue
        for target_name, target_vals in targets.items():
            try:
                r, p = pearsonr(feat_vals, target_vals)
                correlations[f"{target_name}_vs_{feat_name}"] = {
                    "pearson_r": round(float(r), 4),
                    "p_value": round(float(p), 6),
                }
            except Exception:
                pass

    return correlations


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(feature_stats, logreg, per_bench, correlations, n_correct, n_wrong):
    """Print a human-readable summary table."""
    total = n_correct + n_wrong
    print("\n" + "=" * 90)
    print("FEATURE ANALYSIS SUMMARY")
    print("=" * 90)

    # Calibrator accuracy
    print(f"\nCalibrator binary accuracy (threshold=0.5): "
          f"{n_correct}/{total} = {n_correct / total * 100:.1f}%")

    # Feature stats table
    print(f"\n{'Feature':<28} {'Mean':>9} {'Corr.Mean':>10} {'Wrong.Mean':>10} {'MW-U p':>10} {'Sig':>4}")
    print("-" * 75)
    for feat in NUMERIC_FEATURES:
        s = feature_stats[feat]
        cm = f"{s['correct_mean']:.2f}" if s['correct_mean'] is not None else "N/A"
        wm = f"{s['wrong_mean']:.2f}" if s['wrong_mean'] is not None else "N/A"
        p_val = s['mann_whitney_p']
        p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
        sig = ""
        if p_val is not None:
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
        print(f"{feat:<28} {s['mean']:>9.2f} {cm:>10} {wm:>10} {p_str:>10} {sig:>4}")

    # Logistic regression
    print(f"\nLogistic Regression (predicting calibrator error from features):")
    print(f"  CV accuracy: {logreg['cv_accuracy_mean']:.4f} +/- {logreg['cv_accuracy_std']:.4f} "
          f"({logreg['n_folds']}-fold)")
    print(f"  Class balance: {logreg['class_balance']}")
    print(f"\n  Top error predictors (by |coefficient|):")
    for i, entry in enumerate(logreg["top_error_predictors"][:10]):
        direction = "+" if entry["coefficient"] > 0 else "-"
        print(f"    {i + 1}. {entry['feature']:<28} coeff={entry['coefficient']:>+.4f} "
              f"(|{entry['abs_importance']:.4f}|)")

    # Per-benchmark
    print(f"\n{'Benchmark':<24} {'N':>5} {'AUROC':>7} {'Acc':>6} {'MeanRspLen':>11} {'MeanWordR':>10}")
    print("-" * 68)
    for bench in sorted(per_bench.keys()):
        b = per_bench[bench]
        auroc_str = f"{b['auroc']:.3f}" if b['auroc'] is not None else "N/A"
        acc_str = f"{b['accuracy']:.3f}" if b['accuracy'] is not None else "N/A"
        rl = b.get('mean_response_length', 0) or 0
        wc = b.get('mean_word_count_r', 0) or 0
        print(f"{bench:<24} {b['n']:>5} {auroc_str:>7} {acc_str:>6} {rl:>11.1f} {wc:>10.1f}")

    # Top correlations with calibrator error
    error_corrs = {
        k: v for k, v in correlations.items()
        if k.startswith("calibrator_error_vs_")
    }
    if error_corrs:
        sorted_corrs = sorted(error_corrs.items(), key=lambda x: abs(x[1]["pearson_r"]), reverse=True)
        print(f"\nTop correlations with calibrator error:")
        for k, v in sorted_corrs[:10]:
            feat = k.replace("calibrator_error_vs_", "")
            print(f"  {feat:<28} r={v['pearson_r']:>+.4f}  p={v['p_value']:.6f}")

    print("\n" + "=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CPU Feature Analysis for UQ Scored Data")
    parser.add_argument("--scored_dir", type=str,
                        default="data/use_cases/scored_test_only/",
                        help="Directory with *_scored.jsonl files")
    parser.add_argument("--output", type=str,
                        default="data/use_cases/results_test_only/feature_analysis.json",
                        help="Output JSON path")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run on first 200 samples only")
    parser.add_argument("--max_examples", type=int, default=0,
                        help="Max samples to process (0 = all)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0 = auto)")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    scored_dir = Path(args.scored_dir)
    if not scored_dir.is_absolute():
        scored_dir = project_root / scored_dir
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    max_samples = args.max_examples
    if args.smoke_test:
        max_samples = 200
        print("[SMOKE TEST MODE] Processing first 200 samples only.\n")

    # Determine worker count
    n_workers = args.workers
    if n_workers <= 0:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    print(f"Using {n_workers} workers for feature extraction.\n")

    # --- Step 1: Load data ---
    print("Step 1: Loading scored data...")
    samples = load_scored_data(str(scored_dir), max_samples=max_samples)
    if not samples:
        print("ERROR: No samples found. Check --scored_dir path.")
        sys.exit(1)

    # --- Step 2: Extract features (parallel) ---
    print(f"\nStep 2: Extracting features from {len(samples)} samples...")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        features_list = list(pool.map(extract_features, samples, chunksize=64))
    print(f"  Extracted {len(features_list)} feature vectors.")

    # Question type distribution
    type_counts = defaultdict(int)
    for f in features_list:
        type_counts[f["question_type"]] += 1
    print(f"  Question types: { {k: v for k, v in sorted(type_counts.items())} }")

    # --- Step 3: Feature stats + Mann-Whitney tests ---
    print("\nStep 3: Computing feature statistics and Mann-Whitney U tests...")
    feature_stats, n_correct, n_wrong = compute_feature_stats(features_list)

    # --- Step 4: Logistic regression ---
    print("\nStep 4: Running logistic regression (predicting calibrator error)...")
    logreg = run_logistic_regression(features_list)

    # --- Step 5: Per-benchmark analysis ---
    print("\nStep 5: Per-benchmark feature aggregation...")
    per_bench = compute_per_benchmark(features_list)

    # --- Step 6: Benchmark-level correlations ---
    print("\nStep 6: Benchmark-level feature-AUROC correlations...")
    bench_corrs = compute_benchmark_feature_correlations(per_bench)

    # --- Step 7: Full correlation matrix ---
    print("\nStep 7: Correlation matrix (features vs p_correct, is_correct, error)...")
    correlations = compute_correlation_matrix(features_list)

    # --- Build output ---
    output = {
        "n_samples": len(features_list),
        "n_calibrator_correct": n_correct,
        "n_calibrator_wrong": n_wrong,
        "calibrator_accuracy": round(n_correct / (n_correct + n_wrong), 4),
        "question_type_distribution": dict(type_counts),
        "feature_stats": feature_stats,
        "logistic_regression": logreg,
        "per_benchmark": per_bench,
        "benchmark_feature_auroc_correlations": bench_corrs,
        "correlations": correlations,
        "top_error_predictors": logreg["top_error_predictors"],
    }

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # --- Print summary ---
    print_summary(feature_stats, logreg, per_bench, correlations, n_correct, n_wrong)

    print(f"\nDone. Output: {output_path}")


if __name__ == "__main__":
    main()
