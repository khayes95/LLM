#!/usr/bin/env python3
"""Contamination sensitivity analysis: remove near-duplicate test samples at
various Jaccard similarity thresholds and show AUROC is stable.

For each test sample, we compute the character 5-gram Jaccard similarity to its
nearest training sample (matched on question text). Then at a range of
thresholds we remove all test samples whose max_jaccard >= threshold and report
AUROC for the calibrator, verbalized baseline, and combined baseline.

Usage:
    # Full run (submit to debug partition)
    python scripts/cpu_sensitivity_analysis.py

    # Smoke test (100 test, 500 train)
    python scripts/cpu_sensitivity_analysis.py --smoke_test

Output: data/use_cases/results_test_only/sensitivity_analysis.json
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# Constants (mirrored from train_best_uq.py)
# ============================================================

DATA_SOURCES = {
    "gpt5mini": {"dir": "runs/gpt5_mini_combined", "type": "combined"},
    "gpt52": {"dir": "runs", "prefix": "gpt52_high_", "type": "prefixed"},
    "qwen35": {"dir": "runs", "prefix": "qwen35_397b_", "type": "prefixed"},
}

EXCLUDED = {
    "triviaqa", "babilong", "vsr", "aokvqa", "erqa", "tutorbench",
    "healthbench", "arc", "oolong",
}

SCORED_DIR = Path("data/use_cases/scored_test_only")
SPLIT_INFO = Path("uq_models/best_unified/split_info.json")
OUTPUT_PATH = Path("data/use_cases/results_test_only/sensitivity_analysis.json")

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_BOOTSTRAP = 1000


# ============================================================
# Character n-gram Jaccard
# ============================================================

def char_ngrams(text: str, n: int = 5) -> set:
    """Extract all character n-grams from text."""
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity |A & B| / |A | B|."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ============================================================
# Training data loading (from runs/ — same logic as train_best_uq.py)
# ============================================================

def extract_question_text(input_data) -> str:
    """Extract question text from input field."""
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, dict):
        for key in ["question", "query", "query_cot", "prompt", "text"]:
            if key in input_data and input_data[key]:
                val = input_data[key]
                if isinstance(val, str):
                    return val
                if isinstance(val, list):
                    parts = []
                    for item in val:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict) and "text" in item:
                            parts.append(item["text"])
                    return "\n".join(parts)
        # Fallback: concatenate all string values
        parts = []
        for v in input_data.values():
            if isinstance(v, str) and len(v) > 20:
                parts.append(v)
        return "\n".join(parts)
    return ""


def load_predictions_from_dir(pred_file: Path) -> list:
    """Load predictions from a single predictions.jsonl file.
    Returns list of (id, question_text) tuples."""
    results = []
    with open(pred_file) as f:
        for line in f:
            try:
                pred = json.loads(line)
            except json.JSONDecodeError:
                continue

            score = pred.get("score", -1)
            if isinstance(score, dict):
                correct = score.get("correct", -1)
            else:
                correct = score
            if correct not in (0, 1):
                continue

            question = extract_question_text(pred.get("input", {}))
            if not question:
                continue

            sample_id = str(pred.get("id", ""))
            results.append((sample_id, question[:2000]))
    return results


def load_training_questions(train_ids: set) -> dict:
    """Load question text for all training samples from runs/ directories.
    Returns dict: prefixed_id -> question_text (truncated to 200 chars for preview matching).
    Supports both benchmark-prefixed IDs (e.g. 'mmlu_104') and bare IDs ('104')."""
    train_questions = {}

    for model_name, config in DATA_SOURCES.items():
        if config["type"] == "combined":
            combined_path = Path(config["dir"])
            for bench_dir in sorted(combined_path.iterdir()):
                if not bench_dir.is_dir() or bench_dir.name in EXCLUDED:
                    continue
                benchmark = bench_dir.name
                pred_file = bench_dir / "predictions.jsonl"
                if not pred_file.exists():
                    continue
                for sid, question in load_predictions_from_dir(pred_file):
                    prefixed_id = f"{benchmark}_{sid}"
                    if prefixed_id in train_ids or sid in train_ids:
                        # Store the first 200 chars to match question_preview
                        train_questions[prefixed_id] = question[:200]
        else:
            runs_path = Path(config["dir"])
            prefix = config["prefix"]
            for run_dir in sorted(runs_path.iterdir()):
                if not run_dir.name.startswith(prefix):
                    continue
                benchmark = run_dir.name[len(prefix):]
                if benchmark in EXCLUDED:
                    continue
                pred_file = run_dir / "predictions.jsonl"
                if not pred_file.exists():
                    continue
                for sid, question in load_predictions_from_dir(pred_file):
                    prefixed_id = f"{benchmark}_{sid}"
                    if prefixed_id in train_ids or sid in train_ids:
                        train_questions[prefixed_id] = question[:200]

    return train_questions


# ============================================================
# Scored test data loading
# ============================================================

def load_scored_test_data() -> list:
    """Load all scored test data from SCORED_DIR.
    Returns list of dicts with all fields from the JSONL."""
    all_samples = []
    for fname in ["gpt5mini_scored.jsonl", "gpt52_scored.jsonl", "qwen35_scored.jsonl"]:
        fpath = SCORED_DIR / fname
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        with open(fpath) as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    all_samples.append(sample)
                except json.JSONDecodeError:
                    continue
    return all_samples


# ============================================================
# Parallel Jaccard computation
# ============================================================

def compute_max_jaccard_for_sample(args):
    """Worker function: compute max Jaccard between one test sample and all training samples.
    args = (test_question_preview, train_ngrams_list)
    Returns max_jaccard similarity."""
    test_preview, train_ngrams_list = args
    test_ngrams = char_ngrams(test_preview, 5)
    if not test_ngrams:
        return 0.0

    max_sim = 0.0
    for train_ng in train_ngrams_list:
        sim = jaccard_similarity(test_ngrams, train_ng)
        if sim > max_sim:
            max_sim = sim
        # Early exit: can't do better than 1.0
        if max_sim >= 1.0:
            break
    return max_sim


# ============================================================
# AUROC computation with bootstrap CI
# ============================================================

def safe_auroc(labels, scores):
    """Compute AUROC, returning NaN if undefined (single class or NaN values)."""
    from sklearn.metrics import roc_auc_score
    labels = np.array(labels, dtype=float)
    scores = np.array(scores, dtype=float)
    # Remove NaN entries
    mask = ~(np.isnan(labels) | np.isnan(scores))
    labels = labels[mask]
    scores = scores[mask]
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def bootstrap_auroc(labels, scores, n_bootstrap=1000, seed=42):
    """Compute AUROC with bootstrap 95% CI.
    Returns (auroc, ci_low, ci_high)."""
    labels = np.array(labels, dtype=float)
    scores = np.array(scores, dtype=float)
    # Remove NaN entries
    mask = ~(np.isnan(labels) | np.isnan(scores))
    labels = labels[mask]
    scores = scores[mask]
    n = len(labels)

    if len(np.unique(labels)) < 2 or n < 10:
        return float("nan"), float("nan"), float("nan")

    point_estimate = safe_auroc(labels, scores)

    rng = np.random.RandomState(seed)
    boot_aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_labels = labels[idx]
        boot_scores = scores[idx]
        if len(np.unique(boot_labels)) < 2:
            continue
        boot_aurocs.append(safe_auroc(boot_labels, boot_scores))

    min_valid = max(20, n_bootstrap // 5)
    if len(boot_aurocs) < min_valid:
        return point_estimate, float("nan"), float("nan")

    boot_aurocs = np.array(boot_aurocs)
    ci_low = float(np.percentile(boot_aurocs, 2.5))
    ci_high = float(np.percentile(boot_aurocs, 97.5))
    return float(point_estimate), ci_low, ci_high


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Contamination sensitivity analysis")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Use 100 test / 500 train samples for quick test")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help="Output JSON path")
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP,
                        help="Number of bootstrap resamples")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Number of parallel workers (default: auto)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_workers = args.n_workers
    if n_workers is None:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    n_workers = max(1, min(n_workers, 80))

    print("=" * 70)
    print("CONTAMINATION SENSITIVITY ANALYSIS")
    print("=" * 70)
    if args.smoke_test:
        print("  MODE: smoke test (100 test, 500 train)")
    print(f"  Workers: {n_workers}")
    print(f"  Bootstrap resamples: {args.n_bootstrap}")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Output: {output_path}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load training IDs from split_info.json
    # ------------------------------------------------------------------
    print("STEP 1: Loading split info...")
    t0 = time.time()
    if not SPLIT_INFO.exists():
        print(f"  ERROR: {SPLIT_INFO} not found. Run train_best_uq.py first.")
        sys.exit(1)

    with open(SPLIT_INFO) as f:
        split_info = json.load(f)
    train_ids = set(split_info["train_ids"])
    print(f"  {len(train_ids)} training IDs loaded ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # Step 2: Load training questions from runs/
    # ------------------------------------------------------------------
    print("\nSTEP 2: Loading training question text from runs/...")
    t0 = time.time()
    train_questions = load_training_questions(train_ids)
    print(f"  {len(train_questions)} training questions loaded ({time.time()-t0:.1f}s)")

    if len(train_questions) == 0:
        print("  ERROR: No training questions found. Check runs/ directories.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Load scored test data
    # ------------------------------------------------------------------
    print("\nSTEP 3: Loading scored test data...")
    t0 = time.time()
    test_samples = load_scored_test_data()
    print(f"  {len(test_samples)} test samples loaded ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # Step 4: Subsample for smoke test
    # ------------------------------------------------------------------
    if args.smoke_test:
        rng = np.random.RandomState(42)
        if len(test_samples) > 100:
            idx = rng.choice(len(test_samples), 100, replace=False)
            test_samples = [test_samples[i] for i in idx]
        # Subsample training questions
        train_keys = list(train_questions.keys())
        if len(train_keys) > 500:
            keep = set(rng.choice(train_keys, 500, replace=False))
            train_questions = {k: v for k, v in train_questions.items() if k in keep}
        print(f"\n  Smoke test: {len(test_samples)} test, {len(train_questions)} train")

    # ------------------------------------------------------------------
    # Step 5: Pre-compute character 5-grams for all training samples
    # ------------------------------------------------------------------
    print("\nSTEP 4: Pre-computing character 5-grams for training samples...")
    t0 = time.time()
    train_ngrams_list = [char_ngrams(q, 5) for q in train_questions.values()]
    print(f"  {len(train_ngrams_list)} training n-gram sets computed ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # Step 6: Compute max Jaccard for each test sample (parallel)
    # ------------------------------------------------------------------
    print(f"\nSTEP 5: Computing max Jaccard similarity ({len(test_samples)} test x {len(train_ngrams_list)} train)...")
    t0 = time.time()

    # Build worker args: (test_question_preview, train_ngrams_list)
    worker_args = []
    for sample in test_samples:
        preview = sample.get("question_preview", "")
        worker_args.append((preview, train_ngrams_list))

    max_jaccards = []
    # Use ProcessPoolExecutor for parallel Jaccard computation
    # Chunk the work to avoid excessive IPC overhead
    chunk_size = max(1, len(worker_args) // (n_workers * 4))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(
            compute_max_jaccard_for_sample,
            worker_args,
            chunksize=chunk_size,
        ))
    max_jaccards = results

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s)")

    # Attach max_jaccard to each sample
    for sample, mj in zip(test_samples, max_jaccards):
        sample["max_jaccard"] = mj

    # ------------------------------------------------------------------
    # Step 7: Distribution summary
    # ------------------------------------------------------------------
    jaccards_arr = np.array(max_jaccards)
    print(f"\n  Jaccard distribution:")
    print(f"    min={jaccards_arr.min():.4f}  median={np.median(jaccards_arr):.4f}  "
          f"mean={jaccards_arr.mean():.4f}  max={jaccards_arr.max():.4f}")
    for t in THRESHOLDS:
        n_above = int(np.sum(jaccards_arr >= t))
        print(f"    >= {t:.1f}: {n_above:5d} samples ({100*n_above/len(jaccards_arr):.1f}%)")

    # ------------------------------------------------------------------
    # Step 8: Threshold sweep — compute AUROC at each threshold
    # ------------------------------------------------------------------
    print(f"\nSTEP 6: Threshold sweep with bootstrap CIs...")
    t0 = time.time()

    threshold_results = []

    # Also compute "no filtering" baseline (threshold = 0.0 effectively keeps all)
    all_thresholds = [0.0] + THRESHOLDS

    for threshold in all_thresholds:
        if threshold == 0.0:
            filtered = test_samples  # keep all
        else:
            filtered = [s for s in test_samples if s["max_jaccard"] < threshold]

        n_removed = len(test_samples) - len(filtered)
        n_remaining = len(filtered)

        if n_remaining < 20:
            print(f"  threshold={threshold:.1f}: {n_remaining} remaining — too few, skipping")
            threshold_results.append({
                "threshold": threshold,
                "n_removed": n_removed,
                "n_remaining": n_remaining,
                "calibrator_auroc": None,
                "calibrator_ci": [None, None],
                "verbalized_auroc": None,
                "verbalized_ci": [None, None],
                "combined_auroc": None,
                "combined_ci": [None, None],
            })
            continue

        labels = np.array([s["is_correct"] for s in filtered], dtype=float)
        p_correct = np.array([s["p_correct"] if s.get("p_correct") is not None else float("nan")
                              for s in filtered])
        p_verbalized = np.array([s["verbalized_confidence"] if s.get("verbalized_confidence") is not None
                                 else float("nan") for s in filtered])
        p_combined = np.array([s["p_combined_baseline"] if s.get("p_combined_baseline") is not None
                               else float("nan") for s in filtered])

        cal_auroc, cal_lo, cal_hi = bootstrap_auroc(labels, p_correct, args.n_bootstrap)
        verb_auroc, verb_lo, verb_hi = bootstrap_auroc(labels, p_verbalized, args.n_bootstrap)
        comb_auroc, comb_lo, comb_hi = bootstrap_auroc(labels, p_combined, args.n_bootstrap)

        result = {
            "threshold": threshold,
            "n_removed": n_removed,
            "n_remaining": n_remaining,
            "calibrator_auroc": cal_auroc,
            "calibrator_ci": [cal_lo, cal_hi],
            "verbalized_auroc": verb_auroc,
            "verbalized_ci": [verb_lo, verb_hi],
            "combined_auroc": comb_auroc,
            "combined_ci": [comb_lo, comb_hi],
        }
        threshold_results.append(result)

        label_str = "ALL  " if threshold == 0.0 else f">={threshold:.1f}"
        print(f"  {label_str}  removed={n_removed:5d}  remain={n_remaining:5d}  "
              f"Cal={cal_auroc:.4f} [{cal_lo:.4f},{cal_hi:.4f}]  "
              f"Verb={verb_auroc:.4f}  Comb={comb_auroc:.4f}")

    elapsed = time.time() - t0
    print(f"  ({elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # Step 9: Per-benchmark analysis — which benchmarks are most affected
    # ------------------------------------------------------------------
    print(f"\nSTEP 7: Per-benchmark contamination analysis...")

    # Group by benchmark
    bench_groups = defaultdict(list)
    for sample in test_samples:
        bench = sample.get("benchmark", "unknown")
        bench_groups[bench].append(sample)

    benchmark_analysis = {}
    for bench, samples in sorted(bench_groups.items()):
        jacs = [s["max_jaccard"] for s in samples]
        jacs_arr = np.array(jacs)
        n_total = len(samples)

        removals_by_threshold = {}
        for t in THRESHOLDS:
            n_above = int(np.sum(jacs_arr >= t))
            removals_by_threshold[str(t)] = {
                "n_removed": n_above,
                "pct_removed": round(100 * n_above / n_total, 1) if n_total > 0 else 0,
            }

        benchmark_analysis[bench] = {
            "n_total": n_total,
            "jaccard_mean": round(float(jacs_arr.mean()), 4),
            "jaccard_median": round(float(np.median(jacs_arr)), 4),
            "jaccard_max": round(float(jacs_arr.max()), 4),
            "removals_by_threshold": removals_by_threshold,
        }

    # Print summary of most affected benchmarks (at threshold 0.5)
    print(f"\n  Benchmarks most affected at threshold=0.5:")
    print(f"  {'Benchmark':<25s} {'Total':>6s} {'Removed':>8s} {'%':>7s} {'MeanJac':>8s} {'MaxJac':>8s}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")
    for bench, info in sorted(benchmark_analysis.items(),
                               key=lambda x: x[1]["removals_by_threshold"]["0.5"]["pct_removed"],
                               reverse=True):
        rem = info["removals_by_threshold"]["0.5"]
        print(f"  {bench:<25s} {info['n_total']:>6d} {rem['n_removed']:>8d} "
              f"{rem['pct_removed']:>6.1f}% {info['jaccard_mean']:>8.4f} {info['jaccard_max']:>8.4f}")

    # ------------------------------------------------------------------
    # Step 10: Stability assessment
    # ------------------------------------------------------------------
    print(f"\nSTEP 8: Stability assessment...")
    cal_aurocs = [r["calibrator_auroc"] for r in threshold_results
                  if r["calibrator_auroc"] is not None]
    if len(cal_aurocs) >= 2:
        cal_range = max(cal_aurocs) - min(cal_aurocs)
        cal_std = float(np.std(cal_aurocs))
        print(f"  Calibrator AUROC range across thresholds: {cal_range:.4f}")
        print(f"  Calibrator AUROC std across thresholds:   {cal_std:.4f}")
        if cal_range < 0.02:
            print(f"  CONCLUSION: AUROC is STABLE (range < 0.02). "
                  f"Removing potential contaminants does not change performance.")
        elif cal_range < 0.05:
            print(f"  CONCLUSION: AUROC is MOSTLY STABLE (range < 0.05). "
                  f"Minor variation at extreme thresholds.")
        else:
            print(f"  CONCLUSION: AUROC shows NOTABLE variation (range >= 0.05). "
                  f"Investigate further.")

    # ------------------------------------------------------------------
    # Step 11: Save results
    # ------------------------------------------------------------------
    output = {
        "description": "Contamination sensitivity analysis: AUROC stability across Jaccard similarity thresholds",
        "method": "Character 5-gram Jaccard similarity between test question_preview and training question text (truncated to 200 chars)",
        "n_test_total": len(test_samples),
        "n_train_total": len(train_questions),
        "jaccard_distribution": {
            "min": round(float(jaccards_arr.min()), 4),
            "median": round(float(np.median(jaccards_arr)), 4),
            "mean": round(float(jaccards_arr.mean()), 4),
            "max": round(float(jaccards_arr.max()), 4),
            "p90": round(float(np.percentile(jaccards_arr, 90)), 4),
            "p95": round(float(np.percentile(jaccards_arr, 95)), 4),
            "p99": round(float(np.percentile(jaccards_arr, 99)), 4),
        },
        "threshold_results": threshold_results,
        "benchmark_analysis": benchmark_analysis,
        "stability": {
            "calibrator_auroc_range": round(cal_range, 4) if len(cal_aurocs) >= 2 else None,
            "calibrator_auroc_std": round(cal_std, 4) if len(cal_aurocs) >= 2 else None,
        },
        "smoke_test": args.smoke_test,
        "n_bootstrap": args.n_bootstrap,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Threshold':>10s} {'Removed':>8s} {'Remain':>8s} "
          f"{'Cal AUROC':>12s} {'Cal 95% CI':>18s} "
          f"{'Verb AUROC':>12s} {'Comb AUROC':>12s}")
    print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*12} {'-'*18} {'-'*12} {'-'*12}")

    for r in threshold_results:
        t = r["threshold"]
        t_str = "none" if t == 0.0 else f">= {t:.1f}"
        if r["calibrator_auroc"] is None:
            print(f"{t_str:>10s} {r['n_removed']:>8d} {r['n_remaining']:>8d}  {'N/A':>10s}")
        else:
            cal_ci = f"[{r['calibrator_ci'][0]:.4f},{r['calibrator_ci'][1]:.4f}]"
            print(f"{t_str:>10s} {r['n_removed']:>8d} {r['n_remaining']:>8d} "
                  f"{r['calibrator_auroc']:>12.4f} {cal_ci:>18s} "
                  f"{r['verbalized_auroc']:>12.4f} {r['combined_auroc']:>12.4f}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
