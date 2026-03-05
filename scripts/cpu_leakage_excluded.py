#!/scratch/khayes/.conda/envs/uq_eval/bin/python
"""Remove potentially contaminated samples from scored test data and recompute all metrics.

Reads the contamination report to identify test IDs that overlap with training data
(exact Q+R overlaps and exact question overlaps), removes them from the scored test
data, and recomputes AUROC for the calibrator and all baselines on both the full and
clean subsets. Reports deltas and bootstrap 95% CIs on the clean subset.

Usage:
    # Smoke test (100 samples)
    python scripts/cpu_leakage_excluded.py --smoke_test

    # Full run
    python scripts/cpu_leakage_excluded.py

Output: data/use_cases/results_test_only/leakage_excluded.json
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
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCORED_DIR = Path("data/use_cases/scored_test_only")
CONTAMINATION_REPORT = Path("data/use_cases/results_test_only/contamination_report.json")
OUTPUT_PATH = Path("data/use_cases/results_test_only/leakage_excluded.json")

SCORED_FILES = {
    "gpt5mini": SCORED_DIR / "gpt5mini_scored.jsonl",
    "gpt52": SCORED_DIR / "gpt52_scored.jsonl",
    "qwen35": SCORED_DIR / "qwen35_scored.jsonl",
}

# All scoring methods to evaluate
METHODS = {
    "calibrator": "p_correct",
    "platt_verbalized": "p_platt_verbalized",
    "isotonic_verbalized": "p_isotonic_verbalized",
    "length_baseline": "p_length_baseline",
    "combined_baseline": "p_combined_baseline",
    "zeroshot": "p_zeroshot",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path, max_lines=None):
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_contaminated_test_ids(report):
    """Extract the set of test IDs flagged as contaminated (exact overlaps only).

    Collects test_ids from both:
      - details_train_test_qr  (exact question + response overlap)
      - details_train_test_question  (exact question overlap, different response)
    """
    contaminated = set()
    exact = report.get("exact_duplicates", {})

    for entry in exact.get("details_train_test_qr", []):
        for tid in entry.get("test_ids", []):
            contaminated.add(tid)

    for entry in exact.get("details_train_test_question", []):
        for tid in entry.get("test_ids", []):
            contaminated.add(tid)

    return contaminated


def compute_auroc_safe(y_true, y_score):
    """Compute AUROC, returning NaN if undefined (single class)."""
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def _get_score(record, col, default=0.5):
    """Get a score from a record, replacing None/missing with default."""
    val = record.get(col)
    if val is None:
        return default
    return val


def compute_all_aurocs(records):
    """Compute AUROC for every method on a list of scored records.

    Returns dict: method_name -> AUROC (float or NaN).
    """
    if len(records) == 0:
        return {name: float("nan") for name in METHODS}
    y_true = [r["is_correct"] for r in records]
    results = {}
    for name, col in METHODS.items():
        y_score = [_get_score(r, col) for r in records]
        results[name] = compute_auroc_safe(y_true, y_score)
    return results


def _bootstrap_worker(seed, y_true, scores_dict, n):
    """Single bootstrap resample: returns dict method -> auroc."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=n)
    y_t = y_true[idx]
    if len(np.unique(y_t)) < 2:
        return {name: float("nan") for name in scores_dict}
    result = {}
    for name, y_s in scores_dict.items():
        result[name] = roc_auc_score(y_t, y_s[idx])
    return result


def bootstrap_aurocs(records, n_bootstrap=2000, n_workers=None):
    """Bootstrap 95% CI for all methods using multiprocessing.

    Returns dict: method_name -> {"mean", "lower", "upper", "std"}.
    """
    if n_workers is None:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    n = len(records)
    if n == 0:
        empty = {"mean": float("nan"), "lower": float("nan"), "upper": float("nan"), "std": float("nan")}
        return {name: dict(empty) for name in METHODS}

    y_true = np.array([r["is_correct"] for r in records])
    scores_dict = {}
    for name, col in METHODS.items():
        scores_dict[name] = np.array([_get_score(r, col) for r in records])

    worker_fn = partial(_bootstrap_worker, y_true=y_true, scores_dict=scores_dict, n=n)
    seeds = list(range(n_bootstrap))

    print(f"  Bootstrap: {n_bootstrap} resamples, {n_workers} workers, {n} samples")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        boot_results = list(pool.map(worker_fn, seeds, chunksize=max(1, n_bootstrap // n_workers)))

    # Aggregate
    ci = {}
    for name in METHODS:
        vals = np.array([br[name] for br in boot_results])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            ci[name] = {"mean": float("nan"), "lower": float("nan"), "upper": float("nan"), "std": float("nan")}
        else:
            ci[name] = {
                "mean": float(np.mean(vals)),
                "lower": float(np.percentile(vals, 2.5)),
                "upper": float(np.percentile(vals, 97.5)),
                "std": float(np.std(vals)),
            }
    return ci


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Leakage exclusion analysis")
    parser.add_argument("--smoke_test", action="store_true", help="Process only 100 samples per file")
    parser.add_argument("--n_bootstrap", type=int, default=2000, help="Number of bootstrap resamples")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Output JSON path")
    args = parser.parse_args()

    t0 = time.time()
    max_lines = 100 if args.smoke_test else None

    # --- Step 1: Load contamination report ---
    print("Step 1: Loading contamination report ...")
    with open(CONTAMINATION_REPORT, "r") as f:
        report = json.load(f)
    contaminated_ids = extract_contaminated_test_ids(report)
    print(f"  Found {len(contaminated_ids)} unique contaminated test IDs")

    # --- Step 2: Load all scored data ---
    print("Step 2: Loading scored JSONL files ...")
    all_records = []
    per_model_records = {}
    for model_name, path in SCORED_FILES.items():
        recs = load_jsonl(path, max_lines=max_lines)
        per_model_records[model_name] = recs
        all_records.extend(recs)
        print(f"  {model_name}: {len(recs)} samples")
    print(f"  Total: {len(all_records)} samples")

    # --- Step 3: Compute AUROC on FULL test set ---
    print("Step 3: Computing AUROC on full test set ...")
    full_aurocs = compute_all_aurocs(all_records)
    for name, val in full_aurocs.items():
        print(f"  {name}: {val:.4f}")

    # --- Step 4: Identify contaminated records ---
    # A record is contaminated if its 'id' field appears in contaminated_ids
    # (the id field is shared across target models, so the same id in different
    # model files all get removed).
    print("Step 4: Removing contaminated samples ...")
    clean_records = [r for r in all_records if r["id"] not in contaminated_ids]
    removed_records = [r for r in all_records if r["id"] in contaminated_ids]
    n_removed = len(removed_records)
    n_remaining = len(clean_records)
    print(f"  Removed: {n_removed} samples ({n_removed/len(all_records)*100:.1f}%)")
    print(f"  Remaining: {n_remaining} samples")

    # Show which IDs were actually removed vs only in report
    actually_removed_ids = set(r["id"] for r in removed_records)
    report_only_ids = contaminated_ids - actually_removed_ids
    if report_only_ids:
        print(f"  Note: {len(report_only_ids)} contaminated IDs not found in scored data "
              f"(may be in train-only or already excluded)")

    # --- Step 5: Compute AUROC on CLEAN subset ---
    print("Step 5: Computing AUROC on clean subset ...")
    clean_aurocs = compute_all_aurocs(clean_records)
    for name, val in clean_aurocs.items():
        delta = val - full_aurocs[name]
        print(f"  {name}: {val:.4f} (delta: {delta:+.4f})")

    # --- Step 6: Per-target-model breakdown ---
    print("Step 6: Per-target-model breakdown ...")
    per_model_results = {}
    for model_name, recs in per_model_records.items():
        model_clean = [r for r in recs if r["id"] not in contaminated_ids]
        model_removed = len(recs) - len(model_clean)
        full_auc = compute_all_aurocs(recs)
        clean_auc = compute_all_aurocs(model_clean)

        per_model_results[model_name] = {
            "n_total": len(recs),
            "n_removed": model_removed,
            "n_remaining": len(model_clean),
            "full_aurocs": full_auc,
            "clean_aurocs": clean_auc,
            "deltas": {name: clean_auc[name] - full_auc[name] for name in METHODS},
        }

        print(f"\n  {model_name}: {len(recs)} total, {model_removed} removed, "
              f"{len(model_clean)} remaining")
        for name in METHODS:
            print(f"    {name}: {full_auc[name]:.4f} -> {clean_auc[name]:.4f} "
                  f"(delta: {clean_auc[name] - full_auc[name]:+.4f})")

    # --- Step 7: Bootstrap 95% CI on clean subset ---
    print(f"\nStep 7: Bootstrap {args.n_bootstrap} resamples on clean subset ...")
    clean_ci = bootstrap_aurocs(clean_records, n_bootstrap=args.n_bootstrap)
    for name, ci in clean_ci.items():
        print(f"  {name}: {ci['mean']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    # --- Step 8: Assemble output ---
    print("\nStep 8: Saving results ...")
    output = {
        "description": (
            "Leakage exclusion analysis: removes all test samples whose ID appears "
            "in the contamination report (exact Q+R and exact question overlaps with "
            "training data), then recomputes AUROC for the calibrator and all baselines."
        ),
        "contamination_source": str(CONTAMINATION_REPORT),
        "n_contaminated_ids_in_report": len(contaminated_ids),
        "contaminated_ids": sorted(contaminated_ids),
        "n_total": len(all_records),
        "n_removed": n_removed,
        "n_remaining": n_remaining,
        "pct_removed": round(n_removed / len(all_records) * 100, 2),
        "full_aurocs": full_aurocs,
        "clean_aurocs": clean_aurocs,
        "deltas": {name: round(clean_aurocs[name] - full_aurocs[name], 6) for name in METHODS},
        "clean_bootstrap_ci": clean_ci,
        "per_target_model": per_model_results,
        "n_bootstrap": args.n_bootstrap,
        "smoke_test": args.smoke_test,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Output: {args.output}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("LEAKAGE EXCLUSION SUMMARY")
    print("=" * 70)
    print(f"Contaminated IDs removed: {n_removed} / {len(all_records)} "
          f"({n_removed/len(all_records)*100:.1f}%)")
    print(f"Remaining samples: {n_remaining}")
    print()
    print(f"{'Method':<25} {'Full AUROC':>12} {'Clean AUROC':>12} {'Delta':>10} "
          f"{'95% CI':>20}")
    print("-" * 79)
    for name in METHODS:
        ci = clean_ci[name]
        print(f"{name:<25} {full_aurocs[name]:>12.4f} {clean_aurocs[name]:>12.4f} "
              f"{clean_aurocs[name] - full_aurocs[name]:>+10.4f} "
              f"[{ci['lower']:.4f}, {ci['upper']:.4f}]")
    print("=" * 70)

    if all(abs(clean_aurocs[n] - full_aurocs[n]) < 0.01 for n in METHODS
           if not np.isnan(clean_aurocs[n]) and not np.isnan(full_aurocs[n])):
        print("VERDICT: Excluding contaminated samples changes AUROC by <0.01 for "
              "all methods. Results are robust to leakage.")
    else:
        print("VERDICT: Some methods show AUROC change >= 0.01. Investigate further.")


if __name__ == "__main__":
    main()
