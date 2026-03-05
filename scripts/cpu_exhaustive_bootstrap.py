#!/usr/bin/env python3
"""
Exhaustive bootstrap confidence intervals for UQ evaluation.

Upgrade from bootstrap_ci.py:
  - 100,000 bootstrap iterations (configurable)
  - BCa (bias-corrected and accelerated) confidence intervals
  - Multiple CI levels: 90%, 95%, 99%
  - Per-benchmark breakdown with CIs
  - Pairwise comparison tests (P(A > B) for all method pairs)
  - Full parallelization with multiprocessing.Pool
  - Effect sizes (Cohen's d) for calibrator vs each baseline
  - Smoke test mode (500 iterations, 200 samples)

Usage:
    # Full run (submit via SLURM on debug partition):
    python scripts/cpu_exhaustive_bootstrap.py

    # Smoke test:
    python scripts/cpu_exhaustive_bootstrap.py --smoke_test

    # Custom settings:
    python scripts/cpu_exhaustive_bootstrap.py --n_bootstrap 50000 --scored_dir data/use_cases/scored_test_only/
"""

import argparse
import json
import os
import sys
import time
import warnings
from functools import partial
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS = [
    ("p_correct", "Calibrator"),
    ("verbalized_confidence", "Verbalized"),
    ("p_platt_verbalized", "Platt"),
    ("p_isotonic_verbalized", "Isotonic"),
    ("p_length_baseline", "Length"),
    ("p_combined_baseline", "Combined"),
    ("p_zeroshot", "Zero-shot"),
]

TARGETS = ["gpt5mini", "gpt52", "qwen35"]

CI_LEVELS = [0.90, 0.95, 0.99]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scored(path):
    """Load a scored JSONL file into a list of dicts."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_arrays(data, field):
    """Extract valid (y_true, y_score) arrays for a given score field.

    Returns (y_true, y_score) numpy arrays, filtering out rows where
    either is_correct or the score field is None/missing.
    """
    valid = [
        d for d in data
        if d.get(field) is not None and d.get("is_correct") is not None
    ]
    if len(valid) < 10:
        return None, None
    y = np.array([d["is_correct"] for d in valid], dtype=np.int32)
    s = np.array([d[field] for d in valid], dtype=np.float64)
    if len(np.unique(y)) < 2:
        return None, None
    return y, s


# ---------------------------------------------------------------------------
# Vectorized bootstrap AUROC computation
# ---------------------------------------------------------------------------

def _auroc_from_sorted(y_true_sorted, n_pos, n_neg):
    """Compute AUROC from y_true sorted by descending score using the
    trapezoidal (rank-sum) method. This is equivalent to the Wilcoxon-
    Mann-Whitney statistic.
    """
    # Rank-sum approach: sum of ranks of positives
    # When sorted by descending score, rank 0 = highest score
    n = len(y_true_sorted)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    rank_sum = np.sum(ranks[y_true_sorted == 1])
    auroc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    # We sorted descending, so high score = low rank = correct should give
    # high AUROC. But the rank-sum formula gives U/mn where U is the
    # statistic for the *negative* class. We need 1 - that.
    return 1.0 - auroc


def _bootstrap_aurocs_chunk(args):
    """Compute AUROC for a chunk of bootstrap iterations.

    Args:
        args: tuple of (y_true, y_score, boot_indices_chunk)
              boot_indices_chunk is a 2D array of shape (chunk_size, n)

    Returns:
        1D array of AUROC values for each bootstrap sample.
    """
    y_true, y_score, indices_chunk = args
    n_iter = indices_chunk.shape[0]
    results = np.empty(n_iter, dtype=np.float64)

    for i in range(n_iter):
        idx = indices_chunk[i]
        y_b = y_true[idx]
        s_b = y_score[idx]
        # Check both classes present
        n_pos = np.sum(y_b)
        n_neg = len(y_b) - n_pos
        if n_pos == 0 or n_neg == 0:
            results[i] = np.nan
            continue
        try:
            results[i] = roc_auc_score(y_b, s_b)
        except ValueError:
            results[i] = np.nan

    return results


def bootstrap_aurocs_parallel(y_true, y_score, n_bootstrap, n_workers, seed=42):
    """Generate all bootstrap AUROC values in parallel.

    Returns a 1D numpy array of AUROC values (NaNs removed).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    # Generate all indices at once as a 2D array: (n_bootstrap, n)
    all_indices = rng.randint(0, n, size=(n_bootstrap, n)).astype(np.int32)

    # Split into chunks for parallel processing
    chunk_size = max(1, n_bootstrap // n_workers)
    chunks = []
    for start in range(0, n_bootstrap, chunk_size):
        end = min(start + chunk_size, n_bootstrap)
        chunks.append((y_true, y_score, all_indices[start:end]))

    with Pool(processes=n_workers) as pool:
        chunk_results = pool.map(_bootstrap_aurocs_chunk, chunks)

    all_aurocs = np.concatenate(chunk_results)
    # Remove NaNs (from degenerate bootstrap samples)
    all_aurocs = all_aurocs[~np.isnan(all_aurocs)]
    return all_aurocs


# ---------------------------------------------------------------------------
# BCa confidence intervals
# ---------------------------------------------------------------------------

def _jackknife_aurocs(y_true, y_score):
    """Compute leave-one-out jackknife AUROC values.

    Returns a 1D array of length n, where entry i is the AUROC computed
    with sample i removed.
    """
    n = len(y_true)
    jack = np.empty(n, dtype=np.float64)
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        mask[i] = False
        y_j = y_true[mask]
        s_j = y_score[mask]
        if len(np.unique(y_j)) < 2:
            jack[i] = np.nan
        else:
            try:
                jack[i] = roc_auc_score(y_j, s_j)
            except ValueError:
                jack[i] = np.nan
        mask[i] = True

    return jack


def _jackknife_chunk(args):
    """Compute jackknife AUROC for a range of leave-one-out indices.

    Args:
        args: (y_true, y_score, start_idx, end_idx)

    Returns:
        1D array of jackknife AUROCs for indices [start_idx, end_idx).
    """
    y_true, y_score, start, end = args
    n = len(y_true)
    results = np.empty(end - start, dtype=np.float64)
    mask = np.ones(n, dtype=bool)

    for k, i in enumerate(range(start, end)):
        mask[i] = False
        y_j = y_true[mask]
        s_j = y_score[mask]
        if len(np.unique(y_j)) < 2:
            results[k] = np.nan
        else:
            try:
                results[k] = roc_auc_score(y_j, s_j)
            except ValueError:
                results[k] = np.nan
        mask[i] = True

    return results


def jackknife_aurocs_parallel(y_true, y_score, n_workers):
    """Compute leave-one-out jackknife AUROC values in parallel."""
    n = len(y_true)
    chunk_size = max(1, n // n_workers)
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append((y_true, y_score, start, end))

    with Pool(processes=n_workers) as pool:
        chunk_results = pool.map(_jackknife_chunk, chunks)

    return np.concatenate(chunk_results)


def bca_ci(boot_dist, theta_hat, jackknife_values, ci_level):
    """Compute BCa (bias-corrected and accelerated) confidence interval.

    Args:
        boot_dist: 1D array of bootstrap statistics.
        theta_hat: Point estimate of the statistic.
        jackknife_values: 1D array of leave-one-out jackknife values.
        ci_level: Confidence level (e.g. 0.95).

    Returns:
        (ci_low, ci_high) tuple.
    """
    from scipy.stats import norm

    B = len(boot_dist)
    if B == 0:
        return (float("nan"), float("nan"))

    alpha = (1.0 - ci_level) / 2.0

    # Bias correction: z0
    prop_below = np.mean(boot_dist < theta_hat)
    # Clamp to avoid infinite z values
    prop_below = np.clip(prop_below, 1e-10, 1.0 - 1e-10)
    z0 = norm.ppf(prop_below)

    # Acceleration: a (from jackknife)
    jack_valid = jackknife_values[~np.isnan(jackknife_values)]
    if len(jack_valid) < 3:
        # Fall back to percentile method
        ci_low = np.percentile(boot_dist, 100 * alpha)
        ci_high = np.percentile(boot_dist, 100 * (1 - alpha))
        return (float(ci_low), float(ci_high))

    jack_mean = np.mean(jack_valid)
    diff = jack_mean - jack_valid
    a_num = np.sum(diff ** 3)
    a_den = 6.0 * (np.sum(diff ** 2)) ** 1.5
    if a_den == 0:
        a = 0.0
    else:
        a = a_num / a_den

    # Adjusted quantiles
    z_alpha_low = norm.ppf(alpha)
    z_alpha_high = norm.ppf(1 - alpha)

    def adjusted_quantile(z_alpha):
        numer = z0 + z_alpha
        denom = 1.0 - a * numer
        if denom == 0:
            return alpha  # fallback
        adj = norm.cdf(z0 + numer / denom)
        return np.clip(adj, 1e-10, 1.0 - 1e-10)

    q_low = adjusted_quantile(z_alpha_low)
    q_high = adjusted_quantile(z_alpha_high)

    ci_low = np.percentile(boot_dist, 100 * q_low)
    ci_high = np.percentile(boot_dist, 100 * q_high)

    return (float(ci_low), float(ci_high))


# ---------------------------------------------------------------------------
# Pairwise bootstrap comparison
# ---------------------------------------------------------------------------

def _pairwise_bootstrap_chunk(args):
    """Compute AUROC difference for a chunk of bootstrap iterations.

    Args:
        args: (y_true, scores_a, scores_b, indices_chunk)

    Returns:
        1D array of (AUROC_A - AUROC_B) for each bootstrap sample.
    """
    y_true, scores_a, scores_b, indices_chunk = args
    n_iter = indices_chunk.shape[0]
    diffs = np.empty(n_iter, dtype=np.float64)

    for i in range(n_iter):
        idx = indices_chunk[i]
        y_b = y_true[idx]
        sa_b = scores_a[idx]
        sb_b = scores_b[idx]

        n_pos = np.sum(y_b)
        n_neg = len(y_b) - n_pos
        if n_pos == 0 or n_neg == 0:
            diffs[i] = np.nan
            continue
        try:
            auroc_a = roc_auc_score(y_b, sa_b)
            auroc_b = roc_auc_score(y_b, sb_b)
            diffs[i] = auroc_a - auroc_b
        except ValueError:
            diffs[i] = np.nan

    return diffs


def pairwise_bootstrap(y_true, scores_a, scores_b, n_bootstrap, n_workers, seed=42):
    """Compute bootstrap distribution of AUROC(A) - AUROC(B).

    Returns dict with mean_diff, std_diff, p_a_better.
    Uses the SAME bootstrap indices for both methods (paired comparison).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    all_indices = rng.randint(0, n, size=(n_bootstrap, n)).astype(np.int32)

    chunk_size = max(1, n_bootstrap // n_workers)
    chunks = []
    for start in range(0, n_bootstrap, chunk_size):
        end = min(start + chunk_size, n_bootstrap)
        chunks.append((y_true, scores_a, scores_b, all_indices[start:end]))

    with Pool(processes=n_workers) as pool:
        chunk_results = pool.map(_pairwise_bootstrap_chunk, chunks)

    diffs = np.concatenate(chunk_results)
    diffs = diffs[~np.isnan(diffs)]

    if len(diffs) == 0:
        return {"mean_diff": float("nan"), "std_diff": float("nan"),
                "p_a_better": float("nan"), "n_valid": 0}

    return {
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs)),
        "p_a_better": float(np.mean(diffs > 0)),
        "n_valid": int(len(diffs)),
    }


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def cohens_d(boot_a, boot_b):
    """Compute Cohen's d effect size between two bootstrap distributions.

    d = (mean_a - mean_b) / pooled_sd
    """
    mean_a, mean_b = np.mean(boot_a), np.mean(boot_b)
    sd_a, sd_b = np.std(boot_a, ddof=1), np.std(boot_b, ddof=1)
    pooled_sd = np.sqrt((sd_a**2 + sd_b**2) / 2)
    if pooled_sd == 0:
        return float("nan")
    return float((mean_a - mean_b) / pooled_sd)


# ---------------------------------------------------------------------------
# Worker for per-method bootstrap (used in parallel dispatch)
# ---------------------------------------------------------------------------

def _compute_method_bootstrap(args):
    """Worker: compute full bootstrap + BCa CIs for one method.

    Args:
        args: (method_name, y_true, y_score, n_bootstrap, n_workers_inner, ci_levels, seed)
              n_workers_inner: number of workers for the inner parallel loops.
              For per-method parallelism we set this to 1 (serialized inner)
              because the outer level is already parallelized.

    Returns:
        dict with method name, point AUROC, CIs, and boot_dist.
    """
    method_name, y_true, y_score, n_bootstrap, ci_levels, seed = args

    # Point estimate
    auroc = float(roc_auc_score(y_true, y_score))

    # Bootstrap distribution (serial within worker -- parallelism is at method level)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    all_indices = rng.randint(0, n, size=(n_bootstrap, n)).astype(np.int32)

    boot_aurocs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = all_indices[i]
        y_b = y_true[idx]
        s_b = y_score[idx]
        n_pos = np.sum(y_b)
        if n_pos == 0 or n_pos == len(y_b):
            boot_aurocs[i] = np.nan
            continue
        try:
            boot_aurocs[i] = roc_auc_score(y_b, s_b)
        except ValueError:
            boot_aurocs[i] = np.nan

    boot_valid = boot_aurocs[~np.isnan(boot_aurocs)]

    # Jackknife (serial, within worker)
    jack = _jackknife_aurocs(y_true, y_score)

    # BCa CIs at each level
    cis = {}
    for level in ci_levels:
        lo, hi = bca_ci(boot_valid, auroc, jack, level)
        key = f"ci_{int(level * 100)}"
        cis[key] = [lo, hi]

    return {
        "method": method_name,
        "auroc": auroc,
        "n": int(len(y_true)),
        "n_boot_valid": int(len(boot_valid)),
        **cis,
        "_boot_dist": boot_valid,  # kept in memory for pairwise/effect-size; stripped before JSON
    }


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Simple progress tracker that works with or without tqdm."""

    def __init__(self, total, desc=""):
        self.total = total
        self.desc = desc
        self.completed = 0
        self.start = time.time()
        if HAS_TQDM:
            self.bar = tqdm(total=total, desc=desc, file=sys.stderr)
        else:
            self.bar = None
            print(f"  [{desc}] 0/{total}", end="", flush=True, file=sys.stderr)

    def update(self, n=1):
        self.completed += n
        if self.bar:
            self.bar.update(n)
        else:
            elapsed = time.time() - self.start
            rate = self.completed / elapsed if elapsed > 0 else 0
            eta = (self.total - self.completed) / rate if rate > 0 else 0
            print(f"\r  [{self.desc}] {self.completed}/{self.total}"
                  f"  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                  end="", flush=True, file=sys.stderr)

    def close(self):
        if self.bar:
            self.bar.close()
        else:
            elapsed = time.time() - self.start
            print(f"\r  [{self.desc}] {self.completed}/{self.total}"
                  f"  done in {elapsed:.1f}s", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_group(data, group_label, n_bootstrap, n_workers, ci_levels, seed=42):
    """Run full bootstrap analysis on a group of data (target model or benchmark).

    Returns dict with 'methods', 'pairwise', 'effect_sizes', 'n'.
    """
    # Extract arrays for each method
    method_arrays = {}
    for field, label in METHODS:
        y, s = extract_arrays(data, field)
        if y is not None:
            method_arrays[label] = (y, s, field)

    if not method_arrays:
        return None

    # --- Step 1: Per-method bootstrap + BCa CIs ---
    tasks = []
    for label, (y, s, _) in method_arrays.items():
        tasks.append((label, y, s, n_bootstrap, ci_levels, seed))

    # Parallelize across methods
    n_method_workers = min(len(tasks), n_workers)
    if n_method_workers > 1:
        with Pool(processes=n_method_workers) as pool:
            method_results = pool.map(_compute_method_bootstrap, tasks)
    else:
        method_results = [_compute_method_bootstrap(t) for t in tasks]

    # Index by name for pairwise access
    result_by_name = {r["method"]: r for r in method_results}

    # --- Step 2: Pairwise comparisons ---
    # Find the common set of samples across all methods for fair comparison.
    # For simplicity, do pairwise on the intersection of each pair.
    calibrator_label = "Calibrator"
    pairwise_results = []
    method_names = sorted(method_arrays.keys())

    for a_name, b_name in combinations(method_names, 2):
        y_a, s_a, field_a = method_arrays[a_name]
        y_b, s_b, field_b = method_arrays[b_name]

        # Use the data intersection (samples valid for both methods)
        valid_both = [
            d for d in data
            if d.get(field_a) is not None
            and d.get(field_b) is not None
            and d.get("is_correct") is not None
        ]
        if len(valid_both) < 20:
            continue

        y_common = np.array([d["is_correct"] for d in valid_both], dtype=np.int32)
        sa_common = np.array([d[field_a] for d in valid_both], dtype=np.float64)
        sb_common = np.array([d[field_b] for d in valid_both], dtype=np.float64)

        if len(np.unique(y_common)) < 2:
            continue

        pw = pairwise_bootstrap(
            y_common, sa_common, sb_common,
            n_bootstrap=min(n_bootstrap, 10000),  # cap pairwise at 10k for speed
            n_workers=n_workers,
            seed=seed,
        )
        pairwise_results.append({
            "a": a_name,
            "b": b_name,
            **pw,
        })

    # --- Step 3: Effect sizes (Cohen's d) for Calibrator vs each baseline ---
    effect_sizes = []
    if calibrator_label in result_by_name:
        cal_boot = result_by_name[calibrator_label]["_boot_dist"]
        for label in method_names:
            if label == calibrator_label:
                continue
            if label in result_by_name:
                base_boot = result_by_name[label]["_boot_dist"]
                d = cohens_d(cal_boot, base_boot)
                effect_sizes.append({
                    "baseline": label,
                    "cohens_d": d,
                })

    # --- Clean up: remove _boot_dist before serialization ---
    methods_clean = []
    for r in method_results:
        rc = {k: v for k, v in r.items() if not k.startswith("_")}
        methods_clean.append(rc)

    # Sort by AUROC descending
    methods_clean.sort(key=lambda x: x["auroc"], reverse=True)

    return {
        "methods": methods_clean,
        "pairwise": pairwise_results,
        "effect_sizes": effect_sizes,
        "n": len(data),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Exhaustive bootstrap confidence intervals for UQ evaluation"
    )
    parser.add_argument(
        "--scored_dir",
        default="data/use_cases/scored_test_only/",
        help="Directory with {target}_scored.jsonl files",
    )
    parser.add_argument(
        "--output",
        default="data/use_cases/results_test_only/exhaustive_bootstrap.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=100000,
        help="Number of bootstrap iterations (default: 100000)",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Smoke test mode: 500 iterations, first 200 samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Smoke test overrides
    if args.smoke_test:
        args.n_bootstrap = 500
        print("[SMOKE TEST] Using 500 bootstrap iterations, max 200 samples per target",
              file=sys.stderr)

    # Determine worker count
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    print(f"Configuration:", file=sys.stderr)
    print(f"  n_bootstrap:  {args.n_bootstrap:,}", file=sys.stderr)
    print(f"  CI levels:    {CI_LEVELS}", file=sys.stderr)
    print(f"  n_workers:    {n_workers}", file=sys.stderr)
    print(f"  scored_dir:   {args.scored_dir}", file=sys.stderr)
    print(f"  output:       {args.output}", file=sys.stderr)
    print(f"  seed:         {args.seed}", file=sys.stderr)
    print(file=sys.stderr)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    t_start = time.time()

    # -----------------------------------------------------------------------
    # Per-target analysis
    # -----------------------------------------------------------------------
    per_target = {}
    all_data = []  # accumulate for combined analysis

    for target in TARGETS:
        path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if not path.exists():
            print(f"Skipping {target}: {path} not found", file=sys.stderr)
            continue

        data = load_scored(path)
        if args.smoke_test:
            data = data[:200]

        print(f"\n{'=' * 70}", file=sys.stderr)
        print(f"Target: {target} ({len(data)} samples, {args.n_bootstrap:,} bootstrap iterations)",
              file=sys.stderr)
        print(f"{'=' * 70}", file=sys.stderr)

        result = analyze_group(data, target, args.n_bootstrap, n_workers, CI_LEVELS, args.seed)
        if result is not None:
            per_target[target] = result
            _print_table(result, target)

        all_data.extend(data)

    # -----------------------------------------------------------------------
    # Per-benchmark analysis
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}", file=sys.stderr)
    print(f"Per-benchmark breakdown", file=sys.stderr)
    print(f"{'=' * 70}", file=sys.stderr)

    benchmarks = sorted(set(d.get("benchmark", "unknown") for d in all_data))
    per_benchmark = {}

    for bench in benchmarks:
        bench_data = [d for d in all_data if d.get("benchmark") == bench]
        if len(bench_data) < 20:
            print(f"  Skipping {bench}: only {len(bench_data)} samples", file=sys.stderr)
            continue

        # Use fewer bootstrap iterations for per-benchmark (still plenty)
        n_boot_bench = min(args.n_bootstrap, 10000) if not args.smoke_test else args.n_bootstrap
        result = analyze_group(bench_data, bench, n_boot_bench, n_workers, CI_LEVELS, args.seed)
        if result is not None:
            per_benchmark[bench] = result
            _print_table_compact(result, bench)

    # -----------------------------------------------------------------------
    # Combined analysis (all targets pooled)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}", file=sys.stderr)
    print(f"Combined (all targets, {len(all_data)} samples)", file=sys.stderr)
    print(f"{'=' * 70}", file=sys.stderr)

    combined = analyze_group(all_data, "combined", args.n_bootstrap, n_workers, CI_LEVELS, args.seed)
    if combined is not None:
        _print_table(combined, "Combined")

    # -----------------------------------------------------------------------
    # Assemble output
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start

    output = {
        "config": {
            "n_bootstrap": args.n_bootstrap,
            "ci_levels": CI_LEVELS,
            "ci_method": "BCa",
            "seed": args.seed,
            "n_workers": n_workers,
            "smoke_test": args.smoke_test,
            "elapsed_seconds": round(elapsed, 1),
        },
        "per_target": per_target,
        "per_benchmark": per_benchmark,
        "combined": combined,
    }

    # Also extract top-level effect sizes from combined
    if combined and combined.get("effect_sizes"):
        output["effect_sizes"] = combined["effect_sizes"]

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}", file=sys.stderr)
    print(f"Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _print_table(result, label):
    """Print a detailed table of results."""
    methods = result["methods"]
    print(f"\n  {'Method':<20s} {'AUROC':>8s} {'90% CI':>18s} "
          f"{'95% CI':>18s} {'99% CI':>18s} {'N':>6s}", file=sys.stderr)
    print(f"  {'-'*20} {'-'*8} {'-'*18} {'-'*18} {'-'*18} {'-'*6}", file=sys.stderr)

    for m in methods:
        ci90 = m.get("ci_90", [float("nan"), float("nan")])
        ci95 = m.get("ci_95", [float("nan"), float("nan")])
        ci99 = m.get("ci_99", [float("nan"), float("nan")])
        marker = " ***" if m["method"] == "Calibrator" else ""
        print(f"  {m['method']:<20s} {m['auroc']:>8.4f} "
              f"[{ci90[0]:.4f},{ci90[1]:.4f}] "
              f"[{ci95[0]:.4f},{ci95[1]:.4f}] "
              f"[{ci99[0]:.4f},{ci99[1]:.4f}] "
              f"{m['n']:>6d}{marker}", file=sys.stderr)

    # Effect sizes
    if result.get("effect_sizes"):
        print(f"\n  Cohen's d (Calibrator vs baseline):", file=sys.stderr)
        for es in result["effect_sizes"]:
            magnitude = _d_magnitude(es["cohens_d"])
            print(f"    vs {es['baseline']:<16s}: d = {es['cohens_d']:>7.3f} ({magnitude})",
                  file=sys.stderr)

    # Pairwise (just show Calibrator vs others)
    if result.get("pairwise"):
        print(f"\n  Pairwise P(A > B) (selected):", file=sys.stderr)
        for pw in result["pairwise"]:
            if pw["a"] == "Calibrator" or pw["b"] == "Calibrator":
                print(f"    {pw['a']:<16s} vs {pw['b']:<16s}: "
                      f"mean_diff={pw['mean_diff']:>+.4f}, "
                      f"P(A>B)={pw['p_a_better']:.4f}",
                      file=sys.stderr)


def _print_table_compact(result, label):
    """Print a compact one-line-per-method table for per-benchmark results."""
    methods = result["methods"]
    cal = next((m for m in methods if m["method"] == "Calibrator"), None)
    if cal:
        ci95 = cal.get("ci_95", [float("nan"), float("nan")])
        print(f"  {label:<25s}  n={result['n']:<5d}  "
              f"AUROC={cal['auroc']:.4f}  "
              f"95% CI=[{ci95[0]:.4f},{ci95[1]:.4f}]",
              file=sys.stderr)
    else:
        print(f"  {label:<25s}  n={result['n']:<5d}  (no calibrator scores)",
              file=sys.stderr)


def _d_magnitude(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    elif d < 1.2:
        return "large"
    else:
        return "very large"


if __name__ == "__main__":
    main()
