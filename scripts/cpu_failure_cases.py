#!/usr/bin/env python3
"""Curate interesting failure and success cases for qualitative analysis.

Reads scored JSONL files from the test-only scored directory and identifies:
1. False positives  -- high calibrator confidence on wrong answers
2. False negatives  -- low calibrator confidence on correct answers
3. Calibrator-verbalized disagreements -- largest |p_correct - verbalized_confidence|
4. Calibrator successes -- low p_correct on wrong answers that verbalized confidence rated highly

Also computes aggregate statistics (per-benchmark distribution, token length
comparisons via Mann-Whitney U).

Output: data/use_cases/results_test_only/failure_cases.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_scored_jsonl(path: str) -> list[dict]:
    """Load a scored JSONL file, skipping malformed lines."""
    records = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  [WARN] Skipping malformed line {lineno} in {path}", file=sys.stderr)
    return records


def load_all_scored(scored_dir: str) -> list[dict]:
    """Load the three canonical scored files."""
    files = [
        "gpt5mini_scored.jsonl",
        "gpt52_scored.jsonl",
        "qwen35_scored.jsonl",
    ]
    all_records = []
    for fname in files:
        fpath = os.path.join(scored_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [WARN] Missing {fpath}, skipping", file=sys.stderr)
            continue
        records = load_scored_jsonl(fpath)
        print(f"  Loaded {len(records):,} records from {fname}")
        all_records.extend(records)
    return all_records


# ---------------------------------------------------------------------------
# Curation logic
# ---------------------------------------------------------------------------

def _safe_float(val, default=0.5):
    """Safely convert a value to float, returning default on failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _make_entry(record: dict, rank: int) -> dict:
    """Extract the fields we want for each curated example."""
    return {
        "id": record.get("id", ""),
        "benchmark": record.get("benchmark", ""),
        "target_model": record.get("target_model", ""),
        "is_correct": int(record.get("is_correct", 0)),
        "p_correct": _safe_float(record.get("p_correct")),
        "verbalized_confidence": _safe_float(record.get("verbalized_confidence")),
        "question_preview": record.get("question_preview", ""),
        "response_preview": record.get("response_preview", ""),
        "rank_in_category": rank,
    }


def curate_false_positives(records: list[dict], top_k: int = 50) -> list[dict]:
    """High calibrator confidence on WRONG answers (is_correct=0, sorted by p_correct desc)."""
    wrong = [r for r in records if int(r.get("is_correct", 0)) == 0]
    wrong.sort(key=lambda r: _safe_float(r.get("p_correct")), reverse=True)
    return [_make_entry(r, i + 1) for i, r in enumerate(wrong[:top_k])]


def curate_false_negatives(records: list[dict], top_k: int = 50) -> list[dict]:
    """Low calibrator confidence on CORRECT answers (is_correct=1, sorted by p_correct asc)."""
    correct = [r for r in records if int(r.get("is_correct", 0)) == 1]
    correct.sort(key=lambda r: _safe_float(r.get("p_correct")))
    return [_make_entry(r, i + 1) for i, r in enumerate(correct[:top_k])]


def curate_disagreements(records: list[dict], top_k: int = 50) -> list[dict]:
    """Largest |p_correct - verbalized_confidence| — shows when calibrator adds value."""
    def disagreement(r):
        return abs(_safe_float(r.get("p_correct")) - _safe_float(r.get("verbalized_confidence")))

    ranked = sorted(records, key=disagreement, reverse=True)
    return [_make_entry(r, i + 1) for i, r in enumerate(ranked[:top_k])]


def curate_calibrator_successes(records: list[dict], top_k: int = 50) -> list[dict]:
    """Calibrator correctly flags wrong answers that the model self-rated highly.

    Criteria: is_correct=0, verbalized_confidence > 0.8, sorted by p_correct asc
    (lower p_correct = calibrator more correctly confident it is wrong).
    """
    candidates = [
        r for r in records
        if int(r.get("is_correct", 0)) == 0
        and _safe_float(r.get("verbalized_confidence")) > 0.8
    ]
    candidates.sort(key=lambda r: _safe_float(r.get("p_correct")))
    return [_make_entry(r, i + 1) for i, r in enumerate(candidates[:top_k])]


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def compute_aggregate_stats(
    records: list[dict],
    false_positives: list[dict],
    false_negatives: list[dict],
    disagreements: list[dict],
    successes: list[dict],
) -> dict:
    """Compute per-benchmark distribution and token-length analysis."""

    # --- Per-benchmark distribution of failure cases ---
    fp_benchmarks = Counter(e["benchmark"] for e in false_positives)
    fn_benchmarks = Counter(e["benchmark"] for e in false_negatives)
    disagree_benchmarks = Counter(e["benchmark"] for e in disagreements)
    success_benchmarks = Counter(e["benchmark"] for e in successes)

    # Total per-benchmark counts for reference
    total_benchmarks = Counter(r.get("benchmark", "") for r in records)

    # --- Token length analysis: failures vs successes ---
    # "Failures" = false positives + false negatives (calibrator got it wrong)
    # "Successes" = calibrator successes (calibrator got it right, model was overconfident)
    failure_ids = set(e["id"] for e in false_positives) | set(e["id"] for e in false_negatives)
    success_ids = set(e["id"] for e in successes)

    failure_input_tokens = []
    failure_output_tokens = []
    success_input_tokens = []
    success_output_tokens = []

    for r in records:
        rid = r.get("id", "")
        inp = _safe_float(r.get("input_tokens"), default=0)
        out = _safe_float(r.get("output_tokens"), default=0)
        if rid in failure_ids:
            failure_input_tokens.append(inp)
            failure_output_tokens.append(out)
        if rid in success_ids:
            success_input_tokens.append(inp)
            success_output_tokens.append(out)

    # Mann-Whitney U tests
    mw_input = None
    mw_output = None
    if failure_input_tokens and success_input_tokens:
        u_inp, p_inp = stats.mannwhitneyu(
            failure_input_tokens, success_input_tokens, alternative="two-sided"
        )
        mw_input = {"U": float(u_inp), "p_value": float(p_inp),
                     "failure_mean": float(np.mean(failure_input_tokens)),
                     "failure_median": float(np.median(failure_input_tokens)),
                     "success_mean": float(np.mean(success_input_tokens)),
                     "success_median": float(np.median(success_input_tokens))}
    if failure_output_tokens and success_output_tokens:
        u_out, p_out = stats.mannwhitneyu(
            failure_output_tokens, success_output_tokens, alternative="two-sided"
        )
        mw_output = {"U": float(u_out), "p_value": float(p_out),
                      "failure_mean": float(np.mean(failure_output_tokens)),
                      "failure_median": float(np.median(failure_output_tokens)),
                      "success_mean": float(np.mean(success_output_tokens)),
                      "success_median": float(np.median(success_output_tokens))}

    # --- Are failures concentrated or spread evenly? ---
    # Use normalized entropy of the benchmark distribution for failure cases
    all_failure_benchmarks = Counter()
    all_failure_benchmarks.update(fp_benchmarks)
    all_failure_benchmarks.update(fn_benchmarks)
    n_failure = sum(all_failure_benchmarks.values())
    n_benchmarks_with_failures = len(all_failure_benchmarks)

    concentration = "N/A"
    norm_entropy = None
    if n_failure > 0 and n_benchmarks_with_failures > 1:
        probs = np.array(list(all_failure_benchmarks.values()), dtype=float) / n_failure
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(n_benchmarks_with_failures)
        norm_entropy = float(entropy / max_entropy)
        # Interpretation: 1.0 = perfectly uniform, 0.0 = all in one benchmark
        if norm_entropy > 0.85:
            concentration = "spread_evenly"
        elif norm_entropy > 0.6:
            concentration = "moderately_spread"
        else:
            concentration = "concentrated"

    return {
        "per_benchmark_distribution": {
            "false_positives": dict(fp_benchmarks.most_common()),
            "false_negatives": dict(fn_benchmarks.most_common()),
            "disagreements": dict(disagree_benchmarks.most_common()),
            "calibrator_successes": dict(success_benchmarks.most_common()),
            "total_per_benchmark": dict(total_benchmarks.most_common()),
        },
        "token_length_analysis": {
            "input_tokens_mann_whitney": mw_input,
            "output_tokens_mann_whitney": mw_output,
        },
        "failure_concentration": {
            "normalized_entropy": norm_entropy,
            "interpretation": concentration,
            "n_benchmarks_with_failures": n_benchmarks_with_failures,
            "total_failure_cases": n_failure,
        },
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_top_cases(cases: list[dict], category: str, n: int = 10):
    """Pretty-print the top N cases for a given category."""
    print(f"\n{'='*80}")
    print(f"  TOP {n} {category.upper()}")
    print(f"{'='*80}")
    for i, c in enumerate(cases[:n]):
        print(f"\n  [{i+1}] id={c['id']}")
        print(f"      benchmark={c['benchmark']}  model={c['target_model']}  "
              f"correct={c['is_correct']}")
        print(f"      p_correct={c['p_correct']:.4f}  "
              f"verbalized={c['verbalized_confidence']:.4f}  "
              f"gap={abs(c['p_correct'] - c['verbalized_confidence']):.4f}")
        q = c.get("question_preview", "")
        if len(q) > 120:
            q = q[:120] + "..."
        print(f"      Q: {q}")
        r = c.get("response_preview", "")
        if len(r) > 120:
            r = r[:120] + "..."
        print(f"      A: {r}")


def print_aggregate_summary(agg: dict):
    """Print a concise summary of aggregate statistics."""
    print(f"\n{'='*80}")
    print("  AGGREGATE STATISTICS")
    print(f"{'='*80}")

    conc = agg["failure_concentration"]
    print(f"\n  Failure concentration: {conc['interpretation']} "
          f"(normalized entropy={conc['normalized_entropy']:.3f}, "
          f"{conc['n_benchmarks_with_failures']} benchmarks, "
          f"{conc['total_failure_cases']} total failure cases)")

    # Top benchmarks with most failures
    fp_dist = agg["per_benchmark_distribution"]["false_positives"]
    fn_dist = agg["per_benchmark_distribution"]["false_negatives"]
    combined = Counter(fp_dist)
    combined.update(fn_dist)
    print("\n  Top benchmarks with most calibrator failures (FP + FN):")
    for bench, count in combined.most_common(10):
        total = agg["per_benchmark_distribution"]["total_per_benchmark"].get(bench, "?")
        print(f"    {bench:25s}  {count:3d} failures / {total} total")

    # Token length
    inp_mw = agg["token_length_analysis"]["input_tokens_mann_whitney"]
    out_mw = agg["token_length_analysis"]["output_tokens_mann_whitney"]
    if inp_mw:
        sig = "***" if inp_mw["p_value"] < 0.001 else ("**" if inp_mw["p_value"] < 0.01
              else ("*" if inp_mw["p_value"] < 0.05 else "ns"))
        print(f"\n  Input tokens: failure median={inp_mw['failure_median']:.0f} "
              f"vs success median={inp_mw['success_median']:.0f}  "
              f"(U={inp_mw['U']:.0f}, p={inp_mw['p_value']:.4g}) {sig}")
    if out_mw:
        sig = "***" if out_mw["p_value"] < 0.001 else ("**" if out_mw["p_value"] < 0.01
              else ("*" if out_mw["p_value"] < 0.05 else "ns"))
        print(f"  Output tokens: failure median={out_mw['failure_median']:.0f} "
              f"vs success median={out_mw['success_median']:.0f}  "
              f"(U={out_mw['U']:.0f}, p={out_mw['p_value']:.4g}) {sig}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Curate failure and success cases for qualitative analysis."
    )
    parser.add_argument(
        "--scored_dir",
        default="data/use_cases/scored_test_only",
        help="Directory containing scored JSONL files (default: data/use_cases/scored_test_only)",
    )
    parser.add_argument(
        "--output",
        default="data/use_cases/results_test_only/failure_cases.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of examples per category (default: 50)",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Smoke test mode: use fewer examples and reduced top_k",
    )
    args = parser.parse_args()

    if args.smoke_test:
        args.top_k = 5
        print("[SMOKE TEST] Running with top_k=5")

    # Load data
    print(f"Loading scored data from {args.scored_dir} ...")
    records = load_all_scored(args.scored_dir)
    if not records:
        print("ERROR: No records loaded. Check scored_dir path.", file=sys.stderr)
        sys.exit(1)

    if args.smoke_test:
        records = records[:200]
        print(f"[SMOKE TEST] Truncated to {len(records)} records")

    n_correct = sum(1 for r in records if int(r.get("is_correct", 0)) == 1)
    n_wrong = len(records) - n_correct
    print(f"Total records: {len(records):,} (correct={n_correct:,}, wrong={n_wrong:,})")

    # Curate categories
    print("\nCurating cases ...")
    false_positives = curate_false_positives(records, args.top_k)
    false_negatives = curate_false_negatives(records, args.top_k)
    disagreements = curate_disagreements(records, args.top_k)
    calibrator_successes = curate_calibrator_successes(records, args.top_k)

    print(f"  False positives:  {len(false_positives)}")
    print(f"  False negatives:  {len(false_negatives)}")
    print(f"  Disagreements:    {len(disagreements)}")
    print(f"  Calibrator wins:  {len(calibrator_successes)}")

    # Aggregate statistics
    print("\nComputing aggregate statistics ...")
    aggregate = compute_aggregate_stats(
        records, false_positives, false_negatives, disagreements, calibrator_successes
    )

    # Print top 10 of each category
    print_top_cases(false_positives, "False Positives (high confidence, wrong answer)")
    print_top_cases(false_negatives, "False Negatives (low confidence, correct answer)")
    print_top_cases(disagreements, "Calibrator-Verbalized Disagreements")
    print_top_cases(calibrator_successes, "Calibrator Successes (catches overconfident wrong answers)")
    print_aggregate_summary(aggregate)

    # Build output
    output = {
        "metadata": {
            "scored_dir": args.scored_dir,
            "total_records": len(records),
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "top_k": args.top_k,
            "smoke_test": args.smoke_test,
        },
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "calibrator_verbalized_disagreements": disagreements,
        "calibrator_successes": calibrator_successes,
        "aggregate_statistics": aggregate,
    }

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {args.output}")
    print(f"  File size: {os.path.getsize(args.output) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
