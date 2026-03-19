#!/usr/bin/env python3
"""Filter scored JSONL files to test-set-only samples, then re-run all use cases.

Reads the train/test split from split_info.json and keeps only samples whose
`id` field appears in `test_ids`.  Then re-runs every use-case script on the
filtered data and compiles a comparison table against the original results.

Usage:
    python scripts/filter_test_only.py
    python scripts/filter_test_only.py --smoke_test
    python scripts/filter_test_only.py --skip_filter
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PYTHON = "/scratch/khayes/.conda/envs/uq_eval/bin/python"
PROJECT_ROOT = "/scratch/khayes/LLM"

TARGET_NAMES = ["gpt5mini", "gpt52", "qwen35"]
TARGET_FILES = {t: f"{t}_scored.jsonl" for t in TARGET_NAMES}

# Use-case scripts in the order they should be run.
# (display_name, script_path_or_None, output_json_filename)
# script_path=None means "resolve at runtime" (UC3 has multiple candidate names).
UC_SCRIPTS = [
    ("UC1 Selective Prediction",    "scripts/uc1_selective_prediction.py",       "uc1_results.json"),
    ("UC2 Model Routing",           "scripts/uc2_model_routing.py",             "uc2_results.json"),
    ("UC3 Error Detection",         None,                                        "uc3_results.json"),
    ("UC4 Difficulty Estimation",   "scripts/uc4_difficulty_estimation.py",      "uc4_results.json"),
    ("UC5 UQ Reward Model",         "scripts/uc5_uq_reward_model.py",           "uc5_results.json"),
    ("UC8 OOD Detection",           "scripts/uc8_ood_detection.py",             "uc8_results.json"),
    ("UC9 Annotation Priority",     "scripts/uc9_annotation_prioritization.py", "uc9_results.json"),
    ("UC-A DPO Reward",             "scripts/uc_a_dpo_reward.py",               "uc_a_results.json"),
    ("UC-B Best-of-N",              "scripts/uc_b_best_of_n.py",                "uc_b_results.json"),
    ("UC-C Data Curation",          "scripts/uc_c_data_curation.py",            "uc_c_results.json"),
    ("UC-D Agent Steps",            "scripts/uc_d_agent_steps.py",              "uc_d_results.json"),
]


def resolve_uc3_script():
    """UC3 can be uc3_error_detection.py, uc3_error_flagging.py, or
    uc3_hallucination_detection.py -- pick whichever exists (in priority order)."""
    candidates = [
        "scripts/uc3_error_detection.py",
        "scripts/uc3_error_flagging.py",
        "scripts/uc3_hallucination_detection.py",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(PROJECT_ROOT, c)):
            return c
    return None


# ---------------------------------------------------------------------------
# Step 1 -- Filter scored JSONL to test-only
# ---------------------------------------------------------------------------

def load_split_ids(split_info_path: str):
    """Load train and test question IDs from split_info.json.

    Prefers question-level IDs (split_method='question_level') which guarantee
    zero overlap.  Falls back to sample-level IDs for legacy split files but
    prints a loud warning.
    """
    print(f"[filter] Loading split info from {split_info_path}")
    with open(split_info_path) as f:
        info = json.load(f)

    if info.get("split_method") == "question_level":
        train_ids = set(info["train_question_ids"])
        test_ids = set(info["test_question_ids"])
        overlap = train_ids & test_ids
        assert len(overlap) == 0, (
            f"Question-level split has {len(overlap)} overlapping IDs — this should never happen!"
        )
        print(f"[filter] Question-level split: {len(test_ids):,} test questions, "
              f"{len(train_ids):,} train questions, 0 overlap ✓")
    else:
        # Legacy split — sample-level IDs may have question overlap
        train_ids = set(info["train_ids"])
        test_ids = set(info["test_ids"])
        overlap = train_ids & test_ids
        if overlap:
            print(f"[filter] ⚠️ WARNING: Legacy split has {len(overlap)} IDs in both train and test!")
            print(f"[filter] ⚠️ Retrain with question-level split to fix this.")
        print(f"[filter] Loaded {len(test_ids):,} test IDs, "
              f"{len(train_ids):,} train IDs (legacy sample-level split)")

    return train_ids, test_ids


def filter_scored_files(scored_dir: str, output_dir: str,
                        train_ids: set, test_ids: set):
    """Filter each scored JSONL to keep only test-set samples.

    Match rule: sample is kept if its `id` field appears in test_ids.

    Returns dict {target_name: (total, kept, train_removed, unmatched)}.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    counts = {}

    for target, fname in TARGET_FILES.items():
        src = os.path.join(scored_dir, fname)
        dst = os.path.join(output_dir, fname)
        if not os.path.isfile(src):
            print(f"[filter] WARNING: {src} not found -- skipping {target}")
            counts[target] = (0, 0, 0, 0)
            continue

        all_data = []
        with open(src) as f:
            for line in f:
                all_data.append(json.loads(line))

        test_data = []
        train_count = 0
        unmatched = 0
        bench_counts = defaultdict(lambda: {"test": 0, "train": 0, "unmatched": 0})

        for d in all_data:
            bare_id = d["id"]
            bench = d.get("benchmark", "unknown")
            # Match using benchmark-prefixed ID to avoid cross-benchmark collisions
            prefixed_id = f"{bench}_{bare_id}"
            # Check prefixed first, fall back to bare for legacy split files
            if prefixed_id in test_ids or (bare_id in test_ids and prefixed_id not in train_ids):
                test_data.append(d)
                bench_counts[bench]["test"] += 1
            elif prefixed_id in train_ids or bare_id in train_ids:
                train_count += 1
                bench_counts[bench]["train"] += 1
            else:
                unmatched += 1
                bench_counts[bench]["unmatched"] += 1

        total = len(all_data)
        kept = len(test_data)
        counts[target] = (total, kept, train_count, unmatched)

        print(f"[filter] {target}: {kept:,}/{total:,} kept as test "
              f"({100 * kept / max(total, 1):.1f}%), "
              f"{train_count:,} train removed, {unmatched:,} unmatched")

        if unmatched > 0:
            print(f"[filter]   WARNING: {unmatched} samples not in train_ids "
                  f"or test_ids (excluding them)")

        # Per-benchmark breakdown
        print(f"[filter]   Per-benchmark (test / train / unmatched):")
        for bench in sorted(bench_counts.keys()):
            c = bench_counts[bench]
            print(f"[filter]     {bench:<25} "
                  f"{c['test']:>4} / {c['train']:>4} / {c['unmatched']:>4}")

        # Write filtered data
        with open(dst, "w") as f:
            for d in test_data:
                f.write(json.dumps(d) + "\n")

    return counts


# ---------------------------------------------------------------------------
# Step 2 -- Direct AUROC on scored data
# ---------------------------------------------------------------------------

def compute_auroc_from_scored(scored_dir: str) -> dict:
    """Compute AUROC (p_correct vs is_correct) from scored files."""
    from sklearn.metrics import roc_auc_score

    results = {}
    all_labels = []
    all_scores = []

    for target, fname in TARGET_FILES.items():
        path = os.path.join(scored_dir, fname)
        if not os.path.isfile(path):
            results[target] = None
            continue
        labels = []
        scores = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                labels.append(int(row["is_correct"]))
                scores.append(float(row["p_correct"]))

        if len(labels) == 0 or len(set(labels)) < 2:
            print(f"[auroc] {target}: insufficient data -- cannot compute AUROC")
            results[target] = None
            continue

        auroc = float(np.round(roc_auc_score(labels, scores), 4))
        results[target] = auroc
        all_labels.extend(labels)
        all_scores.extend(scores)
        print(f"[auroc] {target}: AUROC = {auroc:.4f}  (n={len(labels):,})")

    if len(set(all_labels)) >= 2:
        overall = float(np.round(roc_auc_score(all_labels, all_scores), 4))
        results["overall"] = overall
        print(f"[auroc] overall: AUROC = {overall:.4f}  (n={len(all_labels):,})")
    else:
        results["overall"] = None

    return results


# ---------------------------------------------------------------------------
# Step 3 -- Run use-case scripts
# ---------------------------------------------------------------------------

def run_use_case(name: str, script: str, scored_dir: str,
                 output_dir: str, fig_dir: str) -> dict:
    """Run a single use-case script as a subprocess.

    Returns {"success": bool, "elapsed": float, "error": str|None}.
    """
    abs_script = os.path.join(PROJECT_ROOT, script)
    if not os.path.isfile(abs_script):
        return {"success": False, "elapsed": 0.0,
                "error": f"Script not found: {abs_script}"}

    cmd = [
        PYTHON, abs_script,
        "--scored_dir", scored_dir,
        "--output_dir", output_dir,
        "--fig_dir", fig_dir,
    ]

    print(f"\n{'='*70}")
    print(f"[run] {name}  ({script})")
    print(f"[run] cmd: {' '.join(cmd)}")
    print(f"{'='*70}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=PROJECT_ROOT,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            err_tail = (result.stderr or "")[-2000:]
            print(f"[run] FAILED ({elapsed:.1f}s)  rc={result.returncode}")
            print(f"[run] stderr (last 2000 chars):\n{err_tail}")
            return {"success": False, "elapsed": elapsed,
                    "error": f"rc={result.returncode}: {err_tail[-500:]}"}
        else:
            print(f"[run] OK ({elapsed:.1f}s)")
            # Print last few lines of stdout for context
            stdout_lines = (result.stdout or "").strip().split("\n")
            for line in stdout_lines[-5:]:
                print(f"  > {line}")
            return {"success": True, "elapsed": elapsed, "error": None}
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"[run] TIMEOUT after {elapsed:.1f}s")
        return {"success": False, "elapsed": elapsed,
                "error": "Timed out after 600s"}
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"[run] EXCEPTION: {exc}")
        return {"success": False, "elapsed": elapsed, "error": str(exc)}


# ---------------------------------------------------------------------------
# Step 4 -- Extract key metrics and build comparison table
# ---------------------------------------------------------------------------

def _safe_get(d, *keys):
    """Safely traverse nested dict."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def extract_key_metrics(results_dir: str) -> dict:
    """Extract headline metrics from each UC result file.

    Returns {metric_label: value}.
    """
    metrics = {}

    def _load(fname):
        path = os.path.join(results_dir, fname)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return json.load(f)

    # UC1 -- AURC and coverage@90 per target
    d = _load("uc1_results.json")
    if d:
        for t in TARGET_NAMES:
            aurc = _safe_get(d, t, "Calibrator P(correct)", "aurc")
            if aurc is not None:
                metrics[f"UC1 AURC ({t})"] = aurc
            cov90 = _safe_get(d, t, "Calibrator P(correct)", "coverage_at_90")
            if cov90 is not None:
                metrics[f"UC1 Cov@90 ({t})"] = cov90

    # UC2 -- cost savings / routing accuracy
    d = _load("uc2_results.json")
    if d:
        for key_path, label in [
            (("methods", "Calibrator", "cost_savings"), "UC2 Cost Savings"),
            (("methods", "Calibrator", "accuracy"), "UC2 Routing Acc"),
            (("cost_savings",), "UC2 Cost Savings"),
            (("routing_accuracy",), "UC2 Routing Acc"),
            (("cost_reduction",), "UC2 Cost Reduction"),
        ]:
            v = _safe_get(d, *key_path)
            if v is not None and label not in metrics:
                metrics[label] = v

    # UC3 -- best F1 and AUROC per target
    d = _load("uc3_results.json")
    if d:
        for t in TARGET_NAMES:
            f1 = _safe_get(d, t, "calibrator", "best_f1")
            if f1 is not None:
                metrics[f"UC3 Best F1 ({t})"] = f1
            auroc = _safe_get(d, t, "calibrator", "auroc")
            if auroc is not None:
                metrics[f"UC3 AUROC ({t})"] = auroc

    # UC4 -- rank correlation per target
    d = _load("uc4_results.json")
    if d:
        for t in TARGET_NAMES:
            rc = _safe_get(d, t, "rank_correlation")
            if rc is not None:
                metrics[f"UC4 Rank Corr ({t})"] = rc

    # UC5 -- pairwise ranking accuracy (p_correct method)
    d = _load("uc5_results.json")
    if d:
        pa = _safe_get(d, "pairwise_ranking", "p_correct", "accuracy")
        if pa is not None:
            metrics["UC5 Pairwise Acc"] = pa

    # UC8 -- per-target benchmark-level metrics
    d = _load("uc8_results.json")
    if d:
        for t in TARGET_NAMES:
            bl = _safe_get(d, t, "benchmark_level")
            if isinstance(bl, dict):
                for mk in ["alert_f1", "f1", "auroc", "mean_auroc"]:
                    v = bl.get(mk)
                    if v is not None:
                        metrics[f"UC8 {mk} ({t})"] = v
                        break

    # UC9 -- AUEDR per target (Calibrator method)
    d = _load("uc9_results.json")
    if d:
        for t in TARGET_NAMES:
            auedr = _safe_get(d, t, "methods", "Calibrator", "auedr")
            if auedr is not None:
                metrics[f"UC9 AUEDR ({t})"] = auedr

    # UC-A -- calibrator pairwise and informative accuracy
    d = _load("uc_a_results.json")
    if d:
        pa = _safe_get(d, "method_comparison", "calibrator", "pairwise_accuracy")
        if pa is not None:
            metrics["UC-A Pairwise Acc"] = pa
        ia = _safe_get(d, "method_comparison", "calibrator", "informative_accuracy")
        if ia is not None:
            metrics["UC-A Inform Acc"] = ia
        n = _safe_get(d, "method_comparison", "calibrator", "n_total")
        if n is not None:
            metrics["UC-A N Pairs"] = n

    # UC-B -- best-of-N accuracy (N=3, calibrator)
    d = _load("uc_b_results.json")
    if d:
        for n_key in ["N=3", "N=2"]:
            acc = _safe_get(d, "selection_results", n_key, "calibrator", "accuracy")
            if acc is not None:
                metrics[f"UC-B {n_key} Acc (cal)"] = acc

    # UC-C -- calibrator thresholds (retention at ~90% accuracy)
    d = _load("uc_c_results.json")
    if d:
        for t in TARGET_NAMES:
            thresholds = _safe_get(d, t, "calibrator_thresholds")
            if isinstance(thresholds, list):
                best = None
                for entry in thresholds:
                    if isinstance(entry, dict) and entry.get("accuracy", 0) >= 0.89:
                        if best is None or entry.get("retention", 1) > best.get("retention", 0):
                            best = entry
                if best and best.get("retention") is not None:
                    metrics[f"UC-C Ret@90acc ({t})"] = best["retention"]
            elif isinstance(thresholds, dict):
                ret = thresholds.get("retention_at_90")
                if ret is not None:
                    metrics[f"UC-C Ret@90acc ({t})"] = ret

    # UC-D -- multi-step overall metrics
    d = _load("uc_d_results.json")
    if d:
        for t in TARGET_NAMES:
            ms = _safe_get(d, t, "multi_step_overall")
            if isinstance(ms, dict):
                for mk in ["auroc", "correlation", "r_squared"]:
                    v = ms.get(mk)
                    if v is not None:
                        metrics[f"UC-D {mk} ({t})"] = v
                        break

    return metrics


def build_comparison_table(old_metrics: dict, new_metrics: dict) -> list:
    """Build comparison rows: metric, old, new, delta."""
    all_keys = sorted(set(list(old_metrics.keys()) + list(new_metrics.keys())))
    rows = []
    for key in all_keys:
        old_val = old_metrics.get(key)
        new_val = new_metrics.get(key)
        if old_val is None and new_val is None:
            continue
        delta = None
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            delta = new_val - old_val
        rows.append({
            "metric": key,
            "old": old_val,
            "new": new_val,
            "delta": delta,
        })
    return rows


def print_comparison_table(rows: list, auroc_old: dict, auroc_new: dict):
    """Pretty-print the comparison table."""
    print("\n")
    print("=" * 90)
    print("COMPARISON: All Data  vs  Test-Only")
    print("=" * 90)

    # AUROC section
    print("\n--- Direct AUROC (p_correct vs is_correct) ---")
    header = f"{'Target':<12} {'All Data':>12} {'Test-Only':>12} {'Delta':>12}"
    print(header)
    print("-" * len(header))
    for t in TARGET_NAMES + ["overall"]:
        old = auroc_old.get(t)
        new = auroc_new.get(t)
        old_s = f"{old:.4f}" if isinstance(old, float) else "N/A"
        new_s = f"{new:.4f}" if isinstance(new, float) else "N/A"
        if isinstance(old, float) and isinstance(new, float):
            delta_s = f"{new - old:+.4f}"
        else:
            delta_s = "---"
        print(f"{t:<12} {old_s:>12} {new_s:>12} {delta_s:>12}")

    # Use-case metrics section
    if rows:
        print("\n--- Use-Case Metrics ---")
        header = f"{'Metric':<30} {'All Data':>12} {'Test-Only':>12} {'Delta':>12}"
        print(header)
        print("-" * len(header))
        for row in rows:
            old_val = row["old"]
            new_val = row["new"]
            delta = row["delta"]
            if isinstance(old_val, float):
                old_s = f"{old_val:.4f}"
            elif old_val is not None:
                old_s = str(old_val)
            else:
                old_s = "N/A"
            if isinstance(new_val, float):
                new_s = f"{new_val:.4f}"
            elif new_val is not None:
                new_s = str(new_val)
            else:
                new_s = "N/A"
            if isinstance(delta, float):
                delta_s = f"{delta:+.4f}"
            else:
                delta_s = "---"
            metric_name = row["metric"][:30]
            print(f"{metric_name:<30} {old_s:>12} {new_s:>12} {delta_s:>12}")
    else:
        print("\n--- No use-case metrics to compare ---")

    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Filter scored data to test-only, re-run use cases, compare.")
    parser.add_argument("--split_info",
                        default="uq_models/best_unified/split_info.json",
                        help="Path to split_info.json with train_ids/test_ids")
    parser.add_argument("--scored_dir",
                        default="data/use_cases/CONTAMINATED_scored_unified",
                        help="Directory with original scored JSONL files (contaminated, for filtering)")
    parser.add_argument("--output_scored",
                        default="data/use_cases/scored_test_only",
                        help="Directory for filtered scored JSONL files")
    parser.add_argument("--output_dir",
                        default="data/use_cases/results_test_only",
                        help="Directory for test-only use-case results")
    parser.add_argument("--fig_dir",
                        default="figures/use_cases_test_only",
                        help="Directory for test-only figures")
    parser.add_argument("--old_results_dir",
                        default="data/use_cases/CONTAMINATED_results_unified",
                        help="Directory with original (all-data) results for comparison")
    parser.add_argument("--skip_filter", action="store_true",
                        help="Skip filtering step, use already-filtered data")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Only run first 2 use-case scripts")
    args = parser.parse_args()

    # Resolve to absolute paths
    for attr in ["split_info", "scored_dir", "output_scored", "output_dir",
                 "fig_dir", "old_results_dir"]:
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, os.path.join(PROJECT_ROOT, val))

    # Resolve UC3 script (pick first candidate that exists)
    uc3_script = resolve_uc3_script()
    for i, (name, script, out_json) in enumerate(UC_SCRIPTS):
        if script is None:  # UC3 placeholder
            UC_SCRIPTS[i] = (name, uc3_script, out_json)

    # Check which UC scripts actually exist
    print("[check] Checking which use-case scripts exist...")
    for name, script, _ in UC_SCRIPTS:
        if script is None:
            print(f"  [MISSING] {name}: no candidate script found")
        elif os.path.isfile(os.path.join(PROJECT_ROOT, script)):
            print(f"  [OK]      {name}: {script}")
        else:
            print(f"  [MISSING] {name}: {script}")

    # ------------------------------------------------------------------
    # Step 1: Filter scored files to test-only
    # ------------------------------------------------------------------
    if not args.skip_filter:
        print(f"\n{'='*70}")
        print("STEP 1: Filtering scored files to test-only samples")
        print(f"{'='*70}")
        train_ids, test_ids = load_split_ids(args.split_info)
        counts = filter_scored_files(args.scored_dir, args.output_scored,
                                     train_ids, test_ids)
        total_kept = sum(c[1] for c in counts.values())
        total_all = sum(c[0] for c in counts.values())
        print(f"\n[filter] TOTAL: {total_kept:,}/{total_all:,} samples kept "
              f"across all targets")
    else:
        print("\n[filter] Skipping filter step (--skip_filter)")

    # Create output dirs
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 2: Compute direct AUROC on filtered data
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 2: Computing AUROC on filtered (test-only) data")
    print(f"{'='*70}")
    auroc_new = compute_auroc_from_scored(args.output_scored)

    print("\n[auroc] Computing AUROC on original (all data) for comparison...")
    auroc_old = compute_auroc_from_scored(args.scored_dir)

    # ------------------------------------------------------------------
    # Step 3: Run use-case scripts
    # ------------------------------------------------------------------
    scripts_to_run = list(UC_SCRIPTS)
    if args.smoke_test:
        scripts_to_run = scripts_to_run[:2]
        print(f"\n[smoke_test] Running only first 2 scripts")

    print(f"\n{'='*70}")
    print(f"STEP 3: Running {len(scripts_to_run)} use-case scripts")
    print(f"{'='*70}")
    print(f"  Scored dir:  {args.output_scored}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Fig dir:     {args.fig_dir}")

    run_results = {}
    for name, script, out_json in scripts_to_run:
        if script is None:
            print(f"\n[run] SKIP {name}: no script found")
            run_results[name] = {"success": False, "elapsed": 0,
                                 "error": "No script found"}
            continue
        result = run_use_case(name, script, args.output_scored,
                              args.output_dir, args.fig_dir)
        run_results[name] = result

    # ------------------------------------------------------------------
    # Step 4: Summary of script runs
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 4: Use-case script run summary")
    print(f"{'='*70}")
    n_ok = sum(1 for r in run_results.values() if r["success"])
    n_fail = sum(1 for r in run_results.values() if not r["success"])
    print(f"  Succeeded: {n_ok}  |  Failed: {n_fail}")
    for name, result in run_results.items():
        status = "OK" if result["success"] else "FAIL"
        elapsed = result["elapsed"]
        err = f"  ({result['error'][:80]})" if result.get("error") else ""
        print(f"  [{status}] {name:<30} {elapsed:6.1f}s{err}")

    # ------------------------------------------------------------------
    # Step 5: Extract metrics and build comparison table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 5: Comparison table (all data vs test-only)")
    print(f"{'='*70}")
    old_metrics = extract_key_metrics(args.old_results_dir)
    new_metrics = extract_key_metrics(args.output_dir)
    comparison = build_comparison_table(old_metrics, new_metrics)

    print_comparison_table(comparison, auroc_old, auroc_new)

    # Save comparison to JSON
    comparison_output = {
        "auroc_all_data": auroc_old,
        "auroc_test_only": auroc_new,
        "use_case_comparison": comparison,
        "run_results": {
            name: {
                "success": r["success"],
                "elapsed": r["elapsed"],
                "error": r.get("error"),
            }
            for name, r in run_results.items()
        },
    }
    comp_path = os.path.join(args.output_dir, "comparison_all_vs_test_only.json")
    with open(comp_path, "w") as f:
        json.dump(comparison_output, f, indent=2)
    print(f"\n[done] Saved comparison to {comp_path}")
    print("[done] Filtered scored data in: " + args.output_scored)
    print("[done] Test-only results in:    " + args.output_dir)
    print("[done] Test-only figures in:    " + args.fig_dir)


if __name__ == "__main__":
    main()
