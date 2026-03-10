#!/usr/bin/env python3
"""Compute AUROC on the near-duplicate-free subset of v3 test data.

Reads the contamination report to identify test samples with near-duplicates
in training, removes them, and computes AUROC on the clean subset.

Usage:
    python scripts/deduped_auroc.py
    python scripts/deduped_auroc.py --contamination_report data/use_cases/results_test_only_v3/contamination_report_v3.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contamination_report",
        default="data/use_cases/results_test_only_v3/contamination_report_v3.json",
    )
    parser.add_argument(
        "--scored_dir",
        default="data/use_cases/scored_test_only_v3",
    )
    parser.add_argument(
        "--output",
        default="data/use_cases/results_test_only_v3/deduped_auroc_v3.json",
    )
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Jaccard threshold above which to exclude")
    args = parser.parse_args()

    # Load contamination report
    print("Loading contamination report...")
    with open(args.contamination_report) as f:
        report = json.load(f)

    # Extract test IDs that have near-duplicates in training
    contaminated_test_ids = set()

    # From per_test_max_jaccard (each test sample's max Jaccard vs any train sample)
    for entry in report.get("per_test_max_jaccard", []):
        if entry.get("max_jaccard", 0) >= args.threshold:
            test_id = entry.get("test_id")
            if test_id:
                contaminated_test_ids.add(test_id)

    print(f"  Found {len(contaminated_test_ids)} contaminated test IDs (Jaccard >= {args.threshold})")

    # Load scored data
    all_samples = []
    for target in ["gpt5mini", "gpt52", "qwen35"]:
        fpath = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            samples = [json.loads(line) for line in f]
        for s in samples:
            s["target_model"] = target
        all_samples.append((target, samples))

    # Compute AUROC on full and deduped subsets
    results = {"threshold": args.threshold}

    all_labels_full = []
    all_scores_full = []
    all_labels_clean = []
    all_scores_clean = []

    results["per_target"] = {}
    for target, samples in all_samples:
        full_labels = [int(s["is_correct"]) for s in samples]
        full_scores = [float(s["p_correct"]) for s in samples]

        clean_samples = [s for s in samples if s.get("id", "") not in contaminated_test_ids]
        clean_labels = [int(s["is_correct"]) for s in clean_samples]
        clean_scores = [float(s["p_correct"]) for s in clean_samples]

        all_labels_full.extend(full_labels)
        all_scores_full.extend(full_scores)
        all_labels_clean.extend(clean_labels)
        all_scores_clean.extend(clean_scores)

        full_auroc = roc_auc_score(full_labels, full_scores) if len(set(full_labels)) > 1 else None
        clean_auroc = roc_auc_score(clean_labels, clean_scores) if len(set(clean_labels)) > 1 else None

        results["per_target"][target] = {
            "n_full": len(samples),
            "n_clean": len(clean_samples),
            "n_removed": len(samples) - len(clean_samples),
            "pct_removed": (len(samples) - len(clean_samples)) / len(samples) * 100,
            "auroc_full": float(full_auroc) if full_auroc else None,
            "auroc_clean": float(clean_auroc) if clean_auroc else None,
            "auroc_delta": float(clean_auroc - full_auroc) if clean_auroc and full_auroc else None,
        }

        print(f"  {target}: {len(samples)} -> {len(clean_samples)} "
              f"({len(samples) - len(clean_samples)} removed, "
              f"{(len(samples) - len(clean_samples))/len(samples)*100:.1f}%)")
        if full_auroc and clean_auroc:
            print(f"    AUROC: {full_auroc:.4f} -> {clean_auroc:.4f} (delta: {clean_auroc - full_auroc:+.4f})")

    # Combined
    full_auroc = roc_auc_score(all_labels_full, all_scores_full)
    clean_auroc = roc_auc_score(all_labels_clean, all_scores_clean) if len(set(all_labels_clean)) > 1 else None

    results["combined"] = {
        "n_full": len(all_labels_full),
        "n_clean": len(all_labels_clean),
        "n_removed": len(all_labels_full) - len(all_labels_clean),
        "pct_removed": (len(all_labels_full) - len(all_labels_clean)) / len(all_labels_full) * 100,
        "auroc_full": float(full_auroc),
        "auroc_clean": float(clean_auroc) if clean_auroc else None,
        "auroc_delta": float(clean_auroc - full_auroc) if clean_auroc else None,
    }

    print(f"\n  Combined: {len(all_labels_full)} -> {len(all_labels_clean)}")
    print(f"    AUROC: {full_auroc:.4f} -> {clean_auroc:.4f} (delta: {clean_auroc - full_auroc:+.4f})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
