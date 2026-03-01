#!/usr/bin/env python3
"""Generate per-benchmark AUROC breakdown table for the paper.

Outputs a detailed table showing calibrator performance on each benchmark,
broken down by target model, modality, and difficulty level.

Usage:
    python scripts/per_benchmark_breakdown.py --scored_dir data/use_cases/scored_test_only
"""
import argparse
import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score, brier_score_loss


def load_scored(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_metrics(labels, preds):
    """Compute AUROC, Brier score, and accuracy rate."""
    if len(set(labels)) < 2:
        return {"auroc": None, "brier": None, "acc_rate": np.mean(labels), "n": len(labels)}
    auroc = roc_auc_score(labels, preds)
    brier = brier_score_loss(labels, preds)
    return {
        "auroc": float(auroc),
        "brier": float(brier),
        "acc_rate": float(np.mean(labels)),
        "n": len(labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only")
    parser.add_argument("--output", default="data/use_cases/results_unified/per_benchmark_breakdown.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    VLM_BENCHMARKS = {
        "charxiv", "mmmu", "mmstar", "hallusionbench", "mathverse",
        "mathvision", "mathvista", "realworldqa", "vizwiz", "hle_multimodal", "mmvet",
    }

    targets = ["gpt5mini", "gpt52", "qwen35"]
    results = {}

    # Aggregate all data
    all_data = []
    for target in targets:
        path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if not path.exists():
            print(f"Skipping {target}: {path} not found")
            continue
        data = load_scored(path)
        for d in data:
            d["target"] = target
        all_data.extend(data)
        print(f"Loaded {len(data)} samples for {target}")

    print(f"\nTotal: {len(all_data)} samples")

    # ============================================
    # Per-benchmark, per-target breakdown
    # ============================================
    bench_target = defaultdict(lambda: defaultdict(lambda: {"labels": [], "preds": []}))
    bench_all = defaultdict(lambda: {"labels": [], "preds": [], "is_vlm": False})

    for d in all_data:
        bench = d["benchmark"]
        target = d["target"]
        if d.get("p_correct") is not None and d.get("is_correct") is not None:
            bench_target[bench][target]["labels"].append(d["is_correct"])
            bench_target[bench][target]["preds"].append(d["p_correct"])
            bench_all[bench]["labels"].append(d["is_correct"])
            bench_all[bench]["preds"].append(d["p_correct"])
            bench_all[bench]["is_vlm"] = d.get("has_image", False)

    # Print table
    print(f"\n{'='*100}")
    print(f"Per-Benchmark AUROC Breakdown")
    print(f"{'='*100}")
    print(f"{'Benchmark':<25} {'Type':>4} {'N':>6} {'AccRate':>8} {'Combined':>9} {'GPT5m':>8} {'GPT52':>8} {'Qwen35':>8}")
    print(f"{'-'*25} {'-'*4} {'-'*6} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*8}")

    breakdown = {}
    for bench in sorted(bench_all.keys()):
        bd = bench_all[bench]
        combined = compute_metrics(bd["labels"], bd["preds"])
        modality = "VLM" if bd["is_vlm"] else "TXT"

        per_target = {}
        target_strs = []
        for t in targets:
            bt = bench_target[bench][t]
            if bt["labels"]:
                m = compute_metrics(bt["labels"], bt["preds"])
                per_target[t] = m
                target_strs.append(f"{m['auroc']:.3f}" if m['auroc'] is not None else "  N/A")
            else:
                target_strs.append("    -")

        auroc_str = f"{combined['auroc']:.3f}" if combined['auroc'] is not None else "N/A"
        print(f"{bench:<25} {modality:>4} {combined['n']:>6} {combined['acc_rate']:>8.1%} "
              f"{auroc_str:>9} {'  '.join(target_strs)}")

        breakdown[bench] = {
            "modality": modality,
            "combined": combined,
            "per_target": per_target,
        }

    # Summary by modality
    print(f"\n{'='*60}")
    print(f"Summary by Modality")
    print(f"{'='*60}")

    for mod_label, is_vlm in [("Text", False), ("VLM", True)]:
        mod_labels = [d["is_correct"] for d in all_data
                      if d.get("p_correct") is not None
                      and d.get("has_image", False) == is_vlm]
        mod_preds = [d["p_correct"] for d in all_data
                     if d.get("p_correct") is not None
                     and d.get("has_image", False) == is_vlm]
        if len(set(mod_labels)) > 1:
            auroc = roc_auc_score(mod_labels, mod_preds)
            print(f"  {mod_label}: AUROC={auroc:.4f} (n={len(mod_labels)})")
        else:
            print(f"  {mod_label}: single class (n={len(mod_labels)})")

    # Summary by difficulty (accuracy rate)
    print(f"\n{'='*60}")
    print(f"Performance vs Benchmark Difficulty")
    print(f"{'='*60}")

    easy = []  # acc > 70%
    medium = []  # 30-70%
    hard = []  # < 30%

    for bench, bd in breakdown.items():
        acc = bd["combined"]["acc_rate"]
        auroc = bd["combined"]["auroc"]
        if auroc is None:
            continue
        entry = (bench, acc, auroc)
        if acc > 0.7:
            easy.append(entry)
        elif acc > 0.3:
            medium.append(entry)
        else:
            hard.append(entry)

    for difficulty, entries in [("Easy (>70% acc)", easy), ("Medium (30-70%)", medium), ("Hard (<30%)", hard)]:
        if entries:
            aurocs = [e[2] for e in entries]
            print(f"  {difficulty}: mean AUROC={np.mean(aurocs):.3f} (n_benchmarks={len(entries)})")
            for bench, acc, auroc in sorted(entries, key=lambda x: x[2]):
                print(f"    {bench:<25} acc={acc:.1%}  AUROC={auroc:.3f}")

    # Save results
    results = {
        "per_benchmark": breakdown,
        "n_total": len(all_data),
        "n_benchmarks": len(breakdown),
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
