#!/usr/bin/env python3
"""Train the UQ model with 3 different random seeds for error bars.

Uses the best config (r=32, combined prompt) and runs 3 seeds sequentially.
Each seed gets a different train/test split, and we report mean +/- std AUROC.

Usage:
    CUDA_VISIBLE_DEVICES=0,4 python scripts/multi_seed_training.py
    CUDA_VISIBLE_DEVICES=0 python scripts/multi_seed_training.py --smoke_test
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/ablations/multi_seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    script_path = os.path.join(os.path.dirname(__file__), "retrain_best_v2.py")

    all_results = []

    for seed in args.seeds:
        seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        cmd = [
            sys.executable, script_path,
            "--output_dir", seed_dir,
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--lora_r", str(args.lora_r),
        ]
        if args.smoke_test:
            cmd.append("--smoke_test")

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=False, text=True)
        elapsed = time.time() - t0

        # Load results
        results_path = os.path.join(seed_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
            results["seed"] = seed
            results["elapsed_min"] = elapsed / 60
            all_results.append(results)
            print(f"  Seed {seed}: AUROC={results.get('auroc', 'N/A')}, "
                  f"time={elapsed/60:.1f}m")
        else:
            print(f"  Seed {seed}: FAILED (no results.json)")
            all_results.append({"seed": seed, "auroc": None, "elapsed_min": elapsed / 60})

    # Summary
    print(f"\n{'='*70}")
    print(f"MULTI-SEED SUMMARY")
    print(f"{'='*70}")

    aurocs = [r["auroc"] for r in all_results if r.get("auroc") is not None]
    vlm_aurocs = [r.get("vlm_auroc") for r in all_results if r.get("vlm_auroc") is not None]
    text_aurocs = [r.get("text_auroc") for r in all_results if r.get("text_auroc") is not None]

    for r in all_results:
        auroc_str = f"{r['auroc']:.4f}" if r.get("auroc") is not None else "FAILED"
        print(f"  Seed {r['seed']}: AUROC={auroc_str}")

    if aurocs:
        print(f"\n  Overall:  {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
    if vlm_aurocs:
        print(f"  VLM:      {np.mean(vlm_aurocs):.4f} +/- {np.std(vlm_aurocs):.4f}")
    if text_aurocs:
        print(f"  Text:     {np.mean(text_aurocs):.4f} +/- {np.std(text_aurocs):.4f}")

    summary = {
        "seeds": args.seeds,
        "config": {"epochs": args.epochs, "lora_r": args.lora_r, "prompt": "combined"},
        "per_seed": all_results,
        "mean_auroc": float(np.mean(aurocs)) if aurocs else None,
        "std_auroc": float(np.std(aurocs)) if aurocs else None,
        "mean_vlm_auroc": float(np.mean(vlm_aurocs)) if vlm_aurocs else None,
        "mean_text_auroc": float(np.mean(text_aurocs)) if text_aurocs else None,
    }

    with open(os.path.join(args.output_dir, "multi_seed_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output_dir}/multi_seed_summary.json")


if __name__ == "__main__":
    main()
