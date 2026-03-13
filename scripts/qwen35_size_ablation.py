#!/usr/bin/env python3
"""Qwen3.5 Model Size Ablation for UQ Calibrator.

Trains the unified UQ pipeline at different Qwen3.5 model sizes (0.8B, 2B, 4B, 9B)
to test whether a newer model family works as a calibrator backbone.

Uses --base_model and --split_info flags of train_best_uq.py to:
1. Load the correct Qwen3.5 model (different architecture from Qwen3-VL)
2. Reuse the exact v2 train/test split (NO data leakage)

Each model is trained sequentially in a subprocess to avoid CUDA context corruption.

Usage:
    # Smoke test all sizes
    CUDA_VISIBLE_DEVICES=1,2,3,4 python scripts/qwen35_size_ablation.py --smoke_test

    # Full ablation (sequential, ~6-7 hours)
    CUDA_VISIBLE_DEVICES=1,2,3,4 python scripts/qwen35_size_ablation.py

    # Single model
    CUDA_VISIBLE_DEVICES=1,2 python scripts/qwen35_size_ablation.py --models 9b
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# Qwen3.5 models are natively multimodal (no separate -VL variant)
MODELS = {
    "0.8b": {
        "name": "Qwen/Qwen3.5-0.8B",
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "grad_accum": 32,
        "batch_size": 1,
        "gpus": 1,
    },
    "2b": {
        "name": "Qwen/Qwen3.5-2B",
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 1e-4,
        "grad_accum": 16,
        "batch_size": 1,
        "gpus": 1,
    },
    "4b": {
        "name": "Qwen/Qwen3.5-4B",
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 5e-5,
        "grad_accum": 16,
        "batch_size": 1,
        "gpus": 1,
    },
    "9b": {
        "name": "Qwen/Qwen3.5-9B",
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 3e-5,
        "grad_accum": 16,
        "batch_size": 1,
        "gpus": 2,
    },
}

OUTPUT_BASE = Path("data/ablations/qwen35_model_size")
SPLIT_INFO = "uq_models/best_v3_qsplit/split_info.json"


def train_single_model(size_key, config, smoke_test=False, epochs=3):
    """Train a single model size using train_best_uq.py as a subprocess."""
    output_dir = OUTPUT_BASE / size_key
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already trained
    results_path = output_dir / "results.json"
    if results_path.exists() and not smoke_test:
        results = json.load(open(results_path))
        print(f"\n{'='*60}")
        print(f"SKIP {size_key}: Already trained (AUROC={results.get('auroc', 'N/A')})")
        print(f"{'='*60}")
        return results

    print(f"\n{'='*60}")
    print(f"TRAINING {size_key}: {config['name']}")
    print(f"  LoRA r={config['lora_r']}, alpha={config['lora_alpha']}")
    print(f"  LR={config['learning_rate']}, batch={config['batch_size']}, grad_accum={config['grad_accum']}")
    print(f"  Split info: {SPLIT_INFO}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "scripts/train_best_uq.py",
        "--output_dir", str(output_dir),
        "--base_model", config["name"],
        "--split_info", SPLIT_INFO,
        "--epochs", str(epochs),
        "--batch_size", str(config["batch_size"]),
        "--grad_accum", str(config["grad_accum"]),
        "--learning_rate", str(config["learning_rate"]),
        "--lora_r", str(config["lora_r"]),
        "--lora_alpha", str(config["lora_alpha"]),
        "--prompt_variant", "combined",
    ]
    if smoke_test:
        cmd.append("--smoke_test")

    log_path = Path("logs") / f"qwen35_ablation_{size_key}.log"
    log_path.parent.mkdir(exist_ok=True)

    env = os.environ.copy()
    # Restrict GPUs per model size to avoid NCCL hangs with device_map="auto"
    available_gpus = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    gpu_count = config.get("gpus", 1)
    subset_gpus = ",".join(available_gpus[:gpu_count])
    env["CUDA_VISIBLE_DEVICES"] = subset_gpus

    t0 = time.time()

    print(f"  GPUs: {subset_gpus} ({gpu_count} device(s))")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_path}")

    with open(log_path, "w") as lf:
        proc = subprocess.run(
            cmd,
            stdout=lf, stderr=subprocess.STDOUT,
            cwd="/scratch/khayes/LLM",
            env=env,
            timeout=28800,  # 8 hour timeout per model
        )

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed/60:.1f} min (exit code: {proc.returncode})")

    if proc.returncode != 0:
        print(f"  FAILED! Check log: {log_path}")
        with open(log_path) as f:
            lines = f.readlines()
            for line in lines[-30:]:
                print(f"    {line.rstrip()}")
        return None

    # Read results
    if results_path.exists():
        results = json.load(open(results_path))
        results["training_time_min"] = elapsed / 60
        results["model_name"] = config["name"]
        results["size_key"] = size_key
        json.dump(results, open(results_path, "w"), indent=2)
        return results
    else:
        print(f"  WARNING: No results.json found at {results_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 Model Size Ablation")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--models", default="0.8b,2b,4b,9b",
                        help="Comma-separated list of model sizes to train")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    sizes = [s.strip() for s in args.models.split(",")]
    for s in sizes:
        if s not in MODELS:
            print(f"ERROR: Unknown model size '{s}'. Available: {list(MODELS.keys())}")
            sys.exit(1)

    print("=" * 60)
    print("QWEN3.5 MODEL SIZE ABLATION")
    print("=" * 60)
    print(f"Models: {sizes}")
    print(f"Epochs: {args.epochs}")
    print(f"Smoke test: {args.smoke_test}")
    print(f"Split info: {SPLIT_INFO}")
    print(f"Output: {OUTPUT_BASE}")
    print()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for size_key in sizes:
        config = MODELS[size_key]
        result = train_single_model(
            size_key, config,
            smoke_test=args.smoke_test,
            epochs=args.epochs,
        )
        if result:
            all_results[size_key] = result

    # Summary
    print("\n" + "=" * 60)
    print("QWEN3.5 SIZE ABLATION RESULTS")
    print("=" * 60)
    print(f"{'Size':<8} {'Model':<25} {'AUROC':>8} {'VLM':>8} {'Text':>8} {'Brier':>8} {'Time':>8}")
    print("-" * 80)

    for size_key in sizes:
        if size_key not in all_results:
            print(f"{size_key:<8} {'FAILED':>25}")
            continue
        r = all_results[size_key]
        vlm = r.get("vlm_auroc", 0)
        txt = r.get("text_auroc", 0)
        t = r.get("training_time_min", 0)
        name = MODELS[size_key]["name"].split("/")[-1]
        print(f"{size_key:<8} {name:<25} {r['auroc']:>8.4f} {vlm:>8.4f} {txt:>8.4f} "
              f"{r['brier']:>8.4f} {t:>7.1f}m")

    # Save summary
    summary_path = OUTPUT_BASE / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Comparison with previous Qwen3-VL ablation
    print("\n--- Comparison with Qwen3-VL (previous) ---")
    print("Qwen3-VL: 2B=0.816, 4B=0.830, 8B=0.827 (v1 config)")
    print("Qwen3.5 (this ablation):")
    for s in sizes:
        if s in all_results:
            print(f"  Qwen3.5-{s.upper()}: {all_results[s]['auroc']:.4f}")


if __name__ == "__main__":
    main()
