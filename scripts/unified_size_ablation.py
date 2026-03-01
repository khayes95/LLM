#!/usr/bin/env python3
"""Unified VLM Model Size Ablation: Qwen3-VL-2B vs 4B vs 8B.

Trains the same unified UQ pipeline (text + VLM with real images) at different
model sizes to measure how small you can go for practical deployment.

This reuses the data loading and training logic from train_best_uq.py but
parametrizes the model name. Each model is trained sequentially in a subprocess
to avoid CUDA context corruption.

Usage:
    # Full ablation (sequential, ~3-4 hours on 4 GPUs)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/unified_size_ablation.py

    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/unified_size_ablation.py --smoke_test

    # Single model
    CUDA_VISIBLE_DEVICES=0,1 python scripts/unified_size_ablation.py --models 2b
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODELS = {
    "2b": {
        "name": "Qwen/Qwen3-VL-2B-Instruct",
        "lora_r": 16,
        "learning_rate": 1e-4,
        "grad_accum": 16,
        "batch_size": 2,  # 2B fits more per GPU
    },
    "4b": {
        "name": "Qwen/Qwen3-VL-4B-Instruct",
        "lora_r": 16,
        "learning_rate": 1e-4,
        "grad_accum": 16,
        "batch_size": 1,
    },
    "8b": {
        "name": "Qwen/Qwen3-VL-8B-Instruct",
        "lora_r": 16,
        "learning_rate": 1e-4,
        "grad_accum": 16,
        "batch_size": 1,
    },
}

OUTPUT_BASE = Path("uq_models/unified_size_ablation")


def train_single_model(size_key, config, smoke_test=False, epochs=3,
                        test_fraction=0.15):
    """Train a single model size using train_best_uq.py as a subprocess."""
    output_dir = OUTPUT_BASE / size_key
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already trained
    results_path = output_dir / "results.json"
    if results_path.exists():
        results = json.load(open(results_path))
        print(f"\n{'='*60}")
        print(f"SKIP {size_key}: Already trained (AUROC={results.get('auroc', 'N/A'):.4f})")
        print(f"{'='*60}")
        return results

    print(f"\n{'='*60}")
    print(f"TRAINING {size_key}: {config['name']}")
    print(f"{'='*60}")

    # We need to modify MODEL_NAME in train_best_uq.py dynamically.
    # Instead, we'll create a thin wrapper that patches it.
    wrapper_code = f'''
import sys
sys.path.insert(0, ".")

# Patch the model name before importing
import scripts.train_best_uq as trainer
trainer.MODEL_NAME = "{config['name']}"

# Override sys.argv
sys.argv = [
    "train_best_uq.py",
    "--output_dir", "{output_dir}",
    "--epochs", "{epochs}",
    "--batch_size", "{config['batch_size']}",
    "--grad_accum", "{config['grad_accum']}",
    "--learning_rate", "{config['learning_rate']}",
    "--lora_r", "{config['lora_r']}",
    "--test_fraction", "{test_fraction}",
]
{"sys.argv.append('--smoke_test')" if smoke_test else ""}

trainer.main()
'''

    wrapper_path = output_dir / "_run_wrapper.py"
    with open(wrapper_path, "w") as f:
        f.write(wrapper_code)

    log_path = Path("logs") / f"size_ablation_{size_key}.log"
    log_path.parent.mkdir(exist_ok=True)

    env = os.environ.copy()
    t0 = time.time()

    print(f"  Log: {log_path}")
    with open(log_path, "w") as lf:
        proc = subprocess.run(
            [sys.executable, str(wrapper_path)],
            stdout=lf, stderr=subprocess.STDOUT,
            cwd="/scratch/khayes/LLM",
            env=env,
            timeout=14400,  # 4 hour timeout per model
        )

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed/60:.1f} min (exit code: {proc.returncode})")

    if proc.returncode != 0:
        print(f"  FAILED! Check log: {log_path}")
        # Print last 20 lines of log
        with open(log_path) as f:
            lines = f.readlines()
            for line in lines[-20:]:
                print(f"    {line.rstrip()}")
        return None

    # Read results
    if results_path.exists():
        results = json.load(open(results_path))
        results["training_time_min"] = elapsed / 60
        json.dump(results, open(results_path, "w"), indent=2)
        return results
    else:
        print(f"  WARNING: No results.json found at {results_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Unified VLM Size Ablation")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--models", default="2b,4b,8b",
                        help="Comma-separated list of model sizes to train")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    args = parser.parse_args()

    sizes = [s.strip() for s in args.models.split(",")]
    for s in sizes:
        if s not in MODELS:
            print(f"ERROR: Unknown model size '{s}'. Available: {list(MODELS.keys())}")
            sys.exit(1)

    print("=" * 60)
    print("UNIFIED VLM SIZE ABLATION")
    print("=" * 60)
    print(f"Models: {sizes}")
    print(f"Epochs: {args.epochs}")
    print(f"Smoke test: {args.smoke_test}")
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
            test_fraction=args.test_fraction,
        )
        if result:
            all_results[size_key] = result

    # Summary
    print("\n" + "=" * 60)
    print("SIZE ABLATION RESULTS")
    print("=" * 60)
    print(f"{'Size':<8} {'AUROC':>8} {'VLM':>8} {'Text':>8} {'Brier':>8} {'Time':>8}")
    print("-" * 50)

    for size_key in sizes:
        if size_key not in all_results:
            print(f"{size_key:<8} {'FAILED':>8}")
            continue
        r = all_results[size_key]
        vlm = r.get("vlm_auroc", 0)
        txt = r.get("text_auroc", 0)
        t = r.get("training_time_min", 0)
        print(f"{size_key:<8} {r['auroc']:>8.4f} {vlm:>8.4f} {txt:>8.4f} "
              f"{r['brier']:>8.4f} {t:>7.1f}m")

    # Save summary
    summary_path = OUTPUT_BASE / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Comparison with text-only ablation
    print("\n--- Comparison ---")
    print("Unified VLM (this ablation):")
    for s in sizes:
        if s in all_results:
            print(f"  Qwen3-VL-{s.upper()}: {all_results[s]['auroc']:.4f}")
    print("Legacy text-only (Qwen2.5):")
    print("  0.5B: 0.758, 1.5B: 0.772, 3B: 0.767, 7B: 0.785")


if __name__ == "__main__":
    main()
