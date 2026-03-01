#!/usr/bin/env python3
"""Run full inference on a benchmark and save features for UQ training.

Usage:
    python scripts/run_inference.py --benchmark vsr --output_dir data/features/vsr
    python scripts/run_inference.py --benchmark hallusionbench --max_samples 500
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molmo_client import load_molmo_client
from src.benchmark_loaders import load_benchmark, score_response
from src.feature_extraction import UQSample, save_sample, CheckpointManager


def run_inference(
    benchmark: str,
    output_dir: str,
    model_size: str = "72B",
    max_samples: int | None = None,
    resume: bool = True,
):
    """Run inference on a benchmark and save features.

    Args:
        benchmark: Benchmark name ("vsr", "hallusionbench", "mmmu_pro")
        output_dir: Directory to save features
        model_size: Molmo model size ("72B" or "7B")
        max_samples: Optional limit on samples
        resume: Whether to resume from checkpoint
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"INFERENCE: {benchmark.upper()}")
    print("=" * 60)

    # Load model
    print(f"\nLoading Molmo-{model_size}...")
    start = time.time()
    client = load_molmo_client(model_size=model_size)
    print(f"Model loaded in {time.time() - start:.1f}s")

    # Load benchmark
    print(f"\nLoading {benchmark} benchmark...")
    samples = list(load_benchmark(benchmark, max_samples=max_samples))
    print(f"Loaded {len(samples)} samples")

    # Setup checkpointing
    checkpoint_path = output_dir / f"{benchmark}_checkpoint.json"
    checkpoint = CheckpointManager(checkpoint_path) if resume else None

    # Track stats
    stats = {"total": 0, "correct": 0, "errors": 0, "skipped": 0}

    # Run inference
    print("\nRunning inference...")
    pbar = tqdm(samples, desc="Processing")

    for sample in pbar:
        sample_id = sample.id

        # Skip if already processed
        if checkpoint and checkpoint.is_completed(sample_id):
            stats["skipped"] += 1
            continue

        try:
            # Generate response with features
            response, features = client.generate_with_features(sample.image, sample.prompt)

            # Score response
            is_correct = score_response(response, sample.ground_truth, sample.task_type)

            # Create UQ sample
            uq_sample = UQSample(
                benchmark=benchmark,
                question_id=sample_id,
                prompt=sample.prompt,
                response=response,
                ground_truth=sample.ground_truth,
                is_correct=is_correct,
                hidden_state=features.hidden_state,
                top_prob=features.top_prob,
                entropy=features.entropy,
            )

            # Save sample
            save_sample(uq_sample, output_dir)

            # Update stats
            stats["total"] += 1
            if is_correct:
                stats["correct"] += 1

            # Update checkpoint
            if checkpoint:
                checkpoint.mark_completed(sample_id)

            # Update progress bar
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            pbar.set_postfix({"acc": f"{acc:.2%}", "err": stats["errors"]})

            # Clear cache periodically
            if stats["total"] % 10 == 0:
                client.clear_cache()

        except Exception as e:
            stats["errors"] += 1
            tqdm.write(f"Error on {sample_id}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)
    print(f"  Benchmark: {benchmark}")
    print(f"  Total processed: {stats['total']}")
    print(f"  Skipped (resumed): {stats['skipped']}")
    print(f"  Errors: {stats['errors']}")
    if stats["total"] > 0:
        print(f"  Correct: {stats['correct']} / {stats['total']} ({100*stats['correct']/stats['total']:.1f}%)")
    print(f"  Features saved to: {output_dir}")

    # Save final stats
    import json
    stats_path = output_dir / f"{benchmark}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to: {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Run inference on vision benchmark")
    parser.add_argument("--benchmark", required=True,
                        choices=["vsr", "hallusionbench", "mmmu_pro"],
                        help="Benchmark to run")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: data/features/<benchmark>)")
    parser.add_argument("--model_size", default="72B", choices=["72B", "7B"],
                        help="Molmo model size")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process")
    parser.add_argument("--no_resume", action="store_true",
                        help="Don't resume from checkpoint")
    args = parser.parse_args()

    output_dir = args.output_dir or f"data/features/{args.benchmark}"

    run_inference(
        benchmark=args.benchmark,
        output_dir=output_dir,
        model_size=args.model_size,
        max_samples=args.max_samples,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
