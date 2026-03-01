#!/usr/bin/env python3
"""Run inference on all vision benchmarks for UQ training data collection.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_all_benchmarks.py
"""
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.internvl_client import load_internvl_client
from src.benchmark_loaders import load_benchmark, score_response, BENCHMARK_LOADERS
from src.feature_extraction import UQSample, save_sample, CheckpointManager


# Benchmark configs: (name, max_samples)
BENCHMARK_CONFIGS = [
    ("vsr", 2000),           # True/False spatial reasoning
    ("mmmu", 2000),          # Multiple choice academic
    ("charxiv", 1000),       # Scientific figure reasoning
    ("hallusionbench", None),  # All ~1100 (Yes/No hallucination)
    ("erqa", None),          # All ~400 (Embodied reasoning MC)
]


def run_benchmark(
    client,
    benchmark: str,
    output_dir: Path,
    max_samples: int | None = None,
    resume: bool = True,
) -> dict:
    """Run inference on a single benchmark.

    Args:
        client: InternVL client (already loaded)
        benchmark: Benchmark name
        output_dir: Directory to save features
        max_samples: Optional limit on samples
        resume: Whether to resume from checkpoint

    Returns:
        Dict with stats
    """
    bench_dir = output_dir / benchmark
    bench_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"BENCHMARK: {benchmark.upper()}")
    print(f"{'='*60}")

    # Load benchmark samples
    print(f"Loading {benchmark}...")
    start = time.time()
    samples = list(load_benchmark(benchmark, max_samples=max_samples))
    print(f"Loaded {len(samples)} samples in {time.time() - start:.1f}s")

    if len(samples) == 0:
        print(f"WARNING: No samples loaded for {benchmark}")
        return {"total": 0, "correct": 0, "errors": 0, "skipped": 0}

    # Setup checkpointing
    checkpoint_path = bench_dir / "checkpoint.json"
    checkpoint = CheckpointManager(checkpoint_path) if resume else None

    # Track stats
    stats = {"total": 0, "correct": 0, "errors": 0, "skipped": 0}

    # Run inference
    pbar = tqdm(samples, desc=benchmark)

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
            save_sample(uq_sample, bench_dir)

            # Update stats
            stats["total"] += 1
            if is_correct:
                stats["correct"] += 1

            # Update checkpoint
            if checkpoint:
                checkpoint.mark_completed(sample_id)

            # Update progress bar
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            pbar.set_postfix({"acc": f"{acc:.1%}", "err": stats["errors"]})

            # Clear cache periodically
            if stats["total"] % 20 == 0:
                client.clear_cache()

        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 5:
                tqdm.write(f"Error on {sample_id}: {e}")

    # Summary for this benchmark
    print(f"\n{benchmark} complete:")
    print(f"  Processed: {stats['total']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")
    if stats["total"] > 0:
        print(f"  Accuracy: {stats['correct']}/{stats['total']} ({100*stats['correct']/stats['total']:.1f}%)")

    # Save stats
    stats_path = bench_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MULTIMODAL UQ DATA COLLECTION")
    print("=" * 60)
    print(f"\nBenchmarks to run:")
    for name, max_samples in BENCHMARK_CONFIGS:
        limit = max_samples if max_samples else "all"
        print(f"  - {name}: {limit} samples")

    # Load model once
    print(f"\nLoading InternVL3-78B...")
    start = time.time()
    client = load_internvl_client()
    print(f"Model loaded in {time.time() - start:.1f}s")

    # Run all benchmarks
    all_stats = {}
    total_start = time.time()

    for benchmark, max_samples in BENCHMARK_CONFIGS:
        stats = run_benchmark(
            client=client,
            benchmark=benchmark,
            output_dir=output_dir,
            max_samples=max_samples,
            resume=True,
        )
        all_stats[benchmark] = stats

    # Final summary
    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    total_samples = sum(s["total"] for s in all_stats.values())
    total_correct = sum(s["correct"] for s in all_stats.values())
    total_errors = sum(s["errors"] for s in all_stats.values())

    print(f"\nPer-benchmark results:")
    for bench, stats in all_stats.items():
        if stats["total"] > 0:
            acc = 100 * stats["correct"] / stats["total"]
            print(f"  {bench:20s}: {stats['total']:5d} samples, {acc:5.1f}% accuracy")

    print(f"\nOverall:")
    print(f"  Total samples: {total_samples}")
    print(f"  Total correct: {total_correct} ({100*total_correct/total_samples:.1f}%)" if total_samples > 0 else "")
    print(f"  Total errors: {total_errors}")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"  Features saved to: {output_dir}/")

    # Save overall stats
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "benchmarks": all_stats,
            "total_samples": total_samples,
            "total_correct": total_correct,
            "total_errors": total_errors,
            "total_time_seconds": total_time,
        }, f, indent=2)
    print(f"  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
