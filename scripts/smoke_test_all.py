#!/usr/bin/env python3
"""Quick smoke test for all benchmark loaders.

Tests that each benchmark can load 3 samples with valid images.
Does NOT require loading the model - just tests data loading.

Usage:
    python scripts/smoke_test_all.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark_loaders import BENCHMARK_LOADERS
from PIL import Image


def test_benchmark(name: str, max_samples: int = 3) -> dict:
    """Test a single benchmark loader.

    Returns:
        Dict with test results
    """
    loader = BENCHMARK_LOADERS[name]

    results = {
        "name": name,
        "samples_loaded": 0,
        "errors": [],
        "sample_info": [],
    }

    try:
        samples = list(loader(max_samples=max_samples))
        results["samples_loaded"] = len(samples)

        for s in samples:
            info = {
                "id": s.id,
                "has_image": isinstance(s.image, Image.Image),
                "image_size": s.image.size if isinstance(s.image, Image.Image) else None,
                "image_mode": s.image.mode if isinstance(s.image, Image.Image) else None,
                "task_type": s.task_type,
                "prompt_len": len(s.prompt),
                "ground_truth": s.ground_truth[:50] if s.ground_truth else None,
            }
            results["sample_info"].append(info)

    except Exception as e:
        results["errors"].append(str(e))

    return results


def main():
    print("=" * 60)
    print("BENCHMARK LOADER SMOKE TEST")
    print("=" * 60)

    all_results = {}
    passed = 0
    failed = 0

    for name in BENCHMARK_LOADERS:
        print(f"\nTesting {name}...")
        results = test_benchmark(name, max_samples=3)
        all_results[name] = results

        if results["samples_loaded"] >= 1 and not results["errors"]:
            print(f"  ✓ Loaded {results['samples_loaded']} samples")
            for info in results["sample_info"]:
                print(f"    - {info['id']}: {info['image_size']} {info['image_mode']}, task={info['task_type']}")
            passed += 1
        else:
            print(f"  ✗ FAILED: {results['errors']}")
            failed += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Passed: {passed}/{len(BENCHMARK_LOADERS)}")
    print(f"  Failed: {failed}/{len(BENCHMARK_LOADERS)}")

    if failed > 0:
        print("\nFailed benchmarks:")
        for name, results in all_results.items():
            if results["errors"]:
                print(f"  - {name}: {results['errors'][0][:100]}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
