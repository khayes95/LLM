#!/usr/bin/env python3
"""Phase 1 Smoke Test: Verify InternVL3-78B works with VSR.

This script:
1. Loads InternVL3-78B (bf16, device_map="auto")
2. Runs on 10 VSR samples
3. Extracts hidden states, verifies shape is [8192]
4. Saves one sample's features to disk, verifies it loads back

Success criteria:
- [ ] Model loads without OOM
- [ ] Can process image + text input
- [ ] Can extract hidden states (shape: [8192])
- [ ] Can generate text response
- [ ] Can save/load features to disk

Usage:
    python scripts/smoke_test.py
"""
import argparse
import sys
import time
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.internvl_client import load_internvl_client
from src.benchmark_loaders import load_vsr, score_response
from src.feature_extraction import UQSample, save_sample, load_sample


def run_smoke_test(num_samples: int = 10):
    """Run smoke test on InternVL3-78B with VSR samples."""
    print("=" * 60)
    print("PHASE 1: SMOKE TEST - InternVL3-78B")
    print("=" * 60)

    # Track success criteria
    success = {
        "model_loads": False,
        "process_input": False,
        "extract_hidden": False,
        "generate_text": False,
        "save_load_features": False,
    }

    # 1. Load model
    print("\n[1/5] Loading InternVL3-78B...")
    start = time.time()
    try:
        client = load_internvl_client(model_size="78B")
        success["model_loads"] = True
        print(f"  ✓ Model loaded in {time.time() - start:.1f}s")
        print(f"  ✓ Device: {client.device}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return success

    # 2. Load VSR samples
    print(f"\n[2/5] Loading {num_samples} VSR samples...")
    try:
        samples = list(load_vsr(split="test", max_samples=num_samples))
        print(f"  ✓ Loaded {len(samples)} samples")
    except Exception as e:
        print(f"  ✗ Failed to load VSR: {e}")
        return success

    # 3. Process first sample
    print("\n[3/5] Testing input processing...")
    sample = samples[0]
    try:
        pixel_values = client.preprocess_image(sample.image)
        print(f"  ✓ Pixel values shape: {pixel_values.shape}")
        success["process_input"] = True
    except Exception as e:
        print(f"  ✗ Failed to process input: {e}")
        return success

    # 4. Extract features
    print("\n[4/5] Testing feature extraction...")
    try:
        features = client.extract_features(pixel_values)
        print(f"  ✓ Hidden state shape: {features.hidden_state.shape}")

        # Verify expected shape (8192 for InternVL3-78B)
        if features.hidden_state.shape[0] == 8192:
            print("  ✓ Hidden state dimension is 8192 as expected")
            success["extract_hidden"] = True
        else:
            print(f"  ✗ Unexpected dimension: {features.hidden_state.shape[0]}")
    except Exception as e:
        print(f"  ✗ Failed to extract features: {e}")
        return success

    # 5. Generate response
    print("\n[5/5] Testing generation...")
    try:
        response = client.generate(sample.image, sample.prompt)
        is_correct = score_response(response, sample.ground_truth, sample.task_type)

        print(f"  ✓ Generated response: {response[:100]}...")
        print(f"  ✓ Ground truth: {sample.ground_truth}")
        print(f"  ✓ Is correct: {is_correct}")
        success["generate_text"] = True
    except Exception as e:
        print(f"  ✗ Failed to generate: {e}")
        return success

    # 6. Save and load features
    print("\n[6/6] Testing save/load...")
    try:
        uq_sample = UQSample(
            benchmark="vsr",
            question_id=sample.id,
            prompt=sample.prompt,
            response=response,
            ground_truth=sample.ground_truth,
            is_correct=is_correct,
            hidden_state=features.hidden_state,
            top_prob=features.top_prob,
            entropy=features.entropy,
        )

        output_dir = Path("data/features/smoke_test")
        filepath = save_sample(uq_sample, output_dir)
        print(f"  ✓ Saved to: {filepath}")

        loaded = load_sample(filepath)
        print(f"  ✓ Loaded sample: {loaded.question_id}")
        print(f"  ✓ Hidden state matches: {torch.allclose(uq_sample.hidden_state, loaded.hidden_state)}")
        success["save_load_features"] = True
    except Exception as e:
        print(f"  ✗ Failed to save/load: {e}")

    # Run on all samples
    print("\n" + "=" * 60)
    print(f"Running inference on {len(samples)} samples...")
    print("=" * 60)

    correct = 0
    total = 0

    for i, sample in enumerate(samples):
        try:
            response, features = client.generate_with_features(sample.image, sample.prompt)
            is_correct = score_response(response, sample.ground_truth, sample.task_type)

            if is_correct:
                correct += 1
            total += 1

            status = "✓" if is_correct else "✗"
            print(f"  [{i+1}/{len(samples)}] {status} GT={sample.ground_truth}, Pred={response[:50]}...")

            # Clear cache periodically
            if i % 5 == 0:
                client.clear_cache()

        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] ERROR: {e}")

    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)

    for criterion, passed in success.items():
        status = "✓" if passed else "✗"
        print(f"  [{status}] {criterion}")

    print(f"\n  Accuracy: {correct}/{total} ({100*correct/total:.1f}%)" if total > 0 else "")
    print(f"  All criteria passed: {all(success.values())}")

    return success


def main():
    parser = argparse.ArgumentParser(description="InternVL3-78B smoke test")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of VSR samples to test")
    args = parser.parse_args()

    success = run_smoke_test(num_samples=args.num_samples)

    # Exit with error if any criteria failed
    sys.exit(0 if all(success.values()) else 1)


if __name__ == "__main__":
    main()
