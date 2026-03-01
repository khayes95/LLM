#!/usr/bin/env python3
"""
Full evaluation of InternVL3.5-38B on VLM benchmarks.

Uses GPT-5-mini combined sample IDs for fair cross-model comparison.
Uses the InternVLClient from uq_eval for proper image handling.

Usage:
    CUDA_VISIBLE_DEVICES=4,5 python scripts/run_internvl3_full.py
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# VLM benchmarks matching GPT-5-mini
VLM_BENCHMARKS = [
    "charxiv", "erqa", "mathverse", "mathvision", "mathvista",
    "mmstar", "realworldqa", "vizwiz", "hle_multimodal", "mmmu"
]


def load_gpt5_sample_ids(benchmark: str) -> list[str] | None:
    """Load sample IDs from combined GPT-5-mini dataset."""
    combined_dir = Path("/scratch/khayes/LLM/runs/gpt5_mini_combined")
    ids_file = combined_dir / benchmark / "sampled_ids.json"
    if ids_file.exists():
        with open(ids_file) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Full eval of InternVL3.5-38B on VLM benchmarks")
    parser.add_argument(
        "--model",
        default="OpenGVLab/InternVL3_5-38B-Flash",
        help="Model name (Flash version is 4x faster)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=250,
        help="Max samples per benchmark (default: 250 to match GPT-5-mini)"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Override benchmark list"
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="Benchmarks to skip"
    )
    args = parser.parse_args()

    import torch
    from uq_eval.registry import load_benchmark
    from uq_eval.types import ModelResponse, ModelRequest
    from uq_eval.models.internvl_client import InternVLClient

    # Build benchmark list
    benchmarks = args.benchmarks if args.benchmarks else VLM_BENCHMARKS
    benchmarks = [b for b in benchmarks if b not in args.skip]

    print("=" * 60)
    print("InternVL3.5 Full Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Benchmarks ({len(benchmarks)}): {benchmarks}")
    print(f"Max samples per benchmark: {args.max_samples}")
    print(f"GPUs available: {torch.cuda.device_count()}")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1].replace("-", "_")
    out_base = Path(f"/scratch/khayes/LLM/runs/{timestamp}_full_{model_short}")
    out_base.mkdir(parents=True, exist_ok=True)

    # Load model using InternVLClient
    print("Loading InternVL3.5 model...")
    start_load = time.time()
    client = InternVLClient(model_name=args.model)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.1f}s")
    print()

    results = {}
    total_samples = 0
    total_time = 0

    for bench_idx, bench_name in enumerate(benchmarks):
        print("=" * 60)
        print(f"[{bench_idx+1}/{len(benchmarks)}] Testing: {bench_name}")
        print("=" * 60)

        bench_out = out_base / bench_name
        bench_out.mkdir(exist_ok=True)

        # Load benchmark
        try:
            benchmark = load_benchmark(bench_name)
        except Exception as e:
            print(f"Error loading {bench_name}: {e}")
            results[bench_name] = {"error": str(e)}
            continue

        # Get examples
        all_examples = list(benchmark.iter_examples("test"))
        print(f"  Available: {len(all_examples)} examples")

        if len(all_examples) == 0:
            print(f"  Skipping {bench_name}: no examples available")
            results[bench_name] = {"error": "No examples available"}
            continue

        # Try to match GPT-5 sample IDs for fair cross-model comparison
        gpt5_ids = load_gpt5_sample_ids(bench_name)
        if gpt5_ids:
            id_set = set(gpt5_ids)
            matched_examples = [ex for ex in all_examples if ex.id in id_set]
            if len(matched_examples) >= len(gpt5_ids) * 0.8:
                examples = matched_examples[:args.max_samples]
                print(f"  Matched {len(matched_examples)}/{len(gpt5_ids)} GPT-5 IDs, using {len(examples)}")
            else:
                # Fall back to first N samples
                examples = all_examples[:args.max_samples]
                print(f"  ID mismatch ({len(matched_examples)}/{len(gpt5_ids)}), using first {len(examples)}")
        else:
            examples = all_examples[:args.max_samples]
            print(f"  No GPT-5 IDs, using first {len(examples)}")

        if len(examples) == 0:
            print(f"  Skipping {bench_name}: no examples after filtering")
            results[bench_name] = {"error": "No examples after filtering"}
            continue

        print(f"  Running with {len(examples)} samples")

        # Save sample IDs
        sample_ids = [ex.id for ex in examples]
        with open(bench_out / "sampled_ids.json", "w") as f:
            json.dump(sample_ids, f)

        # Run inference
        predictions = []
        correct_count = 0
        start_infer = time.time()

        for i, ex in enumerate(examples):
            if i % 25 == 0:
                elapsed = time.time() - start_infer
                rate = elapsed / (i + 1) if i > 0 else 0
                remaining = rate * (len(examples) - i)
                print(f"  Processing {i+1}/{len(examples)}... ({remaining/60:.1f}m remaining)")

            try:
                # Build request using benchmark API
                req = benchmark.build_request(ex)

                # Create ModelRequest for InternVLClient
                model_req = ModelRequest(
                    messages=req.messages,
                    max_output_tokens=2048,
                    temperature=0.0,
                )

                # Generate response using InternVLClient
                resp = client.generate(model_req)
                response_text = resp.text

                # Score using benchmark API
                pred = benchmark.parse_prediction(ex, resp)
                score_dict = benchmark.score(ex, pred)
                score = score_dict.get("correct", 0)
                prediction = pred.answer

            except Exception as e:
                response_text = str(e)[:200]
                prediction = ""
                score = -1

            if score == 1:
                correct_count += 1

            predictions.append({
                "id": ex.id,
                "target": ex.target,
                "response_text": response_text[:1000],
                "prediction": prediction,
                "score": score,
            })

        infer_time = time.time() - start_infer
        total_time += infer_time
        total_samples += len(examples)

        # Save predictions
        with open(bench_out / "predictions.jsonl", "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")

        # Calculate metrics
        scored = [p for p in predictions if p["score"] >= 0]
        accuracy = correct_count / len(scored) if scored else 0

        # Load GPT-5 accuracy for comparison
        gpt5_metrics_file = Path(f"/scratch/khayes/LLM/runs/gpt5_mini_combined/{bench_name}/metrics.json")
        gpt5_acc = None
        if gpt5_metrics_file.exists():
            with open(gpt5_metrics_file) as f:
                gpt5_acc = json.load(f).get("accuracy", None)

        metrics = {
            "n": len(predictions),
            "n_scored": len(scored),
            "accuracy": accuracy,
            "gpt5_accuracy": gpt5_acc,
            "accuracy_diff": accuracy - gpt5_acc if gpt5_acc else None,
            "inference_time_s": infer_time,
        }

        with open(bench_out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  InternVL: {accuracy:.1%} ({correct_count}/{len(scored)})")
        if gpt5_acc:
            print(f"  GPT-5:    {gpt5_acc:.1%}")
            print(f"  Diff:     {accuracy - gpt5_acc:+.1%}")
        if len(examples) > 0:
            print(f"  Time:     {infer_time:.1f}s ({infer_time/len(examples):.2f}s/sample)")
        print()

        results[bench_name] = metrics

    # Summary
    print("=" * 60)
    print("FULL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Total samples: {total_samples}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print()
    print(f"{'Benchmark':<15} {'InternVL':>10} {'GPT-5':>10} {'Diff':>10}")
    print("-" * 55)

    for bench, m in results.items():
        if "error" in m:
            print(f"{bench:<15} {'ERROR':>10}")
        else:
            intern_acc = f"{m['accuracy']:.1%}"
            gpt5_acc = f"{m['gpt5_accuracy']:.1%}" if m.get('gpt5_accuracy') else "N/A"
            diff = f"{m['accuracy_diff']:+.1%}" if m.get('accuracy_diff') is not None else "N/A"
            print(f"{bench:<15} {intern_acc:>10} {gpt5_acc:>10} {diff:>10}")

    # Save summary
    with open(out_base / "summary.json", "w") as f:
        json.dump({
            "model": args.model,
            "model_load_time_s": load_time,
            "total_samples": total_samples,
            "total_inference_time_s": total_time,
            "benchmarks": results,
        }, f, indent=2)

    print()
    print(f"Results saved to: {out_base}")


if __name__ == "__main__":
    main()
