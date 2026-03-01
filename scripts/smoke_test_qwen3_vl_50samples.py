#!/usr/bin/env python3
"""
Smoke test Qwen3-VL-30B on VLM + Text benchmarks.

Uses GPT-5-mini combined sample IDs for fair comparison.
Tests formatting and scoring before full 235B run.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test_qwen3_vl_50samples.py
    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test_qwen3_vl_50samples.py --max_samples 20
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Benchmarks with 250 samples (primary targets for cross-model comparison)
VLM_BENCHMARKS_250 = [
    "charxiv", "erqa", "mathverse", "mathvision", "mathvista",
    "mmstar", "realworldqa", "vizwiz"
]

# VLM benchmarks with partial samples (include but note smaller N)
VLM_BENCHMARKS_PARTIAL = ["hle_multimodal", "mmmu"]

# Text benchmarks with 200+ samples
TEXT_BENCHMARKS = [
    "bbeh", "chembench", "gpqa", "livebench", "omnimath", "simpleqa", "hle"
]

# Skip: too high accuracy (>85%) - not useful for UQ training
# aokvqa (92%), vsr (87%), babilong (93%)
# Skip: need LLM judge (0% accuracy = unscored)
# healthbench, tutorbench, prbench, hallusionbench, mmvet


def load_gpt5_sample_ids(benchmark: str) -> list[str] | None:
    """Load sample IDs from combined GPT-5-mini dataset."""
    combined_dir = Path("/scratch/khayes/LLM/runs/gpt5_mini_combined")
    ids_file = combined_dir / benchmark / "sampled_ids.json"
    if ids_file.exists():
        with open(ids_file) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Smoke test Qwen3-VL on benchmarks")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Model name (default: 30B for smoke test)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs (default: 1 for 30B)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=20,
        help="Max samples per benchmark (default: 20 for quick validation)"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Override benchmark list (default: all VLM + Text benchmarks)"
    )
    parser.add_argument(
        "--vlm_only",
        action="store_true",
        help="Only run VLM benchmarks"
    )
    parser.add_argument(
        "--text_only",
        action="store_true",
        help="Only run text benchmarks"
    )
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from uq_eval.registry import load_benchmark
    from uq_eval.models.qwen3_vl_vllm_client import decode_data_url

    # Build benchmark list
    if args.benchmarks:
        benchmarks = args.benchmarks
    elif args.vlm_only:
        benchmarks = VLM_BENCHMARKS_250 + VLM_BENCHMARKS_PARTIAL
    elif args.text_only:
        benchmarks = TEXT_BENCHMARKS
    else:
        # All benchmarks
        benchmarks = VLM_BENCHMARKS_250 + VLM_BENCHMARKS_PARTIAL + TEXT_BENCHMARKS

    print("=" * 60)
    print("Qwen3-VL Smoke Test")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Benchmarks ({len(benchmarks)}): {benchmarks}")
    print(f"Max samples per benchmark: {args.max_samples}")
    print(f"Total samples: ~{len(benchmarks) * args.max_samples}")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1].replace("-", "_")
    out_base = Path(f"/scratch/khayes/LLM/runs/{timestamp}_smoke{args.max_samples}_{model_short}")
    out_base.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading Qwen3-VL model...")
    start_load = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.9,
        max_model_len=32768,
        trust_remote_code=True,
        max_num_seqs=16,
    )
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.1f}s")
    print()

    sampling_params = SamplingParams(max_tokens=2048, temperature=0.0)

    results = {}

    for bench_name in benchmarks:
        print("=" * 60)
        print(f"Testing: {bench_name}")
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

        # Get examples - iter_examples returns an iterable
        all_examples = list(benchmark.iter_examples("test"))

        # Try to filter to GPT-5 IDs for fair comparison
        gpt5_ids = load_gpt5_sample_ids(bench_name)
        if gpt5_ids:
            id_set = set(gpt5_ids)
            matched_examples = [ex for ex in all_examples if ex.id in id_set]
            if len(matched_examples) >= len(gpt5_ids) * 0.5:  # 50% match for smoke test
                examples = matched_examples[:args.max_samples]
                print(f"  Matched {len(matched_examples)} GPT-5 IDs, using {len(examples)} for test")
            else:
                # IDs don't match - use deterministic first N
                print(f"  ID mismatch ({len(matched_examples)}/{len(gpt5_ids)}), using first {args.max_samples}")
                examples = all_examples[:args.max_samples]
        else:
            examples = all_examples[:args.max_samples]
            print(f"  No GPT-5 IDs found, using first {len(examples)} samples")

        print(f"  Testing with {len(examples)} samples")

        # Save sample IDs
        sample_ids = [ex.id for ex in examples]
        with open(bench_out / "sampled_ids.json", "w") as f:
            json.dump(sample_ids, f)

        # Build prompts using vLLM chat format with inline images
        conversations = []
        prompt_to_example = {}

        for ex in examples:
            req = benchmark.build_request(ex)
            chat_messages = []

            for msg in req.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if isinstance(content, str):
                    chat_messages.append({
                        "role": role,
                        "content": content
                    })
                elif isinstance(content, list):
                    chat_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") in ("text", "input_text"):
                                chat_content.append({
                                    "type": "text",
                                    "text": item.get("text", "")
                                })
                            elif item.get("type") in ("input_image", "image_url"):
                                # Keep data URL for vLLM chat API
                                if item.get("type") == "input_image":
                                    img_url = item.get("image_url", "")
                                else:
                                    url_data = item.get("image_url", {})
                                    img_url = url_data.get("url", "") if isinstance(url_data, dict) else url_data
                                if img_url:
                                    chat_content.append({
                                        "type": "image_url",
                                        "image_url": {"url": img_url}
                                    })
                        elif isinstance(item, str):
                            chat_content.append({"type": "text", "text": item})

                    if chat_content:
                        chat_messages.append({
                            "role": role,
                            "content": chat_content
                        })

            if chat_messages:
                conversations.append(chat_messages)
                prompt_to_example[len(conversations) - 1] = ex

        # Skip if no conversations
        if not conversations:
            print(f"  No valid prompts, skipping benchmark")
            results[bench_name] = {"error": "No valid prompts"}
            continue

        # Run inference using chat API
        print(f"  Running inference on {len(conversations)} prompts...")
        start_infer = time.time()
        outputs = llm.chat(
            messages=conversations,
            sampling_params=sampling_params,
        )
        infer_time = time.time() - start_infer
        print(f"  Inference: {infer_time:.1f}s ({infer_time/len(conversations):.2f}s/sample)")

        # Score results
        from uq_eval.types import ModelResponse, Prediction as UQPrediction

        predictions = []
        correct_count = 0

        for i, output in enumerate(outputs):
            ex = prompt_to_example[i]
            response_text = output.outputs[0].text if output.outputs else ""

            try:
                # Create ModelResponse and parse prediction using benchmark API
                resp = ModelResponse(text=response_text)
                pred = benchmark.parse_prediction(ex, resp)
                score_dict = benchmark.score(ex, pred)
                score = score_dict.get("correct", 0)
                prediction = pred.answer
            except Exception as e:
                prediction = response_text[:200]
                score = -1

            if score == 1:
                correct_count += 1

            predictions.append({
                "id": ex.id,
                "target": ex.target,
                "response_text": response_text[:500],
                "prediction": prediction,
                "score": score,
            })

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

        print(f"  Qwen3-VL: {accuracy:.1%} ({correct_count}/{len(scored)})")
        if gpt5_acc:
            diff = accuracy - gpt5_acc
            print(f"  GPT-5:    {gpt5_acc:.1%}")
            print(f"  Diff:     {diff:+.1%} {'✓' if abs(diff) < 0.2 else '⚠️ CHECK'}")
        print()

        results[bench_name] = metrics

    # Summary
    print("=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"{'Benchmark':<15} {'Qwen3-VL':>10} {'GPT-5':>10} {'Diff':>10}")
    print("-" * 45)

    for bench, m in results.items():
        if "error" in m:
            print(f"{bench:<15} {'ERROR':>10}")
        else:
            qwen_acc = f"{m['accuracy']:.1%}"
            gpt5_acc = f"{m['gpt5_accuracy']:.1%}" if m.get('gpt5_accuracy') else "N/A"
            diff = f"{m['accuracy_diff']:+.1%}" if m.get('accuracy_diff') is not None else "N/A"
            print(f"{bench:<15} {qwen_acc:>10} {gpt5_acc:>10} {diff:>10}")

    # Save summary
    with open(out_base / "summary.json", "w") as f:
        json.dump({
            "model": args.model,
            "model_load_time_s": load_time,
            "benchmarks": results,
        }, f, indent=2)

    print()
    print(f"Results saved to: {out_base}")
    print("=" * 60)


if __name__ == "__main__":
    main()
