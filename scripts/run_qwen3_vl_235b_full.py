#!/usr/bin/env python3
"""
Run Qwen3-VL-235B-A22B-Thinking on all meaningful benchmarks.

Uses the GPT-5-mini combined sample IDs for fair cross-model comparison.
Only runs benchmarks in the 15-85% accuracy range with 150+ samples.

Usage:
    # Full run on 8 A100s
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/run_qwen3_vl_235b_full.py

    # Test with fewer GPUs (slower)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_qwen3_vl_235b_full.py --tensor_parallel_size 4

Requirements:
    - 8x A100-80GB recommended (235B MoE with 22B active)
    - finegrain_vlm conda environment with vLLM >= 0.13.0
    - ~40 min for 3600 samples at ~1.5s/sample
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Meaningful benchmarks: 15-85% GPT-5-mini accuracy, 150+ samples
# Matches the smoke test benchmark list for consistency
MEANINGFUL_BENCHMARKS = [
    # VLM benchmarks (10) - same as smoke test
    ("charxiv", 250, 0.636, "VLM"),
    ("erqa", 250, 0.584, "VLM"),
    ("mathverse", 250, 0.448, "VLM"),
    ("mathvision", 250, 0.336, "VLM"),
    ("mathvista", 250, 0.344, "VLM"),
    ("mmstar", 250, 0.784, "VLM"),
    ("realworldqa", 250, 0.516, "VLM"),
    ("vizwiz", 250, 0.552, "VLM"),
    ("hle_multimodal", 150, None, "VLM"),  # Partial samples
    ("mmmu", 150, 0.720, "VLM"),
    # Text benchmarks (7) - same as smoke test
    ("bbeh", 294, 0.418, "Text"),
    ("chembench", 250, 0.716, "Text"),
    ("gpqa", 198, 0.773, "Text"),
    ("livebench", 250, 0.680, "Text"),
    ("omnimath", 239, 0.364, "Text"),
    ("simpleqa", 250, None, "Text"),
    ("hle", 233, 0.159, "Text"),
]


def load_gpt5_predictions(benchmark: str) -> dict[str, dict] | None:
    """Load predictions from GPT-5-mini combined dataset keyed by ID."""
    combined_dir = Path("/scratch/khayes/LLM/runs/gpt5_mini_combined")
    preds_file = combined_dir / benchmark / "predictions.jsonl"
    if preds_file.exists():
        preds = {}
        with open(preds_file) as f:
            for line in f:
                p = json.loads(line)
                preds[p["id"]] = p
        return preds
    return None


def load_gpt5_sample_ids(benchmark: str) -> list[str] | None:
    """Load sample IDs from combined GPT-5-mini dataset."""
    combined_dir = Path("/scratch/khayes/LLM/runs/gpt5_mini_combined")
    ids_file = combined_dir / benchmark / "sampled_ids.json"
    if ids_file.exists():
        with open(ids_file) as f:
            return json.load(f)
    return None


def sample_examples(examples: list, max_samples: int, seed: int = 42) -> list:
    """Deterministically sample examples using seed."""
    import random
    if len(examples) <= max_samples:
        return examples
    random.seed(seed)
    return random.sample(examples, max_samples)


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-VL-235B on meaningful benchmarks")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-235B-A22B-Thinking",
        help="Model name (default: 235B Thinking model)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
        help="Number of GPUs for tensor parallelism (default: 8)"
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=2048,
        help="Max output tokens (default: 2048)"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)"
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
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Override benchmark list"
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: auto-generated)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=250,
        help="Max samples per benchmark (default: 250)"
    )
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from uq_eval.registry import load_benchmark

    # Build benchmark list
    if args.benchmarks:
        benchmarks = [(b, None, None, None) for b in args.benchmarks]
    elif args.vlm_only:
        benchmarks = [b for b in MEANINGFUL_BENCHMARKS if b[3] == "VLM"]
    elif args.text_only:
        benchmarks = [b for b in MEANINGFUL_BENCHMARKS if b[3] == "Text"]
    else:
        benchmarks = MEANINGFUL_BENCHMARKS

    total_samples = sum(b[1] or 250 for b in benchmarks)

    print("=" * 70)
    print("Qwen3-VL-235B Full Benchmark Run")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Max output tokens: {args.max_output_tokens}")
    print(f"Benchmarks: {len(benchmarks)}")
    print(f"Total samples: ~{total_samples}")
    print()
    print("Benchmark list:")
    for name, samples, gpt5_acc, btype in benchmarks:
        if gpt5_acc:
            print(f"  {name:<15} {samples:>4} samples  GPT-5: {gpt5_acc:.1%}  ({btype})")
        else:
            print(f"  {name:<15}")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1].replace("-", "_")
    if args.out_dir:
        out_base = Path(args.out_dir)
    else:
        out_base = Path(f"/scratch/khayes/LLM/runs/{timestamp}_full_{model_short}")
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_base}")
    print()

    # Load model
    print("Loading Qwen3-VL-235B model with vLLM...")
    print(f"  tensor_parallel_size={args.tensor_parallel_size}")
    print(f"  gpu_memory_utilization={args.gpu_memory_utilization}")
    start_load = time.time()

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=16384,  # Reduced for 235B model
        trust_remote_code=True,
        max_num_seqs=8,  # Reduced for 235B model
        enforce_eager=True,  # More stable for large models
        distributed_executor_backend="mp",  # Use multiprocessing executor
        disable_custom_all_reduce=True,  # Avoid custom NCCL issues
    )

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.1f}s")
    print()

    sampling_params = SamplingParams(
        max_tokens=args.max_output_tokens,
        temperature=0.0,
    )

    results = {}
    total_time = 0
    total_processed = 0

    for bench_info in benchmarks:
        bench_name = bench_info[0]
        expected_samples = bench_info[1] if len(bench_info) > 1 else None
        gpt5_acc = bench_info[2] if len(bench_info) > 2 else None

        print("=" * 70)
        print(f"Running: {bench_name}")
        if gpt5_acc:
            print(f"  GPT-5-mini accuracy: {gpt5_acc:.1%}")
        print("=" * 70)

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

        # Try to filter to GPT-5 IDs for fair comparison
        gpt5_ids = load_gpt5_sample_ids(bench_name)
        if gpt5_ids:
            id_set = set(gpt5_ids)
            matched_examples = [ex for ex in all_examples if ex.id in id_set]
            if len(matched_examples) >= len(gpt5_ids) * 0.8:  # 80% match threshold
                examples = matched_examples
                print(f"  Matched {len(examples)} GPT-5 sample IDs (from {len(all_examples)} total)")
            else:
                # IDs don't match - use seed-based sampling instead
                print(f"  ID mismatch (matched {len(matched_examples)}/{len(gpt5_ids)}), using seed-based sampling")
                max_for_bench = expected_samples if expected_samples else args.max_samples
                examples = sample_examples(all_examples, max_for_bench, args.seed)
                print(f"  Sampled {len(examples)} examples with seed={args.seed}")
        else:
            # No GPT-5 IDs, use seed-based sampling
            max_for_bench = expected_samples if expected_samples else args.max_samples
            examples = sample_examples(all_examples, max_for_bench, args.seed)
            print(f"  No GPT-5 IDs, sampled {len(examples)} examples with seed={args.seed}")

        if not examples:
            print(f"  No examples found, skipping")
            results[bench_name] = {"error": "No examples found"}
            continue

        # Save sample IDs
        sample_ids = [ex.id for ex in examples]
        with open(bench_out / "sampled_ids.json", "w") as f:
            json.dump(sample_ids, f)

        # Build prompts using vLLM chat format with inline images
        print(f"  Building prompts for {len(examples)} examples...")
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
        outputs = llm.chat(messages=conversations, sampling_params=sampling_params)
        infer_time = time.time() - start_infer
        per_sample = infer_time / len(conversations) if conversations else 0
        print(f"  Inference: {infer_time:.1f}s ({per_sample:.2f}s/sample)")

        # Score results
        from uq_eval.types import ModelResponse

        predictions = []
        correct_count = 0

        for i, output in enumerate(outputs):
            ex = prompt_to_example[i]
            response_text = output.outputs[0].text if output.outputs else ""

            try:
                # Use benchmark's parse_prediction and score methods
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

        metrics = {
            "n": len(predictions),
            "n_scored": len(scored),
            "accuracy": accuracy,
            "gpt5_accuracy": gpt5_acc,
            "accuracy_diff": accuracy - gpt5_acc if gpt5_acc else None,
            "inference_time_s": infer_time,
            "per_sample_s": per_sample,
            "benchmark": bench_name,
            "model": args.model,
        }

        with open(bench_out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  Qwen3-VL-235B: {accuracy:.1%} ({correct_count}/{len(scored)})")
        if gpt5_acc:
            diff = accuracy - gpt5_acc
            print(f"  GPT-5-mini:    {gpt5_acc:.1%}")
            print(f"  Difference:    {diff:+.1%}")
        print(f"  Saved to: {bench_out}")
        print()

        results[bench_name] = metrics
        total_time += infer_time
        total_processed += len(predictions)

    # Summary
    print("=" * 70)
    print("FULL RUN SUMMARY")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Total samples: {total_processed}")
    print(f"Total inference time: {total_time:.1f}s ({total_time/60:.1f} min)")
    if total_processed > 0:
        print(f"Average per sample: {total_time/total_processed:.2f}s")
    print()

    print(f"{'Benchmark':<15} {'Qwen-235B':>10} {'GPT-5':>10} {'Diff':>10} {'N':>6}")
    print("-" * 55)

    for bench, m in results.items():
        if "error" in m:
            print(f"{bench:<15} {'ERROR':>10}")
        else:
            qwen_acc = f"{m['accuracy']:.1%}"
            gpt5_acc = f"{m['gpt5_accuracy']:.1%}" if m.get('gpt5_accuracy') else "N/A"
            diff = f"{m['accuracy_diff']:+.1%}" if m.get('accuracy_diff') is not None else "N/A"
            n = m.get('n_scored', m.get('n', 0))
            print(f"{bench:<15} {qwen_acc:>10} {gpt5_acc:>10} {diff:>10} {n:>6}")

    # Save overall summary
    with open(out_base / "summary.json", "w") as f:
        json.dump({
            "model": args.model,
            "total_samples": total_processed,
            "total_time_s": total_time,
            "model_load_time_s": load_time,
            "benchmarks": results,
        }, f, indent=2)

    print()
    print(f"All results saved to: {out_base}")
    print("=" * 70)


if __name__ == "__main__":
    main()
