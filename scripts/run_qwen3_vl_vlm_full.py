#!/usr/bin/env python3
"""
Run Qwen3-VL on VLM benchmarks with vLLM.

This script loads the model ONCE and runs all benchmarks sequentially,
avoiding the ~2min model reload overhead per benchmark.

Usage:
    # Run with 30B model (1 GPU)
    CUDA_VISIBLE_DEVICES=0 python scripts/run_qwen3_vl_vlm_full.py

    # Run with 235B model (4 GPUs)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_qwen3_vl_vlm_full.py \
        --model "Qwen/Qwen3-VL-235B-A22B-Thinking" \
        --tensor_parallel_size 4

    # Use specific sample IDs from GPT-5-mini runs
    python scripts/run_qwen3_vl_vlm_full.py --use_gpt5_ids

Requirements:
    - finegrain_vlm conda environment with vLLM >= 0.13.0
    - 1x A100-80GB for 30B model
    - 4x A100-80GB for 235B model
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_gpt5_sample_ids(benchmark: str) -> list[str] | None:
    """Find sample IDs from GPT-5-mini runs for a benchmark."""
    runs_dir = Path("/scratch/khayes/LLM/runs")

    # Look for GPT-5-mini runs with this benchmark
    for run_dir in sorted(runs_dir.glob(f"*_{benchmark}_gpt-5-mini*"), reverse=True):
        ids_file = run_dir / "sampled_ids.json"
        if ids_file.exists():
            with open(ids_file) as f:
                ids = json.load(f)
            print(f"  Found {len(ids)} sample IDs from {run_dir.name}")
            return ids

    return None


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-VL on VLM benchmarks")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Model name or path (default: Qwen/Qwen3-VL-30B-A3B-Instruct)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=250,
        help="Max examples per benchmark (default: 250)"
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=16384,
        help="Max output tokens (default: 16384 to match GPT-5)"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["mmmu", "mathvista", "charxiv", "realworldqa", "vizwiz",
                 "erqa", "mathverse", "mathvision", "mmstar"],
        help="Benchmarks to run"
    )
    parser.add_argument(
        "--use_gpt5_ids",
        action="store_true",
        help="Use same sample IDs as GPT-5-mini runs"
    )
    parser.add_argument(
        "--extra_samples",
        type=int,
        default=0,
        help="Extra samples beyond GPT-5 IDs (for 5000 total goal)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: auto-generated)"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)"
    )
    args = parser.parse_args()

    # Import vLLM and model components
    print("=" * 60)
    print("Qwen3-VL VLM Benchmark Runner (vLLM)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Max output tokens: {args.max_output_tokens}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Use GPT-5 IDs: {args.use_gpt5_ids}")
    print()

    from vllm import LLM, SamplingParams
    from uq_eval.registry import load_benchmark
    from uq_eval.types import ModelRequest, ModelResponse
    from uq_eval.models.qwen3_vl_vllm_client import decode_data_url

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1].replace("-", "_")
    if args.out_dir:
        out_base = Path(args.out_dir)
    else:
        out_base = Path(f"/scratch/khayes/LLM/runs/{timestamp}_vlm_{model_short}")
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_base}")
    print()

    # Load model once
    print("Loading Qwen3-VL model with vLLM...")
    start_load = time.time()

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=32768,
        trust_remote_code=True,
        max_num_seqs=16,
    )

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.1f}s")
    print()

    # Sampling params
    sampling_params = SamplingParams(
        max_tokens=args.max_output_tokens,
        temperature=0.0,
    )

    # Results tracking
    all_results = {}
    total_samples = 0
    total_time = 0

    # Run each benchmark
    for bench_name in args.benchmarks:
        print("=" * 60)
        print(f"Running: {bench_name}")
        print("=" * 60)

        bench_out = out_base / bench_name
        bench_out.mkdir(exist_ok=True)

        # Load benchmark
        try:
            benchmark = load_benchmark(bench_name)
        except Exception as e:
            print(f"Error loading benchmark {bench_name}: {e}")
            all_results[bench_name] = {"error": str(e)}
            continue

        # Get examples
        examples = benchmark.get_examples()

        # Filter to GPT-5 IDs if requested
        gpt5_ids = None
        if args.use_gpt5_ids:
            gpt5_ids = find_gpt5_sample_ids(bench_name)
            if gpt5_ids:
                id_set = set(gpt5_ids)
                examples = [ex for ex in examples if ex.id in id_set]
                print(f"  Filtered to {len(examples)} examples matching GPT-5-mini IDs")

        # Sample if needed
        if len(examples) > args.max_examples:
            import random
            random.seed(args.seed)
            examples = random.sample(examples, args.max_examples)
            print(f"  Sampled {len(examples)} examples")

        # Save sample IDs
        sample_ids = [ex.id for ex in examples]
        with open(bench_out / "sampled_ids.json", "w") as f:
            json.dump(sample_ids, f)

        # Build prompts
        print(f"  Building prompts for {len(examples)} examples...")
        prompts = []
        prompt_to_example = {}

        for ex in examples:
            # Build request using benchmark's format
            req = benchmark.format_prompt(ex)

            # Extract images and build vLLM input
            images = []
            text_parts = []

            for msg in req.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if isinstance(content, str):
                    text_parts.append(f"{role}: {content}")
                elif isinstance(content, list):
                    msg_text = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") in ("text", "input_text"):
                                msg_text.append(item.get("text", ""))
                            elif item.get("type") in ("input_image", "image_url"):
                                # Extract image
                                if item.get("type") == "input_image":
                                    img_url = item.get("image_url", "")
                                else:
                                    url_data = item.get("image_url", {})
                                    img_url = url_data.get("url", "") if isinstance(url_data, dict) else url_data

                                img = decode_data_url(img_url)
                                if img:
                                    images.append(img)
                                    msg_text.append("<image>")
                        elif isinstance(item, str):
                            msg_text.append(item)
                    text_parts.append(f"{role}: {''.join(msg_text)}")

            full_prompt = "\n".join(text_parts)
            if not full_prompt.endswith("assistant:"):
                full_prompt += "\nassistant:"

            vllm_input = {"prompt": full_prompt}
            if images:
                vllm_input["multi_modal_data"] = {"image": images}

            prompts.append(vllm_input)
            prompt_to_example[len(prompts) - 1] = ex

        # Run inference
        print(f"  Running inference on {len(prompts)} prompts...")
        start_infer = time.time()

        outputs = llm.generate(prompts, sampling_params)

        infer_time = time.time() - start_infer
        per_sample = infer_time / len(prompts) if prompts else 0
        print(f"  Inference completed in {infer_time:.1f}s ({per_sample:.2f}s per sample)")

        # Process results
        predictions = []
        correct_count = 0

        for i, output in enumerate(outputs):
            ex = prompt_to_example[i]
            response_text = output.outputs[0].text if output.outputs else ""

            # Score using benchmark's scorer
            try:
                prediction = benchmark.extract_answer(response_text)
                score = benchmark.score(prediction, ex.target)
            except Exception as e:
                prediction = response_text
                score = -1  # Unable to score

            if score == 1:
                correct_count += 1

            predictions.append({
                "id": ex.id,
                "input": ex.input if isinstance(ex.input, str) else str(ex.input)[:500],
                "target": ex.target,
                "response_text": response_text,
                "prediction": {"answer": prediction},
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
            "inference_time_s": infer_time,
            "per_sample_s": per_sample,
            "benchmark": bench_name,
            "model": args.model,
        }

        with open(bench_out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  Results: {correct_count}/{len(scored)} correct ({accuracy:.1%})")
        print(f"  Saved to: {bench_out}")
        print()

        all_results[bench_name] = metrics
        total_samples += len(predictions)
        total_time += infer_time

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Total samples: {total_samples}")
    print(f"Total inference time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average per sample: {total_time/total_samples:.2f}s" if total_samples else "N/A")
    print()

    for bench, metrics in all_results.items():
        if "error" in metrics:
            print(f"  {bench}: ERROR - {metrics['error']}")
        else:
            print(f"  {bench}: {metrics['accuracy']:.1%} ({metrics['n_scored']} samples)")

    # Save overall summary
    with open(out_base / "summary.json", "w") as f:
        json.dump({
            "model": args.model,
            "total_samples": total_samples,
            "total_time_s": total_time,
            "model_load_time_s": load_time,
            "benchmarks": all_results,
        }, f, indent=2)

    print()
    print(f"All results saved to: {out_base}")
    print("=" * 60)


if __name__ == "__main__":
    main()
