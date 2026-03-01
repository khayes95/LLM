#!/usr/bin/env python3
"""
Run Qwen3-VL-30B on LLM-judged benchmarks (prbench, mmvet).

These benchmarks require LLM-as-judge grading and have GPT-5-mini accuracy in range:
- prbench: 79.2% (text-only, rubric-based legal/finance)
- mmvet: 30% (VLM, general visual QA)

Note: tutorbench (98%) and healthbench (93.2%) are excluded - too high accuracy.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_qwen3_vl_llm_judged.py --benchmark prbench
    CUDA_VISIBLE_DEVICES=1 python scripts/run_qwen3_vl_llm_judged.py --benchmark mmvet
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from io import BytesIO
import base64

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_gpt5_predictions(benchmark: str) -> dict[str, dict]:
    """Load GPT-5-mini predictions for target benchmark."""
    pred_file = Path(f"/scratch/khayes/LLM/runs/gpt5_mini_combined/{benchmark}/predictions.jsonl")
    predictions = {}
    with open(pred_file) as f:
        for line in f:
            pred = json.loads(line)
            predictions[pred["id"]] = pred
    return predictions


def load_gpt5_sample_ids(benchmark: str) -> list[str]:
    """Load sample IDs used in GPT-5-mini evaluation."""
    ids_file = Path(f"/scratch/khayes/LLM/runs/gpt5_mini_combined/{benchmark}/sampled_ids.json")
    with open(ids_file) as f:
        return json.load(f)


def encode_image_to_data_url(img) -> str:
    """Convert PIL image to base64 data URL."""
    from PIL import Image
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"


def build_prbench_examples(gpt5_preds: dict, sample_ids: list, max_samples: int):
    """Build prbench examples from GPT-5 predictions (text-only)."""
    examples = []
    conversations = []

    for sample_id in sample_ids[:max_samples]:
        if sample_id not in gpt5_preds:
            continue

        gpt5_pred = gpt5_preds[sample_id]

        # Get request messages from GPT-5 prediction
        request = gpt5_pred.get("request", {})
        messages = request.get("messages", [])

        if not messages:
            continue

        # Get target from GPT-5 score (may have been graded)
        score = gpt5_pred.get("score", {})
        target = gpt5_pred.get("target", "")
        correct = score.get("correct", -1)

        examples.append({
            "id": sample_id,
            "target": target,
            "gpt5_correct": correct,
            "gpt5_prediction": gpt5_pred.get("prediction", {}).get("answer", ""),
        })
        conversations.append(messages)

    return examples, conversations


def build_mmvet_examples(gpt5_preds: dict, sample_ids: list, max_samples: int):
    """Build mmvet examples - requires loading images from dataset."""
    from datasets import load_dataset
    from PIL import Image

    print("Loading MM-Vet dataset...")
    ds = load_dataset("lmms-lab/MMVet", split="test")

    # Build index by question_id
    ds_by_id = {}
    for idx, row in enumerate(ds):
        qid = str(row.get("question_id", idx))
        ds_by_id[qid] = row

    examples = []
    conversations = []

    for sample_id in sample_ids[:max_samples]:
        if sample_id not in gpt5_preds:
            continue

        gpt5_pred = gpt5_preds[sample_id]

        # Get image from dataset
        if sample_id not in ds_by_id:
            print(f"  Warning: {sample_id} not in MM-Vet dataset")
            continue

        row = ds_by_id[sample_id]
        img = row.get("image")
        if not isinstance(img, Image.Image):
            print(f"  Warning: {sample_id} has no valid image")
            continue

        question = row.get("question", "")
        target = row.get("answer", "")

        # Build prompt with image
        system = (
            "You are a multimodal assistant capable of recognition, OCR, knowledge retrieval, and reasoning.\n"
            "Analyze the image carefully and provide a concise, accurate answer.\n"
            'Return a JSON object with keys: "reasoning" (your analysis), "answer" (concise answer), '
            'and "confidence" (0..1).'
        )

        img_url = encode_image_to_data_url(img)

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": question}
                ]
            }
        ]

        # Get GPT-5 grading info
        score = gpt5_pred.get("score", {})

        examples.append({
            "id": sample_id,
            "target": target,
            "question": question,
            "gpt5_correct": score.get("correct", -1),
            "gpt5_prediction": gpt5_pred.get("prediction", {}).get("answer", ""),
        })
        conversations.append(messages)

    return examples, conversations


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL on LLM-judged benchmarks")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["prbench", "mmvet"],
        help="Which benchmark to run"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-30B-A3B-Thinking",
        help="Model name"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=250,
        help="Max samples (default: 250)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated)"
    )
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    print("=" * 60)
    print(f"LLM-Judged Benchmark: {args.benchmark}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Max samples: {args.max_samples}")
    print()

    # Load GPT-5-mini predictions
    print("Loading GPT-5-mini predictions...")
    gpt5_preds = load_gpt5_predictions(args.benchmark)
    sample_ids = load_gpt5_sample_ids(args.benchmark)
    print(f"  Loaded {len(gpt5_preds)} predictions, {len(sample_ids)} sample IDs")

    # Build examples based on benchmark type
    if args.benchmark == "prbench":
        examples, conversations = build_prbench_examples(
            gpt5_preds, sample_ids, args.max_samples
        )
    elif args.benchmark == "mmvet":
        examples, conversations = build_mmvet_examples(
            gpt5_preds, sample_ids, args.max_samples
        )
    else:
        print(f"Unknown benchmark: {args.benchmark}")
        return

    print(f"Built {len(conversations)} prompts")

    if not conversations:
        print("No valid prompts, exiting")
        return

    # Create output directory
    if args.output_dir:
        out_base = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split("/")[-1].replace("-", "_")
        out_base = Path(f"/scratch/khayes/LLM/runs/{timestamp}_{args.benchmark}_{model_short}")
    out_base.mkdir(parents=True, exist_ok=True)

    bench_out = out_base / args.benchmark
    bench_out.mkdir(exist_ok=True)

    # Save sample IDs
    with open(bench_out / "sampled_ids.json", "w") as f:
        json.dump([ex["id"] for ex in examples], f)

    # Load model
    print()
    print("Loading Qwen3-VL model...")
    start_load = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=32768,
        trust_remote_code=True,
        max_num_seqs=16,
    )
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.1f}s")
    print()

    sampling_params = SamplingParams(max_tokens=2048, temperature=0.0)

    # Run inference
    print(f"Running inference on {len(conversations)} prompts...")
    start_infer = time.time()
    outputs = llm.chat(
        messages=conversations,
        sampling_params=sampling_params,
    )
    infer_time = time.time() - start_infer
    print(f"Inference: {infer_time:.1f}s ({infer_time/len(conversations):.2f}s/sample)")

    # Parse results
    import re
    predictions = []

    for i, output in enumerate(outputs):
        ex = examples[i]
        response_text = output.outputs[0].text if output.outputs else ""

        # Parse JSON response
        answer = None
        confidence = None

        json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                obj = json.loads(json_match.group())
                answer = obj.get("answer", "")
                conf = obj.get("confidence")
                if conf is not None:
                    confidence = max(0, min(1, float(conf)))
            except:
                pass

        if not answer:
            answer = response_text.strip()[:500]

        predictions.append({
            "id": ex["id"],
            "target": ex.get("target", ""),
            "response_text": response_text[:2000],
            "prediction": answer,
            "confidence": confidence,
            "gpt5_correct": ex.get("gpt5_correct", -1),
            "gpt5_prediction": ex.get("gpt5_prediction", ""),
            "score": {"correct": -1},  # Needs LLM judge grading
        })

    # Save predictions
    with open(bench_out / "predictions.jsonl", "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    # Calculate metrics (without grading - just count)
    with_confidence = sum(1 for p in predictions if p["confidence"] is not None)

    metrics = {
        "n": len(predictions),
        "n_with_confidence": with_confidence,
        "needs_grading": True,
        "inference_time_s": infer_time,
        "model": args.model,
        "benchmark": args.benchmark,
    }

    with open(bench_out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("=" * 60)
    print("RESULTS (Pre-Grading)")
    print("=" * 60)
    print(f"Samples: {len(predictions)}")
    print(f"With confidence: {with_confidence}")
    print(f"Inference time: {infer_time:.1f}s")
    print()
    print(f"NOTE: This benchmark requires LLM-as-judge grading.")
    print(f"Run: python -m uq_eval.grader --predictions {bench_out}/predictions.jsonl")
    print()
    print(f"Results saved to: {bench_out}")


if __name__ == "__main__":
    main()
