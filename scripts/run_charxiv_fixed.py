#!/usr/bin/env python3
"""
Run Qwen3-VL-30B on CharXiv using GPT-5-mini predictions for ground truth.

The CharXiv dataset on HuggingFace was updated and no longer has reasoning_a.
This script loads the dataset for images but uses GPT-5-mini predictions for targets.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_charxiv_fixed.py
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


def load_gpt5_predictions() -> dict[str, dict]:
    """Load GPT-5-mini predictions for charxiv."""
    pred_file = Path("/scratch/khayes/LLM/runs/gpt5_mini_combined/charxiv/predictions.jsonl")
    predictions = {}
    with open(pred_file) as f:
        for line in f:
            pred = json.loads(line)
            predictions[pred["id"]] = pred
    return predictions


def encode_image_to_data_url(img) -> str:
    """Convert PIL image to base64 data URL."""
    from PIL import Image
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"


def main():
    parser = argparse.ArgumentParser(description="CharXiv eval with GPT-5-mini targets")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-30B-A3B-Thinking",
        help="Model name"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=250,
        help="Max samples (default: 250 to match GPT-5-mini)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated)"
    )
    args = parser.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from PIL import Image

    print("=" * 60)
    print("CharXiv Evaluation (with GPT-5-mini targets)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Max samples: {args.max_samples}")
    print()

    # Load GPT-5-mini predictions
    print("Loading GPT-5-mini predictions...")
    gpt5_preds = load_gpt5_predictions()
    print(f"  Loaded {len(gpt5_preds)} predictions")

    # Load charxiv dataset for images
    print("Loading CharXiv dataset...")
    ds = load_dataset("princeton-nlp/CharXiv", split="test")
    print(f"  Dataset size: {len(ds)}")

    # Build examples matching GPT-5-mini sample IDs
    gpt5_ids_file = Path("/scratch/khayes/LLM/runs/gpt5_mini_combined/charxiv/sampled_ids.json")
    with open(gpt5_ids_file) as f:
        sample_ids = json.load(f)

    print(f"  Using {len(sample_ids)} sample IDs from GPT-5-mini")

    # Create output directory
    if args.output_dir:
        out_base = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split("/")[-1].replace("-", "_")
        out_base = Path(f"/scratch/khayes/LLM/runs/{timestamp}_charxiv_{model_short}")
    out_base.mkdir(parents=True, exist_ok=True)

    bench_out = out_base / "charxiv"
    bench_out.mkdir(exist_ok=True)

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

    # Build index mapping from charxiv_{idx} to dataset row
    idx_to_row = {}
    for idx, row in enumerate(ds):
        idx_to_row[f"charxiv_{idx}"] = (idx, row)

    # Build prompts
    examples = []
    conversations = []

    for sample_id in sample_ids[:args.max_samples]:
        if sample_id not in gpt5_preds:
            print(f"  Warning: {sample_id} not in GPT-5 predictions")
            continue
        if sample_id not in idx_to_row:
            print(f"  Warning: {sample_id} not in dataset")
            continue

        gpt5_pred = gpt5_preds[sample_id]
        idx, row = idx_to_row[sample_id]

        # Get image
        img = row.get("image")
        if not isinstance(img, Image.Image):
            print(f"  Warning: {sample_id} has no valid image")
            continue

        # Get question from GPT-5 input
        question = gpt5_pred.get("input", {}).get("question", "")
        if not question:
            print(f"  Warning: {sample_id} has no question")
            continue

        # Get target from GPT-5 prediction
        target = gpt5_pred.get("target", "")

        # Build prompt
        system = (
            "You are an expert at analyzing scientific figures and charts.\n"
            "Study the figure carefully and answer the question.\n"
            'Return a JSON object with keys: "reasoning" (your analysis of the figure), '
            '"answer" (your concise answer), and "confidence" (0..1).'
        )

        # Encode image
        img_url = encode_image_to_data_url(img)

        chat_messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": question}
                ]
            }
        ]

        examples.append({
            "id": sample_id,
            "target": target,
            "question": question,
        })
        conversations.append(chat_messages)

    print(f"Built {len(conversations)} prompts")

    if not conversations:
        print("No valid prompts, exiting")
        return

    # Save sample IDs
    with open(bench_out / "sampled_ids.json", "w") as f:
        json.dump([ex["id"] for ex in examples], f)

    # Run inference
    print(f"Running inference on {len(conversations)} prompts...")
    start_infer = time.time()
    outputs = llm.chat(
        messages=conversations,
        sampling_params=sampling_params,
    )
    infer_time = time.time() - start_infer
    print(f"Inference: {infer_time:.1f}s ({infer_time/len(conversations):.2f}s/sample)")

    # Score results
    predictions = []
    correct_count = 0

    for i, output in enumerate(outputs):
        ex = examples[i]
        response_text = output.outputs[0].text if output.outputs else ""

        # Parse JSON response
        import re
        answer = None
        confidence = None

        # Try to extract JSON
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
            answer = response_text.strip()[:200]

        # Simple scoring - check if target is in answer or vice versa
        gold = ex["target"].lower().strip()
        got = str(answer).lower().strip()

        score = 0
        if gold == got:
            score = 1
        elif gold and got:
            if gold in got or got in gold:
                score = 1

        if score == 1:
            correct_count += 1

        predictions.append({
            "id": ex["id"],
            "target": ex["target"],
            "response_text": response_text[:1000],
            "prediction": answer,
            "score": score,
            "confidence": confidence,
        })

    # Save predictions
    with open(bench_out / "predictions.jsonl", "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    # Calculate metrics
    accuracy = correct_count / len(predictions) if predictions else 0

    # Load GPT-5 accuracy
    gpt5_metrics_file = Path("/scratch/khayes/LLM/runs/gpt5_mini_combined/charxiv/metrics.json")
    gpt5_acc = None
    if gpt5_metrics_file.exists():
        with open(gpt5_metrics_file) as f:
            gpt5_acc = json.load(f).get("accuracy", None)

    metrics = {
        "n": len(predictions),
        "accuracy": accuracy,
        "gpt5_accuracy": gpt5_acc,
        "accuracy_diff": accuracy - gpt5_acc if gpt5_acc else None,
        "inference_time_s": infer_time,
    }

    with open(bench_out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Qwen3-VL: {accuracy:.1%} ({correct_count}/{len(predictions)})")
    if gpt5_acc:
        print(f"GPT-5:    {gpt5_acc:.1%}")
        print(f"Diff:     {accuracy - gpt5_acc:+.1%}")
    print()
    print(f"Results saved to: {bench_out}")


if __name__ == "__main__":
    main()
