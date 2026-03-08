#!/usr/bin/env python3
"""Score UNSEEN benchmarks with the v2 calibrator for out-of-distribution evaluation.

These benchmarks were NEVER in the training data for best_v2_r32_combined.
This gives us genuinely clean AUROC numbers for the paper.

Unseen benchmarks with existing predictions:
  - healthbench (gpt5mini, gpt52, qwen35)
  - triviaqa (gpt5mini only)

Usage:
    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/score_unseen_benchmarks.py --smoke_test

    # Full run
    CUDA_VISIBLE_DEVICES=0 python scripts/score_unseen_benchmarks.py
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CHECKPOINT = "uq_models/best_v2_r32_combined"

# Combined prompt (matches v2 training)
PROMPT_TEMPLATE = """Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes"""

Q_LEN = 1500
R_LEN = 800

# Unseen benchmark data sources — NONE of these were in training
UNSEEN_SOURCES = {
    "gpt5mini": {
        "name": "GPT-5-mini",
        "source_model": "gpt-5-mini",
        "benchmarks": {
            "healthbench": "runs/gpt5_mini_combined/healthbench/predictions.jsonl",
            "triviaqa": "runs/gpt5_mini_combined/triviaqa/predictions.jsonl",
        },
    },
    "gpt52": {
        "name": "GPT-5.2",
        "source_model": "gpt-5.2-high-reasoning",
        "benchmarks": {
            "healthbench": "runs/gpt52_high_healthbench/predictions.jsonl",
        },
    },
    "qwen35": {
        "name": "Qwen3.5-397B",
        "source_model": "Qwen3.5-397B-A17B-FP8",
        "benchmarks": {
            "healthbench": "runs/qwen35_397b_healthbench/predictions.jsonl",
        },
    },
}


# ============================================================
# DATA LOADING
# ============================================================

def extract_question_text(input_data) -> str:
    """Extract question text from input field."""
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, dict):
        for key in ["question", "query", "query_cot", "prompt", "text"]:
            if key in input_data and input_data[key]:
                val = input_data[key]
                if isinstance(val, str):
                    return val
        if "messages" in input_data:
            for msg in input_data["messages"]:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        texts = [p.get("text", "") for p in content
                                 if isinstance(p, dict) and "text" in p]
                        return " ".join(texts)
        clean = {k: v for k, v in input_data.items() if k != "images"}
        return json.dumps(clean)[:2000]
    if isinstance(input_data, list):
        # Chat format: list of messages
        texts = []
        for msg in input_data:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for p in content:
                        if isinstance(p, dict) and "text" in p:
                            texts.append(p["text"])
        return " ".join(texts) if texts else str(input_data)[:2000]
    return str(input_data)[:2000]


def load_predictions(pred_file: str, benchmark: str) -> list:
    """Load and filter predictions from a JSONL file."""
    samples = []
    with open(pred_file) as f:
        for line in f:
            try:
                pred = json.loads(line)
            except json.JSONDecodeError:
                continue

            score = pred.get("score", {})
            if isinstance(score, dict):
                correct = score.get("correct", -1)
            else:
                correct = score
            if correct not in (0, 1):
                continue

            question = extract_question_text(pred.get("input", {}))
            response = pred.get("response_text", "")
            if not response:
                response = str(pred.get("prediction", {}).get("answer", ""))
            if not question or not response:
                continue

            prediction = pred.get("prediction") or {}
            usage = pred.get("usage") or {}

            samples.append({
                "id": str(pred.get("id", "")),
                "benchmark": benchmark,
                "question": question[:3000],
                "response": response[:2000],
                "is_correct": int(correct == 1),
                "has_image": False,  # All unseen benchmarks are text-only
                "verbalized_confidence": prediction.get("confidence") if isinstance(prediction, dict) else None,
                "input_tokens": usage.get("input_tokens") if isinstance(usage, dict) else None,
                "output_tokens": usage.get("output_tokens") if isinstance(usage, dict) else None,
                "total_tokens": usage.get("total_tokens") if isinstance(usage, dict) else None,
            })

    return samples


# ============================================================
# INFERENCE
# ============================================================

def get_p_correct(model, processor, question: str, response: str,
                  device, benchmark="", source_model="") -> float:
    """Extract P(correct) from calibrator logits."""
    prompt = PROMPT_TEMPLATE.format(
        question=question[:Q_LEN],
        response=response[:R_LEN],
        benchmark=benchmark,
        source_model=source_model,
    )

    # Use gray placeholder image (text-only benchmarks)
    image = Image.new('RGB', (224, 224), color='gray')
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True,
        min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--output_dir", default="data/use_cases/scored_unseen")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.smoke_test:
        args.max_examples = 5

    # Load model
    print(f"Loading base model: {BASE_MODEL}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    print(f"Loading LoRA from: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()
    device = next(model.parameters()).device
    print(f"Model on device: {device}")

    all_results = {}

    for target_key, target_config in UNSEEN_SOURCES.items():
        target_name = target_config["name"]
        source_model = target_config["source_model"]

        output_path = os.path.join(args.output_dir, f"{target_key}_unseen_scored.jsonl")
        print(f"\n{'='*60}")
        print(f"Scoring {target_name} predictions on UNSEEN benchmarks")
        print(f"Output: {output_path}")
        print(f"{'='*60}")

        all_scored = []

        for benchmark, pred_file in target_config["benchmarks"].items():
            if not os.path.exists(pred_file):
                print(f"  SKIP {benchmark}: {pred_file} not found")
                continue

            samples = load_predictions(pred_file, benchmark)
            if args.max_examples:
                samples = samples[:args.max_examples]

            print(f"\n  {benchmark}: {len(samples)} samples")

            n_correct = sum(s["is_correct"] for s in samples)
            print(f"  Accuracy: {n_correct}/{len(samples)} = {n_correct/len(samples):.1%}")

            t0 = time.time()
            for i, sample in enumerate(samples):
                p = get_p_correct(
                    model, processor,
                    sample["question"], sample["response"],
                    device, benchmark=benchmark, source_model=source_model,
                )
                sample["p_correct"] = p
                sample["target_model"] = target_key

                all_scored.append(sample)

                if (i + 1) % 25 == 0 or (i + 1) == len(samples):
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    print(f"    [{i+1}/{len(samples)}] {rate:.1f} samples/sec")

            # Per-benchmark AUROC
            labels = np.array([s["is_correct"] for s in samples])
            scores = np.array([s["p_correct"] for s in samples])
            if len(set(labels)) >= 2:
                auroc = roc_auc_score(labels, scores)
                print(f"  AUROC: {auroc:.3f}")

                # Verbalized comparison
                verb = np.array([s.get("verbalized_confidence") or 0.5 for s in samples])
                v_auroc = roc_auc_score(labels, verb)
                print(f"  Verbalized AUROC: {v_auroc:.3f}")
                print(f"  Advantage: +{auroc - v_auroc:.3f}")
            else:
                auroc = float("nan")
                print(f"  AUROC: N/A (only one class)")

            all_results[f"{target_key}_{benchmark}"] = {
                "benchmark": benchmark,
                "target_model": target_key,
                "n": len(samples),
                "accuracy": float(n_correct / len(samples)),
                "auroc": float(auroc) if not np.isnan(auroc) else None,
            }

        # Write scored JSONL
        with open(output_path, "w") as f:
            for s in all_scored:
                # Remove large text fields to save space
                out = {k: v for k, v in s.items() if k not in ("question", "response")}
                out["question_preview"] = s["question"][:200]
                out["response_preview"] = s["response"][:200]
                f.write(json.dumps(out) + "\n")
        print(f"\n  Wrote {len(all_scored)} scored samples to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("UNSEEN BENCHMARK RESULTS (OUT-OF-DISTRIBUTION)")
    print("=" * 60)

    # Combined AUROC across all unseen data
    all_labels = []
    all_scores = []
    for target_key in UNSEEN_SOURCES:
        output_path = os.path.join(args.output_dir, f"{target_key}_unseen_scored.jsonl")
        if not os.path.exists(output_path):
            continue
        with open(output_path) as f:
            for line in f:
                d = json.loads(line)
                all_labels.append(d["is_correct"])
                all_scores.append(d["p_correct"])

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    if len(set(all_labels)) >= 2:
        combined_auroc = roc_auc_score(all_labels, all_scores)
        print(f"\nCombined AUROC (all unseen): {combined_auroc:.3f}")
        print(f"Total samples: {len(all_labels)}")
        print(f"Base accuracy: {all_labels.mean():.3f}")

    # Per-result summary
    print(f"\n{'Benchmark':>20} {'Model':>10} {'N':>6} {'Acc':>6} {'AUROC':>8}")
    for key, res in sorted(all_results.items()):
        auroc_str = f"{res['auroc']:.3f}" if res['auroc'] is not None else "N/A"
        print(f"{res['benchmark']:>20} {res['target_model']:>10} {res['n']:>6} "
              f"{res['accuracy']:>6.1%} {auroc_str:>8}")

    # Save summary
    summary_path = os.path.join(args.output_dir, "unseen_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "description": "Out-of-distribution evaluation on benchmarks NEVER in training data",
            "checkpoint": args.checkpoint,
            "combined_auroc": float(combined_auroc) if len(set(all_labels)) >= 2 else None,
            "total_samples": int(len(all_labels)),
            "per_benchmark": all_results,
        }, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
