#!/usr/bin/env python3
"""
Grade predictions using Qwen3-VL as LLM judge.

This script takes predictions from LLM-judged benchmarks (mmvet, prbench)
and grades them using Qwen3-VL as the judge model.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/grade_with_qwen3_vl.py \
        --predictions runs/<run>/mmvet/predictions.jsonl

To run after inference completes, use the --wait_for option to wait for
specific run directories to have predictions.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_judge_response(text: str) -> tuple[int, float, str]:
    """Parse judge response, extracting verdict, score, and reasoning."""
    # Try JSON parsing first
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            verdict = obj.get("verdict", obj.get("correct", ""))
            score = obj.get("score", obj.get("confidence", 0.5))
            reasoning = obj.get("reasoning", obj.get("explanation", ""))

            # Normalize verdict to int
            if isinstance(verdict, bool):
                correct = 1 if verdict else 0
            elif isinstance(verdict, (int, float)):
                correct = 1 if verdict > 0.5 else 0
            elif isinstance(verdict, str):
                v = verdict.lower().strip()
                if v in ("correct", "yes", "true", "pass", "1"):
                    correct = 1
                elif v in ("incorrect", "no", "false", "fail", "0"):
                    correct = 0
                else:
                    correct = -1
            else:
                correct = -1

            # Normalize score
            try:
                score = float(score)
                score = max(0.0, min(1.0, score))
            except:
                score = 0.5

            return correct, score, str(reasoning)
    except:
        pass

    # Fallback: look for keywords
    text_lower = text.lower()
    if "correct" in text_lower and "incorrect" not in text_lower:
        return 1, 0.8, text
    elif "incorrect" in text_lower or "wrong" in text_lower:
        return 0, 0.2, text

    return -1, 0.5, text


def build_judge_prompt(question: str, model_answer: str, reference: str = None) -> list:
    """Build judge prompt for simple correctness check."""
    system = """You are an expert judge evaluating answer correctness.
Compare the model's answer to the reference answer and determine if it is correct.
Consider semantic equivalence - answers don't need to be word-for-word identical.

Return a JSON object with:
- "verdict": "correct" or "incorrect"
- "score": 0.0 to 1.0 confidence in your judgment
- "reasoning": brief explanation of your decision"""

    user_content = f"""Question: {question}

Reference Answer: {reference if reference else "N/A"}

Model's Answer: {model_answer}

Is the model's answer correct?"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def main():
    parser = argparse.ArgumentParser(description="Grade predictions with Qwen3-VL judge")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions.jsonl file or run directory"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-30B-A3B-Thinking",
        help="Judge model name"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: predictions_graded.jsonl in same dir)"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Max examples to grade"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    # Resolve predictions path
    pred_path = Path(args.predictions)
    if pred_path.is_dir():
        pred_path = pred_path / "predictions.jsonl"

    if not pred_path.exists():
        print(f"Error: {pred_path} not found")
        return

    print("=" * 60)
    print("Grading with Qwen3-VL Judge")
    print("=" * 60)
    print(f"Predictions: {pred_path}")
    print(f"Judge model: {args.model}")
    print()

    # Load predictions
    print("Loading predictions...")
    predictions = []
    with open(pred_path) as f:
        for line in f:
            predictions.append(json.loads(line))

    if args.max_examples:
        predictions = predictions[:args.max_examples]

    print(f"  Loaded {len(predictions)} predictions")

    # Filter to only those needing grading
    needs_grading = [p for p in predictions if p.get("score", {}).get("correct", -1) == -1]
    print(f"  {len(needs_grading)} need grading")

    if not needs_grading:
        print("No predictions need grading, exiting")
        return

    # Load judge model
    print()
    print("Loading Qwen3-VL judge model...")
    start_load = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=8192,  # Shorter context for judging
        trust_remote_code=True,
        max_num_seqs=16,
    )
    load_time = time.time() - start_load
    print(f"Judge model loaded in {load_time:.1f}s")
    print()

    sampling_params = SamplingParams(max_tokens=512, temperature=0.0)

    # Build judge prompts
    print("Building judge prompts...")
    judge_prompts = []
    prompt_indices = []

    for i, pred in enumerate(predictions):
        if pred.get("score", {}).get("correct", -1) != -1:
            continue  # Already graded

        # Get question and answers
        question = pred.get("input", pred.get("question", ""))
        if isinstance(question, dict):
            question = question.get("question", str(question))

        model_answer = pred.get("prediction", "")
        reference = pred.get("target", "")

        # Build prompt
        messages = build_judge_prompt(question, model_answer, reference)
        judge_prompts.append(messages)
        prompt_indices.append(i)

    print(f"  Built {len(judge_prompts)} judge prompts")

    # Run judge inference
    print()
    print(f"Running judge inference on {len(judge_prompts)} examples...")
    start_infer = time.time()
    outputs = llm.chat(
        messages=judge_prompts,
        sampling_params=sampling_params,
    )
    infer_time = time.time() - start_infer
    print(f"Judge inference: {infer_time:.1f}s ({infer_time/len(judge_prompts):.2f}s/sample)")

    # Parse judge results and update predictions
    correct_count = 0
    incorrect_count = 0
    uncertain_count = 0

    for j, output in enumerate(outputs):
        pred_idx = prompt_indices[j]
        response_text = output.outputs[0].text if output.outputs else ""

        correct, score, reasoning = parse_judge_response(response_text)

        predictions[pred_idx]["score"]["correct"] = correct
        predictions[pred_idx]["score"]["judge_score"] = score
        predictions[pred_idx]["score"]["judge_reasoning"] = reasoning[:500]
        predictions[pred_idx]["score"]["judge_model"] = args.model

        if correct == 1:
            correct_count += 1
        elif correct == 0:
            incorrect_count += 1
        else:
            uncertain_count += 1

        # Compute brier score if we have confidence
        confidence = predictions[pred_idx].get("confidence")
        if confidence is not None and correct in (0, 1):
            predictions[pred_idx]["score"]["brier"] = (float(confidence) - correct) ** 2

    # Save graded results
    output_path = args.output
    if output_path is None:
        output_path = pred_path.parent / "predictions_graded.jsonl"
    else:
        output_path = Path(output_path)

    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    # Calculate final metrics
    judged = correct_count + incorrect_count
    accuracy = correct_count / judged if judged > 0 else None

    print()
    print("=" * 60)
    print("GRADING RESULTS")
    print("=" * 60)
    print(f"Total: {len(predictions)}")
    print(f"Graded: {judged}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {incorrect_count}")
    print(f"Uncertain: {uncertain_count}")
    if accuracy is not None:
        print(f"Accuracy: {accuracy:.1%}")
    print()
    print(f"Saved to: {output_path}")

    # Update metrics file
    metrics_path = pred_path.parent / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    metrics["n_graded"] = judged
    metrics["n_correct"] = correct_count
    metrics["n_incorrect"] = incorrect_count
    metrics["n_uncertain"] = uncertain_count
    metrics["accuracy"] = accuracy
    metrics["judge_model"] = args.model
    metrics["judge_time_s"] = infer_time

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Updated metrics: {metrics_path}")


if __name__ == "__main__":
    main()
