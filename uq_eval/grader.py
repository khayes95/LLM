"""Offline grading script for benchmarks that need LLM-as-judge.

Usage:
    python -m uq_eval.grader --predictions runs/<run>/predictions.jsonl --judge_model gpt-5-mini

Benchmark-specific judge recommendations:
    - HealthBench: gpt-5-mini or gpt-5 (rubric-based, medical)
    - TutorBench: claude-3-5-sonnet or gpt-5-mini (rubric-based, education)
    - PRBench: gpt-5-mini or o1-mini (rubric-based, legal/finance)
    - HLE: gpt-5-mini or o1-mini (short answer correctness)

Note: The official benchmarks may use specific models, but any capable LLM
works as an approximation. Results should be consistent across judge models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import iter_jsonl
from .judge import JudgeResult, judge_correctness, judge_rubric
from .registry import load_model_client


# Recommended judge models per benchmark (approximations)
# Using gpt-5-mini ($0.25/$2.00) - better cost/performance than gpt-4o
BENCHMARK_JUDGE_DEFAULTS = {
    "healthbench": "gpt-5-mini",
    "tutorbench": "gpt-5-mini",
    "prbench": "gpt-5-mini",
    "hle": "gpt-5-mini",
    "bullshitbench": "gpt-5-mini",
}


def grade_predictions(
    predictions_path: str,
    output_path: str | None = None,
    judge_backend: str = "chat_http",
    judge_model: str | None = None,  # Auto-detect from benchmark if None
    base_url: str | None = None,
    max_examples: int | None = None,
) -> dict:
    """Grade predictions using LLM judge.

    Args:
        predictions_path: Path to predictions.jsonl file
        output_path: Where to save graded results (default: same dir as input)
        judge_backend: Model backend for judge
        judge_model: Model name for judge
        base_url: API base URL
        max_examples: Limit number of examples to grade

    Returns:
        Summary stats dict
    """
    predictions = list(iter_jsonl(Path(predictions_path)))

    if max_examples:
        predictions = predictions[:max_examples]

    # Auto-detect benchmark from predictions path or first example
    detected_benchmark = None
    path_lower = predictions_path.lower()
    for bench_name in BENCHMARK_JUDGE_DEFAULTS:
        if bench_name in path_lower:
            detected_benchmark = bench_name
            break

    if not detected_benchmark and predictions:
        # Try to detect from example ID
        first_id = predictions[0].get("example", {}).get("id", "")
        for bench_name in BENCHMARK_JUDGE_DEFAULTS:
            if bench_name in first_id.lower():
                detected_benchmark = bench_name
                break

    # Use detected or default judge model
    if judge_model is None:
        judge_model = BENCHMARK_JUDGE_DEFAULTS.get(detected_benchmark, "gpt-5-mini")
        if detected_benchmark:
            print(f"Auto-detected benchmark: {detected_benchmark}, using judge: {judge_model}")

    # Initialize judge client
    client_kwargs = {"model_name": judge_model}
    if base_url:
        client_kwargs["base_url"] = base_url
    judge_client = load_model_client(judge_backend, **client_kwargs)

    graded = []
    stats = {"total": 0, "correct": 0, "incorrect": 0, "uncertain": 0, "errors": 0}

    for pred in predictions:
        stats["total"] += 1

        # Skip if already graded
        score = pred.get("score", {})
        if score.get("correct", -1) != -1 and not score.get("needs_grading"):
            graded.append(pred)
            if score.get("correct") == 1:
                stats["correct"] += 1
            else:
                stats["incorrect"] += 1
            continue

        # Determine grading method based on metadata
        example_meta = pred.get("example", {}).get("meta", {})
        rubrics = example_meta.get("rubrics") or score.get("rubrics")

        try:
            if rubrics:
                # Use rubric-based grading
                question = example_meta.get("last_user_message", "")
                if not question:
                    # Try to get from example input
                    ex_input = pred.get("example", {}).get("input", [])
                    if isinstance(ex_input, list):
                        for msg in reversed(ex_input):
                            if msg.get("role") == "user":
                                question = msg.get("content", "")
                                break
                    else:
                        question = str(ex_input)

                result = judge_rubric(
                    client=judge_client,
                    question=question,
                    model_answer=pred.get("prediction", {}).get("answer", ""),
                    rubric=rubrics,
                )
            else:
                # Use correctness-based grading
                question = str(pred.get("example", {}).get("input", ""))
                reference = str(pred.get("example", {}).get("target", ""))
                model_answer = pred.get("prediction", {}).get("answer", "")

                result = judge_correctness(
                    client=judge_client,
                    question=question,
                    reference_answer=reference,
                    model_answer=model_answer,
                )

            # Update prediction with graded score
            pred["score"]["correct"] = result.correct
            pred["score"]["judge_score"] = result.score
            pred["score"]["judge_reasoning"] = result.reasoning
            pred["score"]["needs_grading"] = False

            if result.correct == 1:
                stats["correct"] += 1
            elif result.correct == 0:
                stats["incorrect"] += 1
            else:
                stats["uncertain"] += 1

            # Compute brier score if we have confidence
            confidence = pred.get("prediction", {}).get("confidence")
            if confidence is not None and result.correct in (0, 1):
                pred["score"]["brier"] = (float(confidence) - result.correct) ** 2

        except Exception as e:
            pred["score"]["grading_error"] = str(e)
            stats["errors"] += 1

        graded.append(pred)

        # Progress
        if stats["total"] % 10 == 0:
            print(f"Graded {stats['total']}/{len(predictions)}...")

    # Save graded results
    if output_path is None:
        output_path = str(Path(predictions_path).parent / "predictions_graded.jsonl")

    # Write graded results
    with open(output_path, "w", encoding="utf-8") as f:
        for record in graded:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # Compute final stats
    judged = stats["correct"] + stats["incorrect"]
    if judged > 0:
        stats["accuracy"] = stats["correct"] / judged
    else:
        stats["accuracy"] = None

    print(f"\nGrading complete:")
    print(f"  Total: {stats['total']}")
    print(f"  Correct: {stats['correct']}")
    print(f"  Incorrect: {stats['incorrect']}")
    print(f"  Uncertain: {stats['uncertain']}")
    print(f"  Errors: {stats['errors']}")
    if stats["accuracy"] is not None:
        print(f"  Accuracy: {stats['accuracy']:.2%}")
    print(f"\nSaved to: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Grade predictions using LLM judge")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--output", help="Output path (default: predictions_graded.jsonl)")
    parser.add_argument("--judge_backend", default="chat_http", help="Judge model backend")
    parser.add_argument("--judge_model", default="gpt-5-mini", help="Judge model name")
    parser.add_argument("--base_url", help="API base URL")
    parser.add_argument("--max_examples", type=int, help="Max examples to grade")

    args = parser.parse_args()

    grade_predictions(
        predictions_path=args.predictions,
        output_path=args.output,
        judge_backend=args.judge_backend,
        judge_model=args.judge_model,
        base_url=args.base_url,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
