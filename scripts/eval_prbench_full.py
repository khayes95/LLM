#!/usr/bin/env python3
"""Full PRBench evaluation: generate responses + grade with LLM judge.

Runs all 4 PRBench splits (legal, legal_hard, finance, finance_hard) across
GPT-5-mini and GPT-5.2. Skips already-completed examples (resume support).

Pipeline per model x split:
1. Generate responses via OpenAI API
2. Grade with GPT-5-mini judge (rubric-based)

Usage:
    # Smoke test (5 examples per split)
    python scripts/eval_prbench_full.py --smoke_test

    # Legal splits only
    python scripts/eval_prbench_full.py --splits legal,legal_hard

    # Full run
    python scripts/eval_prbench_full.py

    # Grade only (skip generation)
    python scripts/eval_prbench_full.py --skip_generate

    # Summary only
    python scripts/eval_prbench_full.py --summary_only
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uq_eval.benchmarks.prbench import PRBenchBenchmark
from uq_eval.registry import load_model_client
from uq_eval.runner import run_eval
from uq_eval.grader import grade_predictions


API_MODELS = {
    "gpt5mini": {"backend": "openai", "model_name": "gpt-5-mini"},
    "gpt52": {"backend": "openai", "model_name": "gpt-5.2"},
}

PRBENCH_SPLITS = ["legal", "legal_hard", "finance", "finance_hard"]

BASE_DIR = Path("/scratch/khayes/LLM")
RUNS_DIR = BASE_DIR / "runs"


def count_predictions(pred_path):
    if not pred_path.exists():
        return 0
    with open(pred_path) as f:
        return sum(1 for _ in f)


def count_graded(pred_path):
    if not pred_path.exists():
        return 0, 0
    graded = 0
    total = 0
    with open(pred_path) as f:
        for line in f:
            total += 1
            d = json.loads(line)
            score = d.get("score", {})
            if isinstance(score, dict) and score.get("correct", -1) != -1:
                graded += 1
    return graded, total


def generate_for_split(model_key, split, api_key, max_examples=None):
    """Generate responses for one model x split combination."""
    model_cfg = API_MODELS[model_key]
    out_dir = RUNS_DIR / f"prbench_{split}_{model_key}"
    pred_path = out_dir / "predictions.jsonl"

    # Check existing
    existing = count_predictions(pred_path)
    bench = PRBenchBenchmark(domain=split)
    total_available = len(list(bench.iter_examples("dev")))
    target = min(max_examples, total_available) if max_examples else total_available

    if existing >= target:
        print(f"[SKIP] {out_dir.name}: {existing}/{target} already done")
        return out_dir

    print(f"\n[GEN] {out_dir.name}: {existing} existing, targeting {target}")

    model = load_model_client(
        model_cfg["backend"],
        model_name=model_cfg["model_name"],
        api_key=api_key,
    )

    bench = PRBenchBenchmark(domain=split)

    kwargs = {}
    if model_cfg["model_name"] == "gpt-5.2":
        kwargs["reasoning_effort"] = "medium"

    run_eval(
        model=model,
        bench=bench,
        split="dev",
        out_dir=out_dir,
        max_examples=max_examples,
        resume=True,
        max_output_tokens=16384,
        **kwargs,
    )

    final_count = count_predictions(pred_path)
    print(f"[GEN] {out_dir.name}: {final_count} predictions total")
    return out_dir


def grade_for_split(run_dir, api_key):
    """Grade ungraded predictions using GPT-5-mini judge."""
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        print(f"[SKIP] No predictions at {pred_path}")
        return

    graded, total = count_graded(pred_path)
    if graded >= total:
        print(f"[SKIP] {run_dir.name}: all {total} already graded")
        return

    print(f"[GRADE] {run_dir.name}: {graded}/{total} graded, grading {total - graded}...")

    os.environ["OPENAI_API_KEY"] = api_key
    grade_predictions(
        predictions_path=str(pred_path),
        output_path=str(pred_path),
        judge_backend="openai",
        judge_model="gpt-5-mini",
    )


def summarize_run(run_dir):
    """Print summary for a graded run."""
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        return None

    correct = 0
    incorrect = 0
    ungraded = 0
    domains = {}

    with open(pred_path) as f:
        for line in f:
            d = json.loads(line)
            score = d.get("score", {})
            c = score.get("correct", -1)
            domain = d.get("meta", {}).get("domain", "?")

            if domain not in domains:
                domains[domain] = {"correct": 0, "incorrect": 0, "total": 0}
            domains[domain]["total"] += 1

            if c == 1:
                correct += 1
                domains[domain]["correct"] += 1
            elif c == 0:
                incorrect += 1
                domains[domain]["incorrect"] += 1
            else:
                ungraded += 1

    total = correct + incorrect + ungraded
    acc = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0

    print(f"\n--- {run_dir.name} ---")
    print(f"  Total: {total}, Correct: {correct}, Incorrect: {incorrect}, Ungraded: {ungraded}")
    print(f"  Accuracy: {acc:.1%}")
    for dom, stats in sorted(domains.items()):
        dom_graded = stats["correct"] + stats["incorrect"]
        dom_acc = stats["correct"] / dom_graded if dom_graded > 0 else 0
        print(f"    {dom}: {stats['total']} total, {dom_graded} graded, {dom_acc:.1%} acc")

    return {
        "run": run_dir.name,
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "ungraded": ungraded,
        "accuracy": round(acc, 3),
        "domains": domains,
    }


def main():
    parser = argparse.ArgumentParser(description="Full PRBench evaluation pipeline")
    parser.add_argument("--smoke_test", action="store_true", help="5 examples per split")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--splits", default=None, help="Comma-separated (default: all)")
    parser.add_argument("--models", default=None, help="Comma-separated (default: gpt5mini,gpt52)")
    parser.add_argument("--skip_generate", action="store_true")
    parser.add_argument("--skip_grade", action="store_true")
    parser.add_argument("--summary_only", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    max_examples = 5 if args.smoke_test else args.max_examples
    splits = args.splits.split(",") if args.splits else PRBENCH_SPLITS
    models = args.models.split(",") if args.models else list(API_MODELS.keys())

    print(f"PRBench Full Evaluation")
    print(f"  Splits: {splits}")
    print(f"  Models: {models}")
    print(f"  Max examples: {max_examples or 'all'}")

    all_runs = []

    for model_key in models:
        if model_key not in API_MODELS:
            print(f"[SKIP] Unknown model: {model_key}")
            continue

        for split in splits:
            run_dir = RUNS_DIR / f"prbench_{split}_{model_key}"

            if not args.summary_only:
                if not args.skip_generate:
                    try:
                        generate_for_split(model_key, split, api_key, max_examples)
                    except Exception as e:
                        print(f"[ERROR] Generation failed for {model_key}/{split}: {e}")

                if not args.skip_grade:
                    try:
                        grade_for_split(run_dir, api_key)
                    except Exception as e:
                        print(f"[ERROR] Grading failed for {model_key}/{split}: {e}")

            summary = summarize_run(run_dir)
            if summary:
                all_runs.append(summary)

    # Save combined summary
    summary_path = BASE_DIR / "data" / "prbench_full_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
