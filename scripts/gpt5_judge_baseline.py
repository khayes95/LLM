#!/usr/bin/env python3
"""GPT-5-as-judge baseline: Can GPT-5 judge whether a response is correct?

This addresses Reviewer Issue 3: "The paper is missing the most obvious strong
baseline: prompting a capable LLM to judge whether the answer is correct."

The judge sees ONLY the question and response (no reference answer, no benchmark
metadata) — same information our calibrator gets (minus metadata).

We also run a "with-metadata" variant where the judge sees benchmark name and
source model, matching the calibrator's full prompt.

Usage:
    # Smoke test (5 samples)
    python scripts/gpt5_judge_baseline.py --smoke_test

    # Full run on test-only samples
    python scripts/gpt5_judge_baseline.py --judge_model gpt-5.2

    # Use GPT-5-mini as judge (cheaper)
    python scripts/gpt5_judge_baseline.py --judge_model gpt-5-mini
"""
import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

# ============================================================
# CONFIG
# ============================================================

RUNS_DIR = "runs"

SOURCE_CONFIGS = {
    "gpt5mini": {"dir": "runs/gpt5_mini_combined", "type": "combined"},
    "gpt52": {"dir": "runs", "prefix": "gpt52_high_", "type": "prefixed"},
    "qwen35": {"dir": "runs", "prefix": "qwen35_397b_", "type": "prefixed"},
}

EXCLUDED_BENCHMARKS = {
    "erqa", "babilong", "mmvet", "vizwiz",
    "hle_multimodal", "arc_agi", "oolong",
}

VLM_BENCHMARKS = {
    "aokvqa", "charxiv", "hallusionbench", "mathverse", "mathvision",
    "mathvista", "mmmu", "mmstar", "realworldqa", "vsr",
}

# Judge prompts
JUDGE_PROMPT_NO_META = """You are evaluating whether a model's answer to a question is correct.

Question: {question}

Model's Answer: {response}

Based on the response above, estimate the probability that the answer is correct.
Consider factual accuracy, logical reasoning, and completeness.

Respond with ONLY a single number between 0.0 and 1.0 representing your confidence
that the answer is correct. Do not include any other text."""

JUDGE_PROMPT_WITH_META = """You are evaluating whether a model's answer to a question is correct.

Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Model's Answer: {response}

Based on the response above, estimate the probability that the answer is correct.
Consider factual accuracy, logical reasoning, and completeness.

Respond with ONLY a single number between 0.0 and 1.0 representing your confidence
that the answer is correct. Do not include any other text."""


@dataclass
class Sample:
    id: str
    benchmark: str
    source_model: str
    question: str
    response: str
    is_correct: bool
    has_image: bool


def extract_question_text(input_data) -> str:
    """Extract question text from input field (matches train_best_uq.py)."""
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
    return str(input_data)[:2000]


def load_combined_dir(combined_dir: str, source_model: str):
    """Load predictions from a combined directory."""
    samples = []
    combined_path = Path(combined_dir)
    for bench_dir in sorted(combined_path.iterdir()):
        if not bench_dir.is_dir():
            continue
        benchmark = bench_dir.name
        if benchmark in EXCLUDED_BENCHMARKS:
            continue
        pred_file = bench_dir / "predictions.jsonl"
        if not pred_file.exists():
            continue
        with open(pred_file) as f:
            for line in f:
                try:
                    pred = json.loads(line)
                except json.JSONDecodeError:
                    continue
                score = pred.get("score", -1)
                if isinstance(score, dict):
                    correct = score.get("correct", -1)
                else:
                    correct = score
                if correct not in (0, 1):
                    continue
                question = extract_question_text(pred.get("input", {}))
                response = pred.get("response_text", "") or str(pred.get("prediction", ""))
                if not question or not response:
                    continue
                samples.append(Sample(
                    id=str(pred.get("id", "")),
                    benchmark=benchmark,
                    source_model=source_model,
                    question=question[:2000],
                    response=response[:1000],
                    is_correct=bool(correct == 1),
                    has_image=benchmark in VLM_BENCHMARKS,
                ))
    return samples


def load_prefixed_runs(runs_dir: str, prefix: str, source_model: str):
    """Load predictions from prefixed run directories."""
    samples = []
    runs_path = Path(runs_dir)
    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.name.startswith(prefix):
            continue
        benchmark = run_dir.name[len(prefix):]
        if benchmark in EXCLUDED_BENCHMARKS:
            continue
        pred_file = run_dir / "predictions.jsonl"
        if not pred_file.exists():
            continue
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
                samples.append(Sample(
                    id=str(pred.get("id", "")),
                    benchmark=benchmark,
                    source_model=source_model,
                    question=question[:2000],
                    response=response[:1000],
                    is_correct=bool(correct == 1),
                    has_image=benchmark in VLM_BENCHMARKS,
                ))
    return samples


def load_all_samples():
    """Load ALL samples from all source models (same as train_best_uq.py)."""
    all_samples = []
    for model_key, config in SOURCE_CONFIGS.items():
        if config["type"] == "combined":
            samples = load_combined_dir(config["dir"], model_key)
        else:
            samples = load_prefixed_runs(config["dir"], config["prefix"], model_key)
        print(f"  {model_key}: {len(samples)} samples")
        all_samples.extend(samples)
    return all_samples


def filter_to_test(samples, split_info_path):
    """Filter samples to test-only using split_info.json."""
    with open(split_info_path) as f:
        split_info = json.load(f)
    test_ids = set(split_info["test_ids"])
    filtered = [s for s in samples
                if f"{s.benchmark}_{s.id}" in test_ids or s.id in test_ids]
    print(f"  Filtered {len(samples)} -> {len(filtered)} test-only samples")
    return filtered


def parse_probability(text: str) -> float:
    """Parse a probability value from GPT-5's response."""
    text = text.strip()
    # Try direct float parse
    try:
        val = float(text)
        return max(0.0, min(1.0, val))
    except ValueError:
        pass
    # Try to find a float in the text
    import re
    matches = re.findall(r'\b(0\.\d+|1\.0|0|1)\b', text)
    if matches:
        return float(matches[0])
    # Fall back: look for yes/no
    lower = text.lower()
    if "yes" in lower or "correct" in lower:
        return 0.8
    if "no" in lower or "incorrect" in lower:
        return 0.2
    return 0.5


async def judge_batch_async(samples, judge_model, prompt_template, max_concurrent=20,
                            q_trunc=1500, r_trunc=800):
    """Send samples to GPT-5 for judging, with async concurrency."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(samples)
    total_input_tokens = 0
    total_output_tokens = 0

    async def judge_one(idx, sample):
        nonlocal total_input_tokens, total_output_tokens
        prompt = prompt_template.format(
            question=sample.question[:q_trunc],
            response=sample.response[:r_trunc],
            benchmark=sample.benchmark,
            source_model=sample.source_model,
        )
        async with semaphore:
            for attempt in range(3):
                try:
                    kwargs = {
                        "model": judge_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_completion_tokens": 150,
                    }
                    # Only set temperature for non-reasoning models
                    if "mini" not in judge_model and "5" not in judge_model:
                        kwargs["temperature"] = 0.0
                    resp = await client.chat.completions.create(**kwargs)
                    text = resp.choices[0].message.content or ""
                    usage = resp.usage
                    if usage:
                        total_input_tokens += usage.prompt_tokens
                        total_output_tokens += usage.completion_tokens
                    results[idx] = parse_probability(text)
                    return
                except Exception as e:
                    if attempt == 2:
                        print(f"  FAILED sample {idx} ({sample.id}): {e}")
                        results[idx] = 0.5  # fallback
                    else:
                        await asyncio.sleep(2 ** attempt)

    tasks = [judge_one(i, s) for i, s in enumerate(samples)]

    # Progress tracking
    done_count = 0
    start = time.time()
    batch_size = 50
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        await asyncio.gather(*batch)
        done_count += len(batch)
        elapsed = time.time() - start
        rate = done_count / elapsed if elapsed > 0 else 0
        print(f"  Progress: {done_count}/{len(tasks)} ({rate:.1f}/s)")

    print(f"  Total tokens: {total_input_tokens} input, {total_output_tokens} output")
    return results, total_input_tokens, total_output_tokens


def compute_metrics(labels, scores, name=""):
    """Compute AUROC, Brier, and per-benchmark breakdown."""
    labels = np.array(labels)
    scores = np.array(scores)

    result = {"name": name, "n": len(labels), "pos_rate": float(labels.mean())}

    if len(set(labels)) >= 2:
        result["auroc"] = float(roc_auc_score(labels, scores))
        result["brier"] = float(brier_score_loss(labels, scores))
    else:
        result["auroc"] = None
        result["brier"] = None

    return result


def bootstrap_ci(labels, scores, n_bootstrap=1000, seed=42):
    """Bootstrap confidence interval for AUROC."""
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    scores = np.array(scores)
    aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(labels), len(labels), replace=True)
        if len(set(labels[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(labels[idx], scores[idx]))
    if not aurocs:
        return None, None
    return float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


def main():
    parser = argparse.ArgumentParser(description="GPT-5-as-judge baseline")
    parser.add_argument("--judge_model", default="gpt-5-mini",
                        help="OpenAI model to use as judge (gpt-5-mini, gpt-5.2)")
    parser.add_argument("--split_info", default="uq_models/best_v2_r32_combined/split_info.json",
                        help="Path to split_info.json for test-only filtering")
    parser.add_argument("--output_dir", default="data/ablations/gpt5_judge_baseline",
                        help="Output directory")
    parser.add_argument("--max_concurrent", type=int, default=20,
                        help="Max concurrent API calls")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run on 5 samples only")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap total samples")
    parser.add_argument("--q_trunc", type=int, default=1500,
                        help="Question truncation length")
    parser.add_argument("--r_trunc", type=int, default=800,
                        help="Response truncation length")
    parser.add_argument("--text_only", action="store_true",
                        help="Skip VLM benchmark samples (judge can't see images)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load data and estimate cost without making API calls")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    # Load all samples
    print("Loading samples...")
    all_samples = load_all_samples()
    print(f"Total samples loaded: {len(all_samples)}")

    # Filter to test-only
    if Path(args.split_info).exists():
        samples = filter_to_test(all_samples, args.split_info)
    else:
        print(f"WARNING: split_info not found at {args.split_info}, using ALL samples")
        samples = all_samples

    # Skip VLM if requested (GPT-5 text judge can't evaluate image-based questions)
    if args.text_only:
        before = len(samples)
        samples = [s for s in samples if not s.has_image]
        print(f"  Text-only filter: {before} -> {len(samples)}")

    # Subsample
    if args.smoke_test:
        samples = samples[:5]
        print(f"  Smoke test: using {len(samples)} samples")
    elif args.max_examples:
        samples = samples[:args.max_examples]
        print(f"  Capped at {len(samples)} samples")

    # Cost estimate
    avg_input_tokens = 400  # rough estimate per sample
    avg_output_tokens = 10
    n = len(samples)
    # 2 prompt variants × n samples
    total_calls = 2 * n

    # Rough pricing (USD per 1M tokens) — adjust for actual model
    if "mini" in args.judge_model:
        input_price = 0.30  # per 1M tokens
        output_price = 1.20
    else:
        input_price = 2.00
        output_price = 8.00

    est_cost = (total_calls * avg_input_tokens * input_price / 1e6 +
                total_calls * avg_output_tokens * output_price / 1e6)

    print(f"\n=== Cost Estimate ===")
    print(f"  Judge model: {args.judge_model}")
    print(f"  Samples: {n}")
    print(f"  Prompt variants: 2 (no-metadata, with-metadata)")
    print(f"  Total API calls: {total_calls}")
    print(f"  Estimated cost: ${est_cost:.2f}")
    print(f"  Estimated time: {total_calls / 20:.0f}s at {args.max_concurrent} concurrent")

    if args.dry_run:
        print("\n  DRY RUN — no API calls made.")
        # Save sample info for review
        summary = {
            "judge_model": args.judge_model,
            "n_samples": n,
            "n_calls": total_calls,
            "estimated_cost_usd": round(est_cost, 2),
            "text_only": args.text_only,
            "per_model": {},
            "per_benchmark": {},
        }
        for s in samples:
            summary["per_model"][s.source_model] = summary["per_model"].get(s.source_model, 0) + 1
            summary["per_benchmark"][s.benchmark] = summary["per_benchmark"].get(s.benchmark, 0) + 1
        with open(output_dir / "dry_run_estimate.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved estimate to {output_dir / 'dry_run_estimate.json'}")
        return

    # Run both prompt variants
    labels = [int(s.is_correct) for s in samples]

    results = {}
    all_per_sample = []

    for variant_name, template in [
        ("no_metadata", JUDGE_PROMPT_NO_META),
        ("with_metadata", JUDGE_PROMPT_WITH_META),
    ]:
        print(f"\n=== Running variant: {variant_name} ===")
        scores, in_tok, out_tok = asyncio.run(
            judge_batch_async(samples, args.judge_model, template,
                              max_concurrent=args.max_concurrent,
                              q_trunc=args.q_trunc, r_trunc=args.r_trunc)
        )

        # Overall metrics
        metrics = compute_metrics(labels, scores, name=f"gpt5_judge_{variant_name}")
        ci_lo, ci_hi = bootstrap_ci(labels, scores)
        metrics["ci_lower"] = ci_lo
        metrics["ci_upper"] = ci_hi
        metrics["input_tokens"] = in_tok
        metrics["output_tokens"] = out_tok

        # Per-model metrics
        metrics["per_model"] = {}
        for model_key in ["gpt5mini", "gpt52", "qwen35"]:
            idx = [i for i, s in enumerate(samples) if s.source_model == model_key]
            if idx:
                m_labels = [labels[i] for i in idx]
                m_scores = [scores[i] for i in idx]
                metrics["per_model"][model_key] = compute_metrics(m_labels, m_scores, model_key)

        # Per-benchmark metrics
        metrics["per_benchmark"] = {}
        benchmarks = set(s.benchmark for s in samples)
        for bench in sorted(benchmarks):
            idx = [i for i, s in enumerate(samples) if s.benchmark == bench]
            if idx:
                b_labels = [labels[i] for i in idx]
                b_scores = [scores[i] for i in idx]
                metrics["per_benchmark"][bench] = compute_metrics(b_labels, b_scores, bench)

        results[variant_name] = metrics
        print(f"  AUROC: {metrics.get('auroc', 'N/A')}")
        if ci_lo:
            print(f"  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

        # Save per-sample predictions
        for i, s in enumerate(samples):
            all_per_sample.append({
                "id": s.id,
                "benchmark": s.benchmark,
                "source_model": s.source_model,
                "is_correct": int(s.is_correct),
                "variant": variant_name,
                f"p_judge_{variant_name}": scores[i],
            })

    # Load calibrator scores for comparison
    scored_dir = Path("data/use_cases/scored_test_only_v2")
    cal_scores = {}
    for scored_file in scored_dir.glob("*_scored.jsonl"):
        with open(scored_file) as f:
            for line in f:
                d = json.loads(line)
                cal_scores[d["id"]] = d.get("p_correct")

    # Compare calibrator vs judge
    comparison = {"judge_model": args.judge_model, "variants": {}}
    for variant_name in results:
        comp = {"judge": results[variant_name]}

        # Match calibrator scores to judge scores
        matched_labels = []
        matched_cal = []
        matched_judge = []
        for i, s in enumerate(samples):
            if s.id in cal_scores and cal_scores[s.id] is not None:
                matched_labels.append(int(s.is_correct))
                matched_cal.append(cal_scores[s.id])
                judge_key = f"p_judge_{variant_name}"
                # Find judge score for this sample
                for ps in all_per_sample:
                    if ps["id"] == s.id and ps["variant"] == variant_name:
                        matched_judge.append(ps[judge_key])
                        break

        if len(set(matched_labels)) >= 2 and matched_cal and matched_judge:
            comp["calibrator_auroc"] = float(roc_auc_score(matched_labels, matched_cal))
            comp["judge_auroc"] = float(roc_auc_score(matched_labels, matched_judge))
            comp["n_matched"] = len(matched_labels)
            comp["delta"] = comp["calibrator_auroc"] - comp["judge_auroc"]

        comparison["variants"][variant_name] = comp

    results["comparison"] = comparison

    # Save results
    with open(output_dir / "judge_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save per-sample JSONL
    with open(output_dir / "judge_per_sample.jsonl", "w") as f:
        for item in all_per_sample:
            f.write(json.dumps(item) + "\n")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: GPT-5-as-Judge Baseline")
    print("=" * 60)
    print(f"Judge model: {args.judge_model}")
    print(f"Samples: {len(samples)}")
    for variant_name, metrics in results.items():
        if variant_name == "comparison":
            continue
        auroc = metrics.get("auroc", "N/A")
        ci_lo = metrics.get('ci_lower')
        ci_hi = metrics.get('ci_upper')
        ci = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if ci_lo is not None else "[N/A]"
        print(f"  {variant_name}: AUROC = {auroc} {ci}")

    if "comparison" in results:
        for vn, comp in results["comparison"].get("variants", {}).items():
            if "delta" in comp:
                print(f"  Calibrator vs {vn}: {comp['calibrator_auroc']:.3f} vs {comp['judge_auroc']:.3f} (delta={comp['delta']:+.3f})")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
