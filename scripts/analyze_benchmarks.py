#!/usr/bin/env python3
"""
Analyze all benchmarks for GPT-5 evaluation planning.
Generates a report with sample counts, context lengths, costs, and recommendations.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from uq_eval.registry import load_benchmark, _BENCH_REGISTRY


@dataclass
class BenchmarkStats:
    name: str
    num_examples: int = 0
    gpt5_accuracy: float = 0.5  # Estimated
    estimated_correct: int = 0
    estimated_incorrect: int = 0
    avg_input_tokens: int = 0
    max_input_tokens: int = 0
    avg_output_tokens: int = 500  # Estimate
    scoring_type: str = "auto"  # auto, execution, rubric
    category: str = "general"
    issues: list = field(default_factory=list)


# GPT-5 accuracy estimates from benchmarks.json
GPT5_ACCURACY = {
    # Tier 1: Ideal range
    "bbeh": 0.50,
    "simpleqa": 0.40,  # 19-54%, use middle
    "tutorbench": 0.55,
    "multichallenge": 0.60,
    "healthbench": 0.60,
    "bigcodebench": 0.56,
    "multinrc": 0.65,
    "prbench": 0.50,
    # Tier 2: Lower accuracy
    "hle": 0.28,  # 25-30%
    "ether0": 0.40,  # 20-60%
    "arc_agi": 0.10,
    "chembench": 0.40,  # 8-70%, varies by subset
    # Tier 3: Higher accuracy
    "gpqa": 0.84,  # 77-90%
    "omnimath": 0.72,
    "livebench": 0.79,
    # Tier 4: Coding
    "livecodebench": 0.50,  # 4-90%, varies
    "swebench": 0.65,  # 52-75%
    # Tier 5: Long context
    "oolong": 0.58,  # 47-70%
    "babilong": 0.70,  # varies
    "longbench_v2": 0.63,
    # Tier 6: Standard (too easy)
    "math": 0.96,
    "mmlu_pro": 0.87,
    "gsm8k": 0.95,
    "arc": 0.95,
    "drop": 0.90,
    "triviaqa": 0.90,
    "hellaswag": 0.95,
    "winogrande": 0.95,
    "mgsm": 0.90,
}

SCORING_TYPES = {
    "auto": ["bbeh", "simpleqa", "multichallenge", "multinrc", "hle", "ether0",
             "arc_agi", "chembench", "gpqa", "omnimath", "livebench", "oolong",
             "babilong", "longbench_v2", "math", "mmlu_pro", "gsm8k", "arc",
             "drop", "triviaqa", "hellaswag", "winogrande", "mgsm"],
    "execution": ["bigcodebench", "livecodebench", "swebench", "humaneval", "mbpp"],
    "rubric": ["healthbench", "tutorbench", "prbench"],
}

CATEGORIES = {
    "reasoning": ["bbeh", "multichallenge", "multinrc", "omnimath", "math", "gsm8k", "mgsm", "drop"],
    "knowledge": ["simpleqa", "gpqa", "hle", "mmlu_pro", "triviaqa", "arc"],
    "science": ["ether0", "chembench", "healthbench"],
    "coding": ["bigcodebench", "livecodebench", "swebench", "humaneval", "mbpp"],
    "long_context": ["oolong", "babilong", "longbench_v2"],
    "instruction": ["tutorbench", "livebench", "hellaswag", "winogrande"],
    "professional": ["prbench"],
    "abstract": ["arc_agi"],
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4)."""
    return len(str(text)) // 4


def analyze_benchmark(name: str, max_samples: int = 100) -> BenchmarkStats:
    """Analyze a single benchmark."""
    stats = BenchmarkStats(name=name)

    # Get scoring type
    for stype, benchmarks in SCORING_TYPES.items():
        if name in benchmarks:
            stats.scoring_type = stype
            break

    # Get category
    for cat, benchmarks in CATEGORIES.items():
        if name in benchmarks:
            stats.category = cat
            break

    # Get GPT-5 accuracy estimate
    stats.gpt5_accuracy = GPT5_ACCURACY.get(name, 0.5)

    try:
        # Load benchmark with appropriate options
        kwargs = {}
        if name == "prbench":
            kwargs["domain"] = "all"
        elif name == "healthbench":
            kwargs["subset"] = "hard"

        bench = load_benchmark(name, **kwargs)

        # Sample examples to estimate sizes
        input_tokens = []
        examples_seen = 0

        for ex in bench.iter_examples("test"):
            input_tokens.append(estimate_tokens(ex.input))
            examples_seen += 1
            if examples_seen >= max_samples:
                break

        # Get full count (if small enough)
        if examples_seen < max_samples:
            stats.num_examples = examples_seen
        else:
            # Count all
            count = 0
            for _ in bench.iter_examples("test"):
                count += 1
            stats.num_examples = count

        if input_tokens:
            stats.avg_input_tokens = sum(input_tokens) // len(input_tokens)
            stats.max_input_tokens = max(input_tokens)

        # Estimate correct/incorrect
        stats.estimated_correct = int(stats.num_examples * stats.gpt5_accuracy)
        stats.estimated_incorrect = stats.num_examples - stats.estimated_correct

        # Check for issues
        if stats.scoring_type == "execution":
            stats.issues.append("Requires code execution for scoring")
        if stats.scoring_type == "rubric":
            stats.issues.append("Requires LLM judge for scoring")
        if stats.max_input_tokens > 8000:
            stats.issues.append(f"Long context: max {stats.max_input_tokens} tokens")
        if stats.num_examples < 100:
            stats.issues.append(f"Small dataset: only {stats.num_examples} examples")
        if stats.gpt5_accuracy > 0.85:
            stats.issues.append(f"Too easy: {stats.gpt5_accuracy*100:.0f}% accuracy, need pass@k")
        if stats.gpt5_accuracy < 0.15:
            stats.issues.append(f"Very hard: only {stats.gpt5_accuracy*100:.0f}% accuracy")

    except Exception as e:
        stats.issues.append(f"Failed to load: {str(e)[:100]}")

    return stats


def main():
    print("=" * 80)
    print("BENCHMARK ANALYSIS FOR GPT-5 EVALUATION")
    print("=" * 80)
    print()

    # Define which benchmarks to analyze
    # Focus on hard benchmarks for GPT-5
    hard_benchmarks = [
        # Tier 1: Ideal range (40-60%)
        "bbeh", "simpleqa", "tutorbench", "multichallenge",
        "healthbench", "bigcodebench", "multinrc", "prbench",
        # Tier 2: Lower accuracy
        "hle", "ether0", "arc_agi", "chembench",
        # Tier 3: Higher accuracy
        "gpqa", "omnimath", "livebench",
        # Tier 4: Coding
        "livecodebench", "swebench",
        # Tier 5: Long context
        "oolong", "babilong", "longbench_v2",
    ]

    easy_benchmarks = [
        "math", "mmlu_pro", "gsm8k", "arc", "drop",
        "triviaqa", "hellaswag", "winogrande", "mgsm",
    ]

    all_stats = []

    print("Analyzing hard benchmarks...")
    print("-" * 40)
    for name in hard_benchmarks:
        print(f"  {name}...", end=" ", flush=True)
        stats = analyze_benchmark(name)
        all_stats.append(stats)
        print(f"{stats.num_examples} examples")

    print()
    print("Analyzing easy benchmarks (for reference)...")
    print("-" * 40)
    for name in easy_benchmarks:
        print(f"  {name}...", end=" ", flush=True)
        stats = analyze_benchmark(name, max_samples=50)
        all_stats.append(stats)
        print(f"{stats.num_examples} examples")

    # Separate by type
    hard_stats = [s for s in all_stats if s.name in hard_benchmarks]
    easy_stats = [s for s in all_stats if s.name in easy_benchmarks]

    # Calculate totals
    print()
    print("=" * 80)
    print("SUMMARY: HARD BENCHMARKS (Recommended for GPT-5)")
    print("=" * 80)

    total_examples = sum(s.num_examples for s in hard_stats)
    total_correct = sum(s.estimated_correct for s in hard_stats)
    total_incorrect = sum(s.estimated_incorrect for s in hard_stats)

    print(f"\nTotal examples: {total_examples:,}")
    print(f"Estimated correct: {total_correct:,} ({total_correct/total_examples*100:.1f}%)")
    print(f"Estimated incorrect: {total_incorrect:,} ({total_incorrect/total_examples*100:.1f}%)")

    # By scoring type
    print("\n--- By Scoring Type ---")
    for stype in ["auto", "execution", "rubric"]:
        subset = [s for s in hard_stats if s.scoring_type == stype]
        if subset:
            n = sum(s.num_examples for s in subset)
            names = ", ".join(s.name for s in subset)
            print(f"{stype.upper()}: {n:,} examples")
            print(f"  Benchmarks: {names}")

    # By category
    print("\n--- By Category ---")
    cat_stats = defaultdict(lambda: {"examples": 0, "correct": 0, "incorrect": 0, "benchmarks": []})
    for s in hard_stats:
        cat_stats[s.category]["examples"] += s.num_examples
        cat_stats[s.category]["correct"] += s.estimated_correct
        cat_stats[s.category]["incorrect"] += s.estimated_incorrect
        cat_stats[s.category]["benchmarks"].append(s.name)

    for cat, data in sorted(cat_stats.items(), key=lambda x: -x[1]["examples"]):
        print(f"\n{cat.upper()}: {data['examples']:,} examples")
        print(f"  Correct: {data['correct']:,} | Incorrect: {data['incorrect']:,}")
        print(f"  Benchmarks: {', '.join(data['benchmarks'])}")

    # Context length analysis
    print("\n" + "=" * 80)
    print("CONTEXT LENGTH ANALYSIS")
    print("=" * 80)

    short_context = [s for s in hard_stats if s.max_input_tokens <= 4000]
    medium_context = [s for s in hard_stats if 4000 < s.max_input_tokens <= 16000]
    long_context = [s for s in hard_stats if s.max_input_tokens > 16000]

    print(f"\nShort context (≤4K tokens): {len(short_context)} benchmarks, {sum(s.num_examples for s in short_context):,} examples")
    print(f"  {', '.join(s.name for s in short_context)}")

    print(f"\nMedium context (4K-16K tokens): {len(medium_context)} benchmarks, {sum(s.num_examples for s in medium_context):,} examples")
    print(f"  {', '.join(s.name for s in medium_context)}")

    print(f"\nLong context (>16K tokens): {len(long_context)} benchmarks, {sum(s.num_examples for s in long_context):,} examples")
    print(f"  {', '.join(s.name for s in long_context)}")

    # Cost estimation
    print("\n" + "=" * 80)
    print("COST ESTIMATION (GPT-5 / GPT-4o pricing)")
    print("=" * 80)

    # GPT-4o pricing: $2.50/1M input, $10/1M output (as reference)
    # GPT-5 pricing unknown, estimate 2x GPT-4o
    input_price_per_m = 5.0  # $/1M tokens (estimate)
    output_price_per_m = 20.0  # $/1M tokens (estimate)

    total_input_tokens = sum(s.num_examples * s.avg_input_tokens for s in hard_stats)
    total_output_tokens = sum(s.num_examples * s.avg_output_tokens for s in hard_stats)

    input_cost = (total_input_tokens / 1_000_000) * input_price_per_m
    output_cost = (total_output_tokens / 1_000_000) * output_price_per_m
    total_cost = input_cost + output_cost

    print(f"\nEstimated tokens (all hard benchmarks):")
    print(f"  Input: {total_input_tokens:,} tokens")
    print(f"  Output: {total_output_tokens:,} tokens (estimated)")
    print(f"\nEstimated cost (at $5/1M in, $20/1M out):")
    print(f"  Input: ${input_cost:.2f}")
    print(f"  Output: ${output_cost:.2f}")
    print(f"  TOTAL: ${total_cost:.2f}")

    # Without long context
    non_long = [s for s in hard_stats if s.category != "long_context"]
    nl_input = sum(s.num_examples * s.avg_input_tokens for s in non_long)
    nl_output = sum(s.num_examples * s.avg_output_tokens for s in non_long)
    nl_cost = (nl_input / 1_000_000) * input_price_per_m + (nl_output / 1_000_000) * output_price_per_m

    print(f"\nWithout long-context benchmarks:")
    print(f"  Examples: {sum(s.num_examples for s in non_long):,}")
    print(f"  Estimated cost: ${nl_cost:.2f}")

    # Issues and recommendations
    print("\n" + "=" * 80)
    print("ISSUES & RECOMMENDATIONS")
    print("=" * 80)

    print("\n--- Benchmarks Requiring Special Handling ---")
    for s in hard_stats:
        if s.issues:
            print(f"\n{s.name}:")
            for issue in s.issues:
                print(f"  ⚠️  {issue}")

    print("\n--- Recommendations for UQ Classifier Training ---")

    auto_scorable = [s for s in hard_stats if s.scoring_type == "auto"]
    auto_examples = sum(s.num_examples for s in auto_scorable)
    auto_correct = sum(s.estimated_correct for s in auto_scorable)
    auto_incorrect = sum(s.estimated_incorrect for s in auto_scorable)

    print(f"""
1. AUTO-SCORABLE BENCHMARKS (Recommended for initial training):
   - {len(auto_scorable)} benchmarks with {auto_examples:,} examples
   - Estimated: {auto_correct:,} correct, {auto_incorrect:,} incorrect
   - Balance ratio: {auto_incorrect/auto_correct:.2f}:1 (incorrect:correct)
   - These can be used immediately for UQ training

2. RUBRIC-BASED BENCHMARKS (healthbench, tutorbench, prbench):
   - Require LLM judge (e.g., GPT-4) for scoring
   - Add ~{sum(s.num_examples for s in hard_stats if s.scoring_type == 'rubric'):,} examples after judging
   - Consider cost of judge API calls

3. EXECUTION-BASED BENCHMARKS (bigcodebench, livecodebench, swebench):
   - Require code execution environment
   - Add ~{sum(s.num_examples for s in hard_stats if s.scoring_type == 'execution'):,} examples after execution
   - Need Docker/sandbox setup

4. LONG-CONTEXT BENCHMARKS:
   - Higher cost per example
   - May need model with extended context
   - Consider running separately or sampling

5. BALANCE CONSIDERATIONS:
   - Current ratio favors incorrect samples (good for UQ)
   - Very hard benchmarks (arc_agi, hle) provide mostly incorrect samples
   - May want to oversample from easier benchmarks for balance
""")

    # Detailed table
    print("\n" + "=" * 80)
    print("DETAILED BENCHMARK TABLE")
    print("=" * 80)
    print()
    print(f"{'Benchmark':<18} {'Examples':>8} {'GPT5%':>6} {'Correct':>8} {'Incorrect':>10} {'AvgTok':>7} {'MaxTok':>8} {'Type':<10} {'Category':<12}")
    print("-" * 100)

    for s in sorted(hard_stats, key=lambda x: -x.num_examples):
        print(f"{s.name:<18} {s.num_examples:>8,} {s.gpt5_accuracy*100:>5.0f}% {s.estimated_correct:>8,} {s.estimated_incorrect:>10,} {s.avg_input_tokens:>7,} {s.max_input_tokens:>8,} {s.scoring_type:<10} {s.category:<12}")

    print("-" * 100)
    print(f"{'TOTAL':<18} {total_examples:>8,} {'':<6} {total_correct:>8,} {total_incorrect:>10,}")

    # Save to JSON
    output = {
        "summary": {
            "total_examples": total_examples,
            "estimated_correct": total_correct,
            "estimated_incorrect": total_incorrect,
            "balance_ratio": total_incorrect / total_correct if total_correct > 0 else 0,
            "estimated_cost_usd": total_cost,
            "cost_without_long_context_usd": nl_cost,
        },
        "by_scoring_type": {
            stype: {
                "examples": sum(s.num_examples for s in hard_stats if s.scoring_type == stype),
                "benchmarks": [s.name for s in hard_stats if s.scoring_type == stype],
            }
            for stype in ["auto", "execution", "rubric"]
        },
        "by_category": {
            cat: {
                "examples": data["examples"],
                "correct": data["correct"],
                "incorrect": data["incorrect"],
                "benchmarks": data["benchmarks"],
            }
            for cat, data in cat_stats.items()
        },
        "benchmarks": [
            {
                "name": s.name,
                "examples": s.num_examples,
                "gpt5_accuracy": s.gpt5_accuracy,
                "estimated_correct": s.estimated_correct,
                "estimated_incorrect": s.estimated_incorrect,
                "avg_input_tokens": s.avg_input_tokens,
                "max_input_tokens": s.max_input_tokens,
                "scoring_type": s.scoring_type,
                "category": s.category,
                "issues": s.issues,
            }
            for s in hard_stats
        ],
    }

    output_path = Path(__file__).parent.parent / "benchmark_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
