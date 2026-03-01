#!/usr/bin/env python3
"""
Analyze where the VLM judge model makes the worst errors.
Shows samples where model was MOST WRONG:
- High P(correct) but actually incorrect
- Low P(correct) but actually correct
"""
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

print("=" * 80)
print("ERROR ANALYSIS: WHERE THE MODEL IS MOST WRONG")
print("=" * 80)

# Load predictions (from selective prediction script)
pred_path = Path("data/vlm_judge_combined/text_predictions.json")

if not pred_path.exists():
    print("Need to run selective_prediction_curve.py first to generate predictions")
    sys.exit(1)

with open(pred_path) as f:
    data = json.load(f)

predictions = np.array(data["predictions"])
labels = np.array(data["labels"])
question_ids = data["question_ids"]
benchmarks = data["benchmarks"]
prompts = data["prompts"]
responses = data["responses"]

# Compute errors
# False positives: High P(correct) but actually incorrect
# False negatives: Low P(correct) but actually correct

errors = []
for i in range(len(predictions)):
    p_correct = predictions[i]
    is_correct = labels[i] > 0.5

    if is_correct and p_correct < 0.5:
        # False negative: model said incorrect, but was correct
        error_type = "FALSE_NEGATIVE"
        error_magnitude = 0.5 - p_correct  # How wrong
    elif not is_correct and p_correct > 0.5:
        # False positive: model said correct, but was incorrect
        error_type = "FALSE_POSITIVE"
        error_magnitude = p_correct - 0.5  # How wrong
    else:
        continue  # Correct prediction

    errors.append({
        "idx": i,
        "type": error_type,
        "magnitude": error_magnitude,
        "p_correct": p_correct,
        "actual": is_correct,
        "question_id": question_ids[i],
        "benchmark": benchmarks[i],
        "prompt": prompts[i],
        "response": responses[i],
    })

# Sort by magnitude (worst errors first)
errors.sort(key=lambda x: -x["magnitude"])

print(f"\nTotal errors: {len(errors)} / {len(predictions)} ({100*len(errors)/len(predictions):.1f}%)")
print(f"- False positives (said correct, was wrong): {sum(1 for e in errors if e['type'] == 'FALSE_POSITIVE')}")
print(f"- False negatives (said wrong, was correct): {sum(1 for e in errors if e['type'] == 'FALSE_NEGATIVE')}")

# Benchmark mapping
benchmark_names = {
    "161904": "BBEH",
    "172641": "Omnimath",
    "020834": "GPQA",
    "172545": "GPQA",
    "011529": "HellaSwag",
    "011259": "ARC",
    "011011": "MMLU",
    "001146": "GSM8K",
    "012344": "SimpleQA",
    "012008": "TriviaQA",
    "012104": "DROP",
    "005402": "Winogrande",
    "011752": "Winogrande",
    "022049": "MGSM",
    "001744": "HellaSwag",
    "011942": "BoolQ",
    "001859": "BoolQ",
}

def get_bench_name(bid):
    return benchmark_names.get(bid, bid)

print("\n" + "=" * 80)
print("TOP 10 WORST FALSE POSITIVES")
print("(Model was confident answer was CORRECT, but it was WRONG)")
print("=" * 80)

fp_errors = [e for e in errors if e["type"] == "FALSE_POSITIVE"]
for i, e in enumerate(fp_errors[:10]):
    print(f"\n--- Error {i+1}: P(correct)={e['p_correct']:.3f}, Actually INCORRECT ---")
    print(f"Benchmark: {get_bench_name(e['benchmark'])}")
    print(f"Question ID: {e['question_id']}")
    print(f"Prompt (truncated): {e['prompt'][:150]}...")
    print(f"Response (truncated): {e['response'][:150]}...")

print("\n" + "=" * 80)
print("TOP 10 WORST FALSE NEGATIVES")
print("(Model was confident answer was WRONG, but it was CORRECT)")
print("=" * 80)

fn_errors = [e for e in errors if e["type"] == "FALSE_NEGATIVE"]
for i, e in enumerate(fn_errors[:10]):
    print(f"\n--- Error {i+1}: P(correct)={e['p_correct']:.3f}, Actually CORRECT ---")
    print(f"Benchmark: {get_bench_name(e['benchmark'])}")
    print(f"Question ID: {e['question_id']}")
    print(f"Prompt (truncated): {e['prompt'][:150]}...")
    print(f"Response (truncated): {e['response'][:150]}...")

# Pattern analysis
print("\n" + "=" * 80)
print("ERROR PATTERN ANALYSIS")
print("=" * 80)

# By benchmark
print("\n--- Errors by Benchmark ---")
bench_errors = defaultdict(lambda: {"fp": 0, "fn": 0, "total": 0})
bench_totals = defaultdict(int)

for i in range(len(predictions)):
    bench_totals[benchmarks[i]] += 1

for e in errors:
    bench = e["benchmark"]
    if e["type"] == "FALSE_POSITIVE":
        bench_errors[bench]["fp"] += 1
    else:
        bench_errors[bench]["fn"] += 1
    bench_errors[bench]["total"] += 1

print(f"{'Benchmark':<15} | {'Total':>6} | {'Errors':>6} | {'Error%':>7} | {'FP':>4} | {'FN':>4}")
print("-" * 60)
for bench in sorted(bench_errors.keys(), key=lambda x: -bench_errors[x]["total"]):
    be = bench_errors[bench]
    total = bench_totals[bench]
    err_pct = 100 * be["total"] / total if total > 0 else 0
    print(f"{get_bench_name(bench):<15} | {total:>6} | {be['total']:>6} | {err_pct:>6.1f}% | {be['fp']:>4} | {be['fn']:>4}")

# By response length
print("\n--- Errors by Response Length ---")
short_errors = [e for e in errors if len(e["response"]) < 50]
medium_errors = [e for e in errors if 50 <= len(e["response"]) < 150]
long_errors = [e for e in errors if len(e["response"]) >= 150]

short_total = sum(1 for i in range(len(responses)) if len(responses[i]) < 50)
medium_total = sum(1 for i in range(len(responses)) if 50 <= len(responses[i]) < 150)
long_total = sum(1 for i in range(len(responses)) if len(responses[i]) >= 150)

print(f"Short (<50 chars):  {len(short_errors):>3} errors / {short_total:>3} samples ({100*len(short_errors)/short_total if short_total > 0 else 0:.1f}%)")
print(f"Medium (50-150):    {len(medium_errors):>3} errors / {medium_total:>3} samples ({100*len(medium_errors)/medium_total if medium_total > 0 else 0:.1f}%)")
print(f"Long (>150 chars):  {len(long_errors):>3} errors / {long_total:>3} samples ({100*len(long_errors)/long_total if long_total > 0 else 0:.1f}%)")

# By prompt length
print("\n--- Errors by Prompt Length ---")
short_p = [e for e in errors if len(e["prompt"]) < 100]
long_p = [e for e in errors if len(e["prompt"]) >= 100]

short_p_total = sum(1 for i in range(len(prompts)) if len(prompts[i]) < 100)
long_p_total = sum(1 for i in range(len(prompts)) if len(prompts[i]) >= 100)

print(f"Short (<100 chars): {len(short_p):>3} errors / {short_p_total:>3} samples ({100*len(short_p)/short_p_total if short_p_total > 0 else 0:.1f}%)")
print(f"Long (≥100 chars):  {len(long_p):>3} errors / {long_p_total:>3} samples ({100*len(long_p)/long_p_total if long_p_total > 0 else 0:.1f}%)")

# Confidence distribution of errors
print("\n--- Confidence Distribution of Errors ---")
confidences = np.maximum(predictions, 1 - predictions)
error_confidences = [confidences[e["idx"]] for e in errors]
correct_confidences = [confidences[i] for i in range(len(predictions))
                       if ((predictions[i] > 0.5) == (labels[i] > 0.5))]

print(f"Mean confidence on ERRORS:  {np.mean(error_confidences):.3f}")
print(f"Mean confidence on CORRECT: {np.mean(correct_confidences):.3f}")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

# Find patterns
max_err_bench = max(bench_errors.keys(), key=lambda x: bench_errors[x]["total"]/bench_totals[x] if bench_totals[x] > 10 else 0)
print(f"\n1. Highest error rate benchmark: {get_bench_name(max_err_bench)}")

if len(long_errors)/long_total > len(short_errors)/short_total:
    print("2. Model struggles MORE with long responses")
else:
    print("2. Model struggles MORE with short responses")

if np.mean(error_confidences) > 0.7:
    print("3. ⚠️ Model is overconfident on errors (mean conf {:.3f})".format(np.mean(error_confidences)))
else:
    print("3. ✓ Model shows appropriate uncertainty on errors")

# Save analysis
analysis = {
    "total_samples": len(predictions),
    "total_errors": len(errors),
    "false_positives": len(fp_errors),
    "false_negatives": len(fn_errors),
    "errors_by_benchmark": {get_bench_name(k): v for k, v in bench_errors.items()},
    "mean_error_confidence": float(np.mean(error_confidences)),
    "mean_correct_confidence": float(np.mean(correct_confidences)),
}

with open("data/vlm_judge_combined/error_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)
print(f"\nAnalysis saved to data/vlm_judge_combined/error_analysis.json")
