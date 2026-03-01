#!/usr/bin/env python3
"""
Demonstrate practical use cases for cross-model UQ.

Use cases:
  1. Selective prediction (abstain when uncertain → boost accuracy)
  2. Automatic routing (easy → cheap model, hard → expensive model)
  3. Hallucination detection (flag low-confidence factual claims)
  4. Exam/benchmark difficulty estimation (which questions are hardest?)
  5. Active learning proxy (which samples to label next?)
  6. [Longshot] Reward model for RLHF (confidence as reward signal)
  7. [Longshot] Cascade inference (lightweight judge gates expensive retry)

Usage:
    python scripts/use_case_demos.py
"""
import json
import os
from pathlib import Path


def load_predictions_with_scores(model_prefix="qwen35_397b"):
    """Load predictions with confidence and correctness."""
    samples = []
    for d in sorted(os.listdir("runs")):
        if not d.startswith(f"{model_prefix}_") or "backup" in d:
            continue
        bench = d.replace(f"{model_prefix}_", "")
        p = f"runs/{d}/predictions.jsonl"
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                s = json.loads(line)
                conf = s.get("prediction", {}).get("confidence")
                correct = s.get("score", {}).get("correct")
                if conf is not None and correct is not None:
                    samples.append({
                        "bench": bench,
                        "confidence": float(conf),
                        "correct": 1 if correct else 0,
                        "id": s.get("id", ""),
                    })
    return samples


def load_calibrator_scores(path):
    """Load calibrator P(correct) scores from cross-model eval."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def use_case_1_selective_prediction(samples, name="Verbalized"):
    """
    USE CASE 1: Selective Prediction
    "Answer only when confident" — trade coverage for accuracy.

    Real-world scenario: Medical QA, legal advice, financial analysis.
    The model should say "I don't know" rather than give a wrong answer.
    """
    print("=" * 70)
    print("USE CASE 1: Selective Prediction (Abstain When Uncertain)")
    print("=" * 70)
    print(f"Method: {name}")
    print(f"Scenario: Model only answers when confidence > threshold.")
    print(f"         Remaining questions escalated to human expert.")
    print()

    # Sort by confidence descending
    sorted_samples = sorted(samples, key=lambda s: s["confidence"], reverse=True)
    n = len(sorted_samples)

    print(f"{'Threshold':>10} {'Coverage':>10} {'Accuracy':>10} {'Answered':>10} {'Correct':>10}")
    print("-" * 55)

    results = []
    for threshold in [0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.0]:
        answered = [s for s in sorted_samples if s["confidence"] >= threshold]
        if not answered:
            continue
        coverage = len(answered) / n
        accuracy = sum(s["correct"] for s in answered) / len(answered)
        results.append({
            "threshold": threshold,
            "coverage": coverage,
            "accuracy": accuracy,
            "n_answered": len(answered),
        })
        print(f"{threshold:>10.2f} {coverage:>10.1%} {accuracy:>10.1%} {len(answered):>10} {sum(s['correct'] for s in answered):>10}")

    # Key finding
    base_acc = sum(s["correct"] for s in samples) / len(samples)
    print(f"\nBaseline (answer everything): {base_acc:.1%} accuracy on {n} samples")

    # Find threshold for 90% accuracy
    for r in results:
        if r["accuracy"] >= 0.90:
            print(f"→ At {r['threshold']} threshold: {r['accuracy']:.1%} accuracy, {r['coverage']:.1%} coverage")
            break

    return results


def use_case_2_model_routing(samples):
    """
    USE CASE 2: Automatic Model Routing
    "Easy questions → cheap model, hard questions → expensive model"

    Real-world scenario: API cost optimization. Use GPT-5-mini for easy queries,
    escalate to GPT-5.2 only for hard ones. UQ judge decides which.
    """
    print("\n" + "=" * 70)
    print("USE CASE 2: Automatic Model Routing (Cost Optimization)")
    print("=" * 70)
    print("Scenario: Route easy queries to cheap model, hard to expensive.")
    print("          UQ judge on cheap model decides when to escalate.")
    print()

    # Simulate: high confidence = stay with cheap model, low = escalate
    # Assume cheap model costs $0.001/query, expensive costs $0.01/query
    cheap_cost = 0.001
    expensive_cost = 0.01
    n = len(samples)

    sorted_samples = sorted(samples, key=lambda s: s["confidence"], reverse=True)

    print(f"{'Strategy':>25} {'Accuracy':>10} {'Cost/1K':>10} {'Savings':>10}")
    print("-" * 60)

    # All cheap
    base_acc = sum(s["correct"] for s in samples) / n
    all_cheap = cheap_cost * 1000
    print(f"{'All cheap model':>25} {base_acc:>10.1%} ${all_cheap:>9.2f} {'baseline':>10}")

    # All expensive (assume 10% better accuracy)
    all_expensive = expensive_cost * 1000
    print(f"{'All expensive model':>25} {'~+10%':>10} ${all_expensive:>9.2f} {'none':>10}")

    # Routing: top 70% by confidence stay cheap, bottom 30% go expensive
    for route_pct in [0.5, 0.7, 0.8, 0.9]:
        n_cheap = int(n * route_pct)
        cheap_samples = sorted_samples[:n_cheap]
        expensive_samples = sorted_samples[n_cheap:]

        cheap_acc = sum(s["correct"] for s in cheap_samples) / len(cheap_samples) if cheap_samples else 0
        # The expensive model would do better on the hard ones — estimate
        cost = (len(cheap_samples) * cheap_cost + len(expensive_samples) * expensive_cost) / n * 1000
        savings = (1 - cost / all_expensive) * 100

        print(f"{'Route ' + f'{route_pct:.0%} cheap':>25} {cheap_acc:>10.1%}* ${cost:>9.2f} {savings:>9.0f}%")

    print("\n* Accuracy shown for cheap-model portion only. Expensive model handles the rest.")
    print("→ Key insight: UQ judge enables 50-90% cost savings by keeping easy queries cheap.")


def use_case_3_hallucination_flagging(samples):
    """
    USE CASE 3: Hallucination Detection
    Flag responses where the model is likely wrong.

    Real-world scenario: Content moderation, fact-checking pipeline,
    RAG systems flagging unreliable generations.
    """
    print("\n" + "=" * 70)
    print("USE CASE 3: Hallucination / Error Flagging")
    print("=" * 70)
    print("Scenario: Flag model responses likely to be wrong.")
    print("          Flagged responses get human review before publication.")
    print()

    # Focus on factual benchmarks
    factual_benches = {"simpleqa", "gpqa", "chembench", "hallusionbench", "hle"}
    factual = [s for s in samples if s["bench"] in factual_benches]

    if not factual:
        print("No factual benchmark samples available.")
        return

    n = len(factual)
    n_wrong = sum(1 - s["correct"] for s in factual)
    base_error_rate = n_wrong / n

    print(f"Factual benchmarks: {', '.join(factual_benches)}")
    print(f"Total: {n} samples, {n_wrong} wrong ({base_error_rate:.1%} error rate)")
    print()

    # Flag bottom K% by confidence
    sorted_factual = sorted(factual, key=lambda s: s["confidence"])

    print(f"{'Flag bottom':>12} {'Flagged':>8} {'Errors caught':>14} {'Precision':>10} {'Recall':>8}")
    print("-" * 55)

    for flag_pct in [0.10, 0.20, 0.30, 0.50]:
        n_flag = int(n * flag_pct)
        flagged = sorted_factual[:n_flag]
        errors_caught = sum(1 - s["correct"] for s in flagged)
        precision = errors_caught / len(flagged) if flagged else 0
        recall = errors_caught / n_wrong if n_wrong else 0
        print(f"{flag_pct:>11.0%} {n_flag:>8} {errors_caught:>14} {precision:>10.1%} {recall:>8.1%}")

    print("\n→ Key insight: Flagging bottom 20% by confidence catches disproportionate errors.")


def use_case_4_difficulty_estimation(samples):
    """
    USE CASE 4: Question Difficulty Estimation
    Estimate how hard questions are based on model confidence patterns.

    Real-world scenario: Adaptive testing, curriculum design,
    benchmark analysis.
    """
    print("\n" + "=" * 70)
    print("USE CASE 4: Benchmark Difficulty Estimation")
    print("=" * 70)

    bench_stats = {}
    for s in samples:
        b = s["bench"]
        if b not in bench_stats:
            bench_stats[b] = {"confs": [], "corrects": []}
        bench_stats[b]["confs"].append(s["confidence"])
        bench_stats[b]["corrects"].append(s["correct"])

    print(f"{'Benchmark':<20} {'Accuracy':>8} {'Mean Conf':>10} {'Overconf':>10} {'N':>5}")
    print("-" * 58)

    rows = []
    for bench in sorted(bench_stats.keys()):
        confs = bench_stats[bench]["confs"]
        corrects = bench_stats[bench]["corrects"]
        acc = sum(corrects) / len(corrects)
        mean_conf = sum(confs) / len(confs)
        overconf = mean_conf - acc  # positive = overconfident
        rows.append((bench, acc, mean_conf, overconf, len(confs)))

    for bench, acc, mc, oc, n in sorted(rows, key=lambda r: r[3], reverse=True):
        print(f"{bench:<20} {acc:>8.1%} {mc:>10.2f} {oc:>+10.2f} {n:>5}")

    print("\n→ Overconfidence = mean_confidence - accuracy. Higher = model doesn't know what it doesn't know.")


def use_case_5_longshot_reward_model():
    """
    USE CASE 5 [LONGSHOT]: UQ as Reward Signal for RLHF

    Instead of training a separate reward model, use the UQ judge's
    P(correct) as a reward signal. Lower confidence → lower reward →
    model learns to be more careful on hard questions.
    """
    print("\n" + "=" * 70)
    print("[LONGSHOT] USE CASE 5: UQ Judge as Reward Model for RLHF")
    print("=" * 70)
    print("""
Idea: Use the UQ judge's P(correct) score as a reward signal for RLHF/DPO.

How it works:
  1. Generate N responses to each question
  2. Score each with UQ judge → P(correct)
  3. Use as reward: high P(correct) = preferred, low = rejected
  4. Train with DPO/PPO

Why it's interesting:
  - No human annotation needed (fully automated)
  - Cross-model: train reward on Model A, use to improve Model B
  - Scales to any domain without domain-specific reward models
  - Could make models "know what they don't know" (calibrated abstention)

What we'd need to test:
  - Generate 5-10 responses per question from Qwen3.5
  - Score each with our calibrator
  - Check if P(correct) ranking correlates with actual correctness ranking
  - If yes → viable reward signal
""")


def use_case_6_longshot_cascade():
    """
    USE CASE 6 [LONGSHOT]: Cascade Inference with UQ Gating

    Chain of models: fast→medium→slow, with UQ judge deciding
    when to stop or escalate.
    """
    print("\n" + "=" * 70)
    print("[LONGSHOT] USE CASE 6: Cascade Inference with UQ Gating")
    print("=" * 70)
    print("""
Idea: Multi-stage inference where UQ judge gates escalation.

Pipeline:
  Stage 1: GPT-5-mini answers → UQ judge scores
    If P(correct) > 0.8 → return answer (cheap, fast)
    Else → escalate to Stage 2

  Stage 2: GPT-5.2 answers → UQ judge scores
    If P(correct) > 0.7 → return answer
    Else → escalate to Stage 3

  Stage 3: Human expert reviews

Why it's interesting:
  - Same UQ judge works across all stages (cross-model transfer!)
  - Optimizes cost-accuracy tradeoff automatically
  - No retraining needed when swapping models in/out of cascade

What we'd need:
  - Responses from 2-3 models at different cost tiers (we have this!)
  - Simulate the cascade: how often does Stage 1 suffice?
  - Measure: total cost vs accuracy vs all-expensive baseline
""")


def use_case_7_longshot_self_improvement():
    """
    USE CASE 7 [LONGSHOT]: Targeted Self-Improvement

    Use UQ to identify systematic weaknesses, then generate
    targeted training data for those weak areas.
    """
    print("\n" + "=" * 70)
    print("[LONGSHOT] USE CASE 7: UQ-Guided Self-Improvement")
    print("=" * 70)
    print("""
Idea: Use UQ judge to find where models systematically fail,
then generate targeted training data for those failure modes.

Pipeline:
  1. Run UQ judge across diverse benchmarks
  2. Identify clusters of low-confidence, incorrect responses
  3. Analyze: what topics/question types fail most?
  4. Generate synthetic training data targeting those weak areas
  5. Fine-tune model on targeted data
  6. Re-evaluate with UQ judge → measure improvement

Why it's interesting:
  - Automated curriculum learning
  - UQ judge identifies "unknown unknowns" (overconfident errors)
  - Works for closed-source models: identify weaknesses, then
    pressure vendor to improve specific capabilities
  - Data-efficient: only train on what's actually broken
""")


def main():
    print("UNCERTAINTY QUANTIFICATION: USE CASE DEMONSTRATIONS")
    print("=" * 70)
    print()

    samples = load_predictions_with_scores()
    print(f"Loaded {len(samples)} predictions from Qwen3.5-397B")
    print()

    # Practical use cases with data
    results_1 = use_case_1_selective_prediction(samples)
    use_case_2_model_routing(samples)
    use_case_3_hallucination_flagging(samples)
    use_case_4_difficulty_estimation(samples)

    # Longshots (conceptual)
    use_case_5_longshot_reward_model()
    use_case_6_longshot_cascade()
    use_case_7_longshot_self_improvement()

    # Save results
    output = {
        "selective_prediction": results_1,
        "n_samples": len(samples),
        "model": "Qwen3.5-397B-A17B-FP8",
    }
    os.makedirs("data/use_cases", exist_ok=True)
    with open("data/use_cases/demo_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print("Results saved to data/use_cases/demo_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
