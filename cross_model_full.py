#!/usr/bin/env python3
"""
Full cross-model evaluation using existing Qwen runs.

This script:
1. Loads Qwen predictions from existing runs
2. Scores them for correctness using the target field
3. Finds matching Llama predictions for the same questions
4. Runs UQ scoring on both and compares
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np


def load_predictions(run_path: Path) -> list[dict]:
    """Load predictions from a run directory."""
    pred_file = run_path / "predictions.jsonl"
    if not pred_file.exists():
        return []

    examples = []
    with open(pred_file) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                examples.append(row)
    return examples


def extract_answer(response: str, benchmark: str) -> str:
    """Extract the final answer from a model response."""
    response = response.strip()

    # For math benchmarks, try to extract boxed answer
    if "\\boxed{" in response:
        match = re.search(r'\\boxed\{([^}]+)\}', response)
        if match:
            return match.group(1).strip()

    # For MCQ, look for letter answer
    if benchmark in ["arc", "mmlu", "gpqa", "hellaswag", "winogrande"]:
        # Look for patterns like "The answer is A" or just "A" at the end
        match = re.search(r'(?:answer is|Answer:|answer:)\s*([A-E])', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # Check last line for single letter
        last_line = response.split('\n')[-1].strip()
        if len(last_line) == 1 and last_line.upper() in 'ABCDE':
            return last_line.upper()

    # For numeric answers (gsm8k, etc), extract last number
    if benchmark in ["gsm8k", "mgsm"]:
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            return numbers[-1]

    # Default: return the response (or last line if long)
    if len(response) > 100:
        return response.split('\n')[-1].strip()[:100]
    return response


def score_prediction(pred: dict, benchmark: str) -> bool:
    """Score a prediction for correctness."""
    target = str(pred.get("target", "")).strip()

    # Get the model's response
    response = ""
    if "prediction" in pred and isinstance(pred["prediction"], dict):
        response = pred["prediction"].get("answer", "")
    elif "response" in pred:
        response = pred["response"]

    if not response or not target:
        return None

    # Extract answer from response
    answer = extract_answer(response, benchmark)

    # Normalize for comparison
    target_norm = target.lower().strip()
    answer_norm = answer.lower().strip()

    # Exact match
    if target_norm == answer_norm:
        return True

    # Numeric comparison for math
    try:
        if float(target_norm) == float(answer_norm):
            return True
    except (ValueError, TypeError):
        pass

    # Containment check for short answers
    if len(target_norm) < 20:
        if target_norm in answer_norm or answer_norm in target_norm:
            return True

    return False


def get_uq_confidences(examples: list[dict], model, tokenizer, temperature: float = 1.0) -> list[float]:
    """Get UQ confidence scores using official implementation."""
    from uq_eval.uq_finetune import get_confidence_batch
    return get_confidence_batch(model, tokenizer, examples, temperature=temperature)


def compute_metrics(confidences: list[float], correct: list[bool]) -> dict:
    """Compute AUROC, ECE, Brier score."""
    if len(confidences) == 0:
        return {"auroc": 0.5, "ece": 0.0, "brier": 0.25, "accuracy": 0.5}

    predictions = [c > 0.5 for c in confidences]
    accuracy = sum(p == c for p, c in zip(predictions, correct)) / len(correct)
    brier = sum((c - int(corr)) ** 2 for c, corr in zip(confidences, correct)) / len(correct)

    n_bins = 10
    bin_correct = [0] * n_bins
    bin_conf = [0.0] * n_bins
    bin_count = [0] * n_bins

    for conf, corr in zip(confidences, correct):
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bin_correct[bin_idx] += int(corr)
        bin_conf[bin_idx] += conf
        bin_count[bin_idx] += 1

    ece = 0.0
    for i in range(n_bins):
        if bin_count[i] > 0:
            avg_conf = bin_conf[i] / bin_count[i]
            avg_acc = bin_correct[i] / bin_count[i]
            ece += bin_count[i] * abs(avg_conf - avg_acc)
    ece /= len(correct)

    try:
        auroc = roc_auc_score(correct, confidences)
    except ValueError:
        auroc = 0.5

    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "ece": ece,
        "brier": brier,
        "mean_confidence": sum(confidences) / len(confidences),
    }


def plot_cross_model_comparison(llama_metrics: dict, qwen_metrics: dict, output_path: str):
    """Generate cross-model comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    methods = ["Llama-3.1-8B\n(in-distribution)", "Qwen-2.5-7B\n(cross-model)"]
    colors = ["#2ecc71", "#e74c3c"]

    # AUROC
    ax = axes[0]
    values = [llama_metrics["auroc"], qwen_metrics["auroc"]]
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("AUROC (↑)", fontsize=12)
    ax.set_title("Discrimination", fontsize=12)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # ECE
    ax = axes[1]
    values = [llama_metrics["ece"], qwen_metrics["ece"]]
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("ECE (↓)", fontsize=12)
    ax.set_title("Calibration Error", fontsize=12)
    ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Brier Score
    ax = axes[2]
    values = [llama_metrics["brier"], qwen_metrics["brier"]]
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Brier Score (↓)", fontsize=12)
    ax.set_title("Overall Quality", fontsize=12)
    ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--uq_model_path", default="uq_models/llama-8b-uq-lora-v2")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max_per_benchmark", type=int, default=200)
    args = parser.parse_args()

    print("=" * 50)
    print("CROSS-MODEL EVALUATION (Full)")
    print("=" * 50)

    runs_dir = Path(args.runs_dir)

    # Find Qwen runs
    qwen_runs = list(runs_dir.glob("*Qwen*"))
    print(f"\nFound {len(qwen_runs)} Qwen runs")

    # Find matching Llama runs
    llama_runs = list(runs_dir.glob("*Llama*")) + list(runs_dir.glob("*llama*"))
    print(f"Found {len(llama_runs)} Llama runs")

    # Load and score Qwen predictions
    qwen_examples = []
    for run in qwen_runs:
        # Extract benchmark name from run directory
        parts = run.name.split("_")
        if len(parts) >= 3:
            benchmark = parts[2]
        else:
            continue

        preds = load_predictions(run)
        print(f"  {benchmark}: {len(preds)} predictions")

        for pred in preds[:args.max_per_benchmark]:
            # Get question
            question = pred.get("input", "")
            if isinstance(question, dict):
                question = question.get("question", question.get("text", str(question)))

            # Get full response (use response_text like Llama's model_response)
            response = pred.get("response_text", "")
            if not response and "prediction" in pred and isinstance(pred["prediction"], dict):
                response = pred["prediction"].get("answer", "")
            if not response:
                response = pred.get("response", "")

            # Score
            correct = score_prediction(pred, benchmark)
            if correct is not None and question and response:
                qwen_examples.append({
                    "id": pred.get("id", ""),
                    "benchmark": benchmark,
                    "question": str(question),
                    "answer": str(response),
                    "correct": correct,
                })

    print(f"\nLoaded {len(qwen_examples)} scored Qwen examples")
    if qwen_examples:
        correct_count = sum(1 for ex in qwen_examples if ex["correct"])
        print(f"  Correct: {correct_count}/{len(qwen_examples)} ({100*correct_count/len(qwen_examples):.1f}%)")

    # Load Llama examples using EXACT same logic as evaluate_and_plot.py
    llama_examples = []
    test_path = Path("data/finetune/test_v2.jsonl")
    if test_path.exists():
        with open(test_path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    ex_id = row.get("id", "")
                    bench = ex_id.split("_")[0] if "_" in ex_id else "unknown"
                    # EXACT same logic as evaluate_and_plot.py line 48-49
                    llama_examples.append({
                        "id": ex_id,
                        "benchmark": bench,
                        "question": row.get("input", ""),
                        "answer": row.get("model_response", row.get("prediction", {}).get("answer", "")),
                        "correct": row.get("correct", 0) == 1,
                    })

    print(f"Loaded {len(llama_examples)} Llama examples from test set")

    if not qwen_examples or not llama_examples:
        print("ERROR: Not enough examples to compare")
        return

    # Load calibration temperature
    calib_path = Path(args.uq_model_path) / "calibration.json"
    if calib_path.exists():
        with open(calib_path) as f:
            temperature = json.load(f)["temperature"]
        print(f"\nUsing calibration temperature: {temperature:.4f}")
    else:
        temperature = 1.0

    # Load UQ model (use exact same loading as evaluate_and_plot.py)
    print(f"\nLoading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading finetuned model from {args.uq_model_path}...")
    model = PeftModel.from_pretrained(base_model, args.uq_model_path)
    # Load tokenizer from finetuned model path (like evaluate_and_plot.py)
    tokenizer = AutoTokenizer.from_pretrained(args.uq_model_path)
    model.eval()

    # Get confidences for Llama examples
    print(f"\nScoring {len(llama_examples)} Llama examples...")
    llama_confidences = get_uq_confidences(llama_examples, model, tokenizer, temperature)
    llama_correct = [ex["correct"] for ex in llama_examples]

    # Get confidences for Qwen examples
    print(f"Scoring {len(qwen_examples)} Qwen examples...")
    qwen_confidences = get_uq_confidences(qwen_examples, model, tokenizer, temperature)
    qwen_correct = [ex["correct"] for ex in qwen_examples]

    # Compute metrics
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    llama_metrics = compute_metrics(llama_confidences, llama_correct)
    print(f"\nLlama-3.1-8B (in-distribution, test set):")
    print(f"  N = {len(llama_examples)}")
    print(f"  Accuracy: {llama_metrics['accuracy']:.3f}")
    print(f"  AUROC: {llama_metrics['auroc']:.3f}")
    print(f"  ECE:   {llama_metrics['ece']:.3f}")
    print(f"  Brier: {llama_metrics['brier']:.3f}")

    qwen_metrics = compute_metrics(qwen_confidences, qwen_correct)
    print(f"\nQwen-2.5-7B (cross-model transfer):")
    print(f"  N = {len(qwen_examples)}")
    print(f"  Accuracy: {qwen_metrics['accuracy']:.3f}")
    print(f"  AUROC: {qwen_metrics['auroc']:.3f}")
    print(f"  ECE:   {qwen_metrics['ece']:.3f}")
    print(f"  Brier: {qwen_metrics['brier']:.3f}")

    # Generate figure
    plot_cross_model_comparison(llama_metrics, qwen_metrics, "figures/fig_cross_model.pdf")

    # Save results
    results = {
        "llama": llama_metrics,
        "qwen": qwen_metrics,
        "n_llama": len(llama_examples),
        "n_qwen": len(qwen_examples),
    }
    with open("results/cross_model_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/cross_model_metrics.json")

    # Per-benchmark breakdown for Qwen
    print("\n" + "=" * 50)
    print("PER-BENCHMARK BREAKDOWN (Qwen)")
    print("=" * 50)

    bench_data = defaultdict(lambda: {"confidences": [], "correct": []})
    for ex, conf in zip(qwen_examples, qwen_confidences):
        bench_data[ex["benchmark"]]["confidences"].append(conf)
        bench_data[ex["benchmark"]]["correct"].append(ex["correct"])

    print(f"\n{'Benchmark':<15} {'N':>6} {'Acc':>8} {'AUROC':>8}")
    print("-" * 40)
    for bench, data in sorted(bench_data.items(), key=lambda x: -len(x[1]["confidences"])):
        if len(data["confidences"]) >= 5:
            metrics = compute_metrics(data["confidences"], data["correct"])
            acc = sum(data["correct"]) / len(data["correct"])
            print(f"{bench:<15} {len(data['confidences']):>6} {acc:>8.3f} {metrics['auroc']:>8.3f}")

    print("\n" + "=" * 50)
    print("CROSS-MODEL EVALUATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
