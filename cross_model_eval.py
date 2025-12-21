#!/usr/bin/env python3
"""
Cross-model evaluation: Test if UQ model trained on Llama generalizes to Qwen.

This script:
1. Loads the test set questions
2. Runs Qwen-2.5-7B-Instruct on the same questions
3. Scores both Llama and Qwen answers with the finetuned UQ model
4. Compares AUROC/ECE metrics
"""

import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np


def load_test_data(path: str) -> list[dict]:
    """Load test examples with questions and Llama answers."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                ex_id = row.get("id", "")
                bench = ex_id.split("_")[0] if "_" in ex_id else "unknown"
                examples.append({
                    "id": ex_id,
                    "benchmark": bench,
                    "question": row.get("input", ""),
                    "llama_answer": row.get("model_response", row.get("prediction", {}).get("answer", "")),
                    "llama_correct": row.get("correct", 0) == 1,
                    "ground_truth": row.get("ground_truth", row.get("expected", "")),
                })
    return examples


def run_qwen_inference(examples: list[dict], model, tokenizer, max_new_tokens: int = 256) -> list[dict]:
    """Run Qwen on the test questions and get answers."""
    print(f"Running Qwen inference on {len(examples)} examples...")

    for i, ex in enumerate(examples):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(examples)}")

        # Ensure question is a string
        question = ex["question"]
        if isinstance(question, dict):
            # Handle dict format (e.g., from some benchmarks)
            question = question.get("text", question.get("content", str(question)))
        question = str(question)

        # Format prompt for Qwen using simple format (avoid chat template issues)
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        ex["qwen_answer"] = response.strip()

    return examples


def score_qwen_correctness(examples: list[dict], benchmark_scorers: dict = None) -> list[dict]:
    """Score Qwen answers for correctness using simple heuristics."""
    # Simple scoring: check if ground truth appears in answer
    for ex in examples:
        gt = str(ex.get("ground_truth", "")).strip().lower()
        answer = ex.get("qwen_answer", "").strip().lower()

        # Simple containment check (works for many benchmarks)
        if gt and len(gt) > 0:
            # For short answers, check exact match or containment
            if len(gt) < 50:
                ex["qwen_correct"] = gt in answer or answer in gt
            else:
                # For longer answers, check key overlap
                gt_words = set(gt.split())
                answer_words = set(answer.split())
                overlap = len(gt_words & answer_words) / max(len(gt_words), 1)
                ex["qwen_correct"] = overlap > 0.5
        else:
            # No ground truth available, mark as unknown
            ex["qwen_correct"] = None

    # Filter out examples without ground truth
    valid = [ex for ex in examples if ex.get("qwen_correct") is not None]
    print(f"Scored {len(valid)}/{len(examples)} Qwen answers (others lack ground truth)")

    return examples


def format_uq_prompt(question: str, answer: str) -> str:
    """Format a question/answer pair for UQ inference."""
    return (
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Is the answer correct? (i) No (ii) Yes\n\n"
    )


def get_uq_confidences(examples: list[dict], model, tokenizer, answer_key: str, temperature: float = 1.0) -> list[float]:
    """Get UQ confidence scores for answers."""
    # Get token IDs for "i" and "ii"
    token_i = tokenizer.encode("i", add_special_tokens=False)[0]
    token_ii = tokenizer.encode("ii", add_special_tokens=False)[0]

    confidences = []
    batch_size = 4

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        prompts = [format_uq_prompt(ex["question"], ex[answer_key]) for ex in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

            for j in range(len(batch)):
                attention_mask = inputs["attention_mask"][j]
                last_pos = attention_mask.sum().item() - 1

                logits = outputs.logits[j, last_pos, :]
                logit_i = logits[token_i].item()
                logit_ii = logits[token_ii].item()

                # Apply temperature and softmax
                logits_pair = torch.tensor([logit_i, logit_ii]) / temperature
                probs = torch.softmax(logits_pair, dim=0)
                confidences.append(probs[1].item())

    return confidences


def compute_metrics(confidences: list[float], correct: list[bool]) -> dict:
    """Compute AUROC, ECE, Brier score."""
    predictions = [c > 0.5 for c in confidences]

    # Accuracy
    accuracy = sum(p == c for p, c in zip(predictions, correct)) / len(correct)

    # Brier score
    brier = sum((c - int(corr)) ** 2 for c, corr in zip(confidences, correct)) / len(correct)

    # ECE
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

    # AUROC
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

    methods = ["Llama-3.1-8B\n(training dist.)", "Qwen-2.5-7B\n(cross-model)"]
    colors = ["#2ecc71", "#e74c3c"]

    # AUROC
    ax = axes[0]
    values = [llama_metrics["auroc"], qwen_metrics["auroc"]]
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("AUROC (↑)", fontsize=12)
    ax.set_title("Discrimination", fontsize=12)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # ECE
    ax = axes[1]
    values = [llama_metrics["ece"], qwen_metrics["ece"]]
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("ECE (↓)", fontsize=12)
    ax.set_title("Calibration Error", fontsize=12)
    ax.set_ylim(0, max(values) * 1.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Brier Score
    ax = axes[2]
    values = [llama_metrics["brier"], qwen_metrics["brier"]]
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Brier Score (↓)", fontsize=12)
    ax.set_title("Overall Quality", fontsize=12)
    ax.set_ylim(0, max(values) * 1.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="data/finetune/test_v2.jsonl")
    parser.add_argument("--uq_model_path", default="uq_models/llama-8b-uq-lora-v2")
    parser.add_argument("--qwen_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    print("=" * 50)
    print("CROSS-MODEL EVALUATION")
    print("=" * 50)

    # Load test data
    print(f"\nLoading test data from {args.test_path}...")
    examples = load_test_data(args.test_path)
    if args.max_examples:
        examples = examples[:args.max_examples]
    print(f"Loaded {len(examples)} examples")

    # Load calibration temperature
    calib_path = Path(args.uq_model_path) / "calibration.json"
    if calib_path.exists():
        with open(calib_path) as f:
            temperature = json.load(f)["temperature"]
        print(f"Using calibration temperature: {temperature:.4f}")
    else:
        temperature = 1.0
        print("No calibration found, using T=1.0")

    # Load Qwen model
    print(f"\nLoading Qwen model: {args.qwen_model}")
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, trust_remote_code=True)
    qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    qwen_model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # Use single GPU
        trust_remote_code=True,
    )
    qwen_model.eval()

    # Run Qwen inference
    examples = run_qwen_inference(examples, qwen_model, qwen_tokenizer)

    # Score Qwen answers
    examples = score_qwen_correctness(examples)

    # Free Qwen model memory
    del qwen_model
    del qwen_tokenizer
    torch.cuda.empty_cache()

    # Load UQ model
    print(f"\nLoading UQ model from {args.uq_model_path}...")
    uq_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    uq_tokenizer.pad_token = uq_tokenizer.eos_token
    uq_tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    uq_model = PeftModel.from_pretrained(base_model, args.uq_model_path)
    uq_model.eval()

    # Get confidences for Llama answers
    print("\nScoring Llama answers with UQ model...")
    llama_confidences = get_uq_confidences(examples, uq_model, uq_tokenizer, "llama_answer", temperature)
    llama_correct = [ex["llama_correct"] for ex in examples]

    # Get confidences for Qwen answers (only where we have correctness labels)
    valid_examples = [ex for ex in examples if ex.get("qwen_correct") is not None]
    print(f"\nScoring {len(valid_examples)} Qwen answers with UQ model...")
    qwen_confidences = get_uq_confidences(valid_examples, uq_model, uq_tokenizer, "qwen_answer", temperature)
    qwen_correct = [ex["qwen_correct"] for ex in valid_examples]

    # Compute metrics
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    llama_metrics = compute_metrics(llama_confidences, llama_correct)
    print(f"\nLlama-3.1-8B (training distribution):")
    print(f"  N = {len(examples)}")
    print(f"  AUROC: {llama_metrics['auroc']:.3f}")
    print(f"  ECE:   {llama_metrics['ece']:.3f}")
    print(f"  Brier: {llama_metrics['brier']:.3f}")

    if len(qwen_correct) > 0:
        qwen_metrics = compute_metrics(qwen_confidences, qwen_correct)
        print(f"\nQwen-2.5-7B (cross-model generalization):")
        print(f"  N = {len(valid_examples)}")
        print(f"  AUROC: {qwen_metrics['auroc']:.3f}")
        print(f"  ECE:   {qwen_metrics['ece']:.3f}")
        print(f"  Brier: {qwen_metrics['brier']:.3f}")

        # Generate figure
        plot_cross_model_comparison(llama_metrics, qwen_metrics, "figures/fig_cross_model.pdf")

        # Save results
        results = {
            "llama": llama_metrics,
            "qwen": qwen_metrics,
            "n_llama": len(examples),
            "n_qwen": len(valid_examples),
        }
        with open("results/cross_model_metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nSaved: results/cross_model_metrics.json")
    else:
        print("\nWarning: Could not score Qwen answers (no ground truth available)")

    print("\n" + "=" * 50)
    print("CROSS-MODEL EVALUATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
