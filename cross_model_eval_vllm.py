#!/usr/bin/env python3
"""
Cross-model evaluation using vLLM for fast Qwen inference.
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

                # Get question - handle various formats
                question = row.get("input", "")
                if isinstance(question, dict):
                    question = question.get("text", question.get("content", str(question)))
                question = str(question)

                examples.append({
                    "id": ex_id,
                    "benchmark": bench,
                    "question": question,
                    "llama_answer": row.get("model_response", row.get("prediction", {}).get("answer", "")),
                    "llama_correct": row.get("correct", 0) == 1,
                    "ground_truth": row.get("ground_truth", row.get("expected", "")),
                })
    return examples


def run_qwen_vllm(examples: list[dict], model_name: str, max_tokens: int = 256) -> list[dict]:
    """Run Qwen inference using vLLM for speed."""
    from vllm import LLM, SamplingParams

    print(f"\nLoading Qwen via vLLM: {model_name}")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
    )

    # Format prompts for Qwen
    prompts = []
    for ex in examples:
        prompt = f"<|im_start|>user\n{ex['question']}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt)

    print(f"Running inference on {len(prompts)} examples...")
    outputs = llm.generate(prompts, sampling_params)

    for ex, output in zip(examples, outputs):
        ex["qwen_answer"] = output.outputs[0].text.strip()

    # Free vLLM memory
    del llm
    torch.cuda.empty_cache()

    return examples


def score_qwen_correctness(examples: list[dict]) -> list[dict]:
    """Score Qwen answers for correctness."""
    for ex in examples:
        gt = str(ex.get("ground_truth", "")).strip().lower()
        answer = ex.get("qwen_answer", "").strip().lower()

        if gt and len(gt) > 0:
            if len(gt) < 50:
                ex["qwen_correct"] = gt in answer or answer in gt
            else:
                gt_words = set(gt.split())
                answer_words = set(answer.split())
                overlap = len(gt_words & answer_words) / max(len(gt_words), 1)
                ex["qwen_correct"] = overlap > 0.5
        else:
            ex["qwen_correct"] = None

    valid = [ex for ex in examples if ex.get("qwen_correct") is not None]
    print(f"Scored {len(valid)}/{len(examples)} Qwen answers")
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

                logits_pair = torch.tensor([logit_i, logit_ii]) / temperature
                probs = torch.softmax(logits_pair, dim=0)
                confidences.append(probs[1].item())

    return confidences


def compute_metrics(confidences: list[float], correct: list[bool]) -> dict:
    """Compute AUROC, ECE, Brier score."""
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
    print("CROSS-MODEL EVALUATION (vLLM)")
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

    # Run Qwen inference with vLLM
    examples = run_qwen_vllm(examples, args.qwen_model)

    # Score Qwen answers
    examples = score_qwen_correctness(examples)

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

    # Get confidences for Qwen answers
    valid_examples = [ex for ex in examples if ex.get("qwen_correct") is not None]
    print(f"\nScoring {len(valid_examples)} Qwen answers with UQ model...")
    qwen_confidences = get_uq_confidences(valid_examples, uq_model, uq_tokenizer, "qwen_answer", temperature)
    qwen_correct = [ex["qwen_correct"] for ex in valid_examples]

    # Compute metrics
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    llama_metrics = compute_metrics(llama_confidences, llama_correct)
    print(f"\nLlama-3.1-8B (in-distribution):")
    print(f"  N = {len(examples)}")
    print(f"  AUROC: {llama_metrics['auroc']:.3f}")
    print(f"  ECE:   {llama_metrics['ece']:.3f}")
    print(f"  Brier: {llama_metrics['brier']:.3f}")

    if len(qwen_correct) > 0:
        qwen_metrics = compute_metrics(qwen_confidences, qwen_correct)
        print(f"\nQwen-2.5-7B (cross-model):")
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

    print("\n" + "=" * 50)
    print("CROSS-MODEL EVALUATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
