"""
UQ Evaluation: Evaluate uncertainty quantification methods.

Metrics from the paper:
1. AUROC - Area Under ROC Curve (discrimination)
2. ECE - Expected Calibration Error (calibration)
3. Brier Score - Combined calibration and discrimination

Methods evaluated:
1. Verbalized confidence (from prompting)
2. Zero-shot classifier P("True") / (P("True") + P("False"))
3. Fine-tuned classifier
4. Perplexity (baseline)
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from sklearn.metrics import roc_auc_score, brier_score_loss
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


@dataclass
class UQMetrics:
    """Container for UQ evaluation metrics."""
    auroc: float
    ece: float
    brier: float
    accuracy: float
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "auroc": self.auroc,
            "ece": self.ece,
            "brier": self.brier,
            "accuracy": self.accuracy,
            "n_samples": self.n_samples,
        }


def compute_ece(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    Where B_b is the set of predictions in bin b.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = correctness[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return ece


def compute_auroc(confidences: np.ndarray, correctness: np.ndarray) -> float:
    """Compute Area Under ROC Curve."""
    if len(np.unique(correctness)) < 2:
        return 0.5  # Can't compute AUROC with single class
    return roc_auc_score(correctness, confidences)


def compute_brier(confidences: np.ndarray, correctness: np.ndarray) -> float:
    """Compute Brier Score."""
    return brier_score_loss(correctness, confidences)


def evaluate_verbalized_confidence(predictions_path: Path) -> UQMetrics:
    """
    Evaluate verbalized confidence from model outputs.
    Uses the 'confidence' field extracted during evaluation.
    """
    confidences = []
    correctness = []

    with open(predictions_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)

            # Skip API failures
            if "API_FAILED" in str(row.get("response_text", "")):
                continue

            conf = row.get("prediction", {}).get("confidence")
            correct = row.get("score", {}).get("correct", 0)

            if conf is not None:
                confidences.append(float(conf))
                correctness.append(int(correct))

    if not confidences:
        return UQMetrics(auroc=0.5, ece=1.0, brier=1.0, accuracy=0.0, n_samples=0)

    confidences = np.array(confidences)
    correctness = np.array(correctness)

    return UQMetrics(
        auroc=compute_auroc(confidences, correctness),
        ece=compute_ece(confidences, correctness),
        brier=compute_brier(confidences, correctness),
        accuracy=correctness.mean(),
        n_samples=len(correctness),
    )


def evaluate_zero_shot_classifier(
    predictions_path: Path,
    model_name: str,
    device: str = "cuda",
) -> UQMetrics:
    """
    Evaluate zero-shot classifier approach.
    P("True") / (P("True") + P("False"))
    """
    print(f"Loading model {model_name} for zero-shot evaluation...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Get token IDs for True/False
    true_id = tokenizer.encode("True", add_special_tokens=False)[0]
    false_id = tokenizer.encode("False", add_special_tokens=False)[0]

    confidences = []
    correctness = []

    with open(predictions_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    for row in examples:
        if "API_FAILED" in str(row.get("response_text", "")):
            continue

        question = row.get("input", "")
        answer = row.get("prediction", {}).get("answer", "")
        correct = row.get("score", {}).get("correct", 0)

        if not question or not answer:
            continue

        # Format prompt for True/False classification
        prompt = (
            f"Question: {question}\n\n"
            f"Model's Answer: {answer}\n\n"
            f"Is this answer correct? Answer True or False:"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

            # Get probabilities for True/False
            probs = torch.softmax(logits[[true_id, false_id]], dim=0)
            p_true = probs[0].item()

        confidences.append(p_true)
        correctness.append(int(correct))

    if not confidences:
        return UQMetrics(auroc=0.5, ece=1.0, brier=1.0, accuracy=0.0, n_samples=0)

    confidences = np.array(confidences)
    correctness = np.array(correctness)

    return UQMetrics(
        auroc=compute_auroc(confidences, correctness),
        ece=compute_ece(confidences, correctness),
        brier=compute_brier(confidences, correctness),
        accuracy=correctness.mean(),
        n_samples=len(correctness),
    )


def evaluate_finetuned_classifier(
    predictions_path: Path,
    uq_model_path: str,
    base_model: Optional[str] = None,
    device: str = "cuda",
) -> UQMetrics:
    """
    Evaluate fine-tuned UQ classifier.
    """
    print(f"Loading fine-tuned model from {uq_model_path}...")

    # Check if it's a LoRA model
    adapter_config_path = Path(uq_model_path) / "adapter_config.json"

    if adapter_config_path.exists():
        # LoRA model - need base model
        if base_model is None:
            with open(adapter_config_path) as f:
                config = json.load(f)
                base_model = config.get("base_model_name_or_path")

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, uq_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(uq_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            uq_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()

    # Get token IDs for True/False
    true_id = tokenizer.encode("True", add_special_tokens=False)[0]
    false_id = tokenizer.encode("False", add_special_tokens=False)[0]

    confidences = []
    correctness = []

    with open(predictions_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    for row in examples:
        if "API_FAILED" in str(row.get("response_text", "")):
            continue

        question = row.get("input", "")
        answer = row.get("prediction", {}).get("answer", "")
        correct = row.get("score", {}).get("correct", 0)

        if not question or not answer:
            continue

        # Format prompt
        prompt = (
            f"Question: {question}\n\n"
            f"Model's Answer: {answer}\n\n"
            f"Is this answer correct? Answer True or False:"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]

            probs = torch.softmax(logits[[true_id, false_id]], dim=0)
            p_true = probs[0].item()

        confidences.append(p_true)
        correctness.append(int(correct))

    if not confidences:
        return UQMetrics(auroc=0.5, ece=1.0, brier=1.0, accuracy=0.0, n_samples=0)

    confidences = np.array(confidences)
    correctness = np.array(correctness)

    return UQMetrics(
        auroc=compute_auroc(confidences, correctness),
        ece=compute_ece(confidences, correctness),
        brier=compute_brier(confidences, correctness),
        accuracy=correctness.mean(),
        n_samples=len(correctness),
    )


def evaluate_all_methods(
    predictions_path: Path,
    base_model: str,
    uq_model_path: Optional[str] = None,
) -> dict[str, UQMetrics]:
    """Evaluate all UQ methods on a prediction file."""
    results = {}

    # 1. Verbalized confidence (from prompting)
    print("\n=== Evaluating Verbalized Confidence ===")
    results["verbalized"] = evaluate_verbalized_confidence(predictions_path)
    print(f"  AUROC: {results['verbalized'].auroc:.4f}")
    print(f"  ECE: {results['verbalized'].ece:.4f}")
    print(f"  Brier: {results['verbalized'].brier:.4f}")

    # 2. Zero-shot classifier
    print("\n=== Evaluating Zero-Shot Classifier ===")
    results["zero_shot"] = evaluate_zero_shot_classifier(predictions_path, base_model)
    print(f"  AUROC: {results['zero_shot'].auroc:.4f}")
    print(f"  ECE: {results['zero_shot'].ece:.4f}")
    print(f"  Brier: {results['zero_shot'].brier:.4f}")

    # 3. Fine-tuned classifier (if available)
    if uq_model_path and Path(uq_model_path).exists():
        print("\n=== Evaluating Fine-tuned Classifier ===")
        results["finetuned"] = evaluate_finetuned_classifier(
            predictions_path, uq_model_path, base_model
        )
        print(f"  AUROC: {results['finetuned'].auroc:.4f}")
        print(f"  ECE: {results['finetuned'].ece:.4f}")
        print(f"  Brier: {results['finetuned'].brier:.4f}")

    return results


def aggregate_results_across_benchmarks(
    runs_dir: Path,
    base_model: str,
    uq_model_path: Optional[str] = None,
) -> dict[str, dict[str, UQMetrics]]:
    """Evaluate across all benchmark runs."""
    all_results = {}

    for pred_file in runs_dir.glob("*/predictions.jsonl"):
        benchmark_name = pred_file.parent.name
        print(f"\n{'='*60}")
        print(f"Evaluating: {benchmark_name}")
        print(f"{'='*60}")

        # Just evaluate verbalized for speed (full eval is slow)
        results = {
            "verbalized": evaluate_verbalized_confidence(pred_file),
        }
        all_results[benchmark_name] = results

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, help="Path to predictions.jsonl")
    parser.add_argument("--runs_dir", type=str, help="Directory with all runs")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--uq_model", type=str, help="Path to fine-tuned UQ model")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    if args.predictions:
        results = evaluate_all_methods(
            Path(args.predictions),
            args.base_model,
            args.uq_model,
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)

    elif args.runs_dir:
        results = aggregate_results_across_benchmarks(
            Path(args.runs_dir),
            args.base_model,
            args.uq_model,
        )

        # Print summary table
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Benchmark':<50} {'AUROC':>8} {'ECE':>8} {'Brier':>8}")
        print("-" * 80)

        for bench, methods in sorted(results.items()):
            verb = methods.get("verbalized")
            if verb:
                print(f"{bench:<50} {verb.auroc:>8.4f} {verb.ece:>8.4f} {verb.brier:>8.4f}")

        if args.output:
            with open(args.output, "w") as f:
                output = {
                    k: {m: v.to_dict() for m, v in methods.items()}
                    for k, methods in results.items()
                }
                json.dump(output, f, indent=2)
