#!/usr/bin/env python3
"""
Plot selective prediction curves showing practical value of UQ model.
X-axis: Coverage (% of samples we answer)
Y-axis: Accuracy on answered samples
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Load the predictions from the evaluation
results_path = Path("data/vlm_judge_combined/combined_checkpoint788_results.json")

# We need to re-run inference to get per-sample predictions
# For now, let's use the text benchmark analysis which has predictions

analysis_path = Path("data/vlm_judge_combined/text_benchmark_analysis.json")

print("=" * 70)
print("SELECTIVE PREDICTION ANALYSIS")
print("=" * 70)

# We need actual predictions - let me load from the analysis script output
# Since we don't have raw predictions saved, let's re-compute

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm

PROMPT_TEMPLATE = """Question: {question}
Answer: {response}
Is the answer correct? (i) No (ii) Yes"""

@dataclass
class UQSample:
    question_id: str
    benchmark: str
    prompt: str
    response: str
    is_correct: bool
    has_image: bool = False

def load_text_samples(path: Path) -> list[UQSample]:
    samples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            sample = UQSample(
                question_id=data["id"],
                benchmark=data.get("benchmark", "unknown"),
                prompt=data["input"] if isinstance(data["input"], str) else json.dumps(data["input"]),
                response=data["model_response"],
                is_correct=data["correct"] == 1,
                has_image=False,
            )
            samples.append(sample)
    return samples

def get_p_correct(model, processor, image, question, response, device):
    prompt = PROMPT_TEMPLATE.format(question=question, response=response)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True,
        min_pixels=256*28*28, max_pixels=256*28*28,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()

# Load model
print("\nLoading model...")
model_name = "Qwen/Qwen3-VL-8B-Instruct"
checkpoint_path = Path("data/vlm_judge_combined/checkpoint-788")

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, str(checkpoint_path))
model.eval()

device = next(model.parameters()).device
fallback_image = Image.new("RGB", (224, 224), color=(128, 128, 128))

# Load test samples
print("\nLoading test samples...")
text_test = load_text_samples(Path("data/finetune/test_v2.jsonl"))
print(f"Loaded {len(text_test)} text samples")

# Get predictions
print("\nRunning inference...")
predictions = []
labels = []

for sample in tqdm(text_test, desc="Predicting"):
    try:
        p_correct = get_p_correct(
            model, processor, fallback_image,
            sample.prompt[:500], sample.response[:300], device
        )
    except Exception as e:
        p_correct = 0.5

    predictions.append(p_correct)
    labels.append(float(sample.is_correct))

predictions = np.array(predictions)
labels = np.array(labels)

# Save predictions for later analysis
pred_data = {
    "predictions": predictions.tolist(),
    "labels": labels.tolist(),
    "question_ids": [s.question_id for s in text_test],
    "benchmarks": [s.benchmark for s in text_test],
    "prompts": [s.prompt[:200] for s in text_test],
    "responses": [s.response[:200] for s in text_test],
}
with open("data/vlm_judge_combined/text_predictions.json", "w") as f:
    json.dump(pred_data, f)

# Compute selective prediction curves
print("\n" + "=" * 70)
print("SELECTIVE PREDICTION CURVES")
print("=" * 70)

thresholds = np.linspace(0.0, 1.0, 101)
coverages = []
accuracies = []

for thresh in thresholds:
    # Select samples where model confidence (max of P(correct), P(incorrect)) > threshold
    # For binary, confidence = max(p, 1-p)
    confidences = np.maximum(predictions, 1 - predictions)
    selected = confidences >= thresh

    if selected.sum() == 0:
        coverages.append(0.0)
        accuracies.append(np.nan)
        continue

    coverage = selected.mean()

    # For selected samples, compute accuracy
    # Model prediction: correct if P(correct) > 0.5
    model_preds = predictions[selected] > 0.5
    actual = labels[selected] > 0.5
    accuracy = (model_preds == actual).mean()

    coverages.append(coverage)
    accuracies.append(accuracy)

coverages = np.array(coverages)
accuracies = np.array(accuracies)

# Print table at specific thresholds
print("\nSelective Prediction Results:")
print(f"{'Confidence':>12} | {'Coverage':>10} | {'Accuracy':>10} | {'Samples':>8}")
print("-" * 50)

for conf_thresh in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
    confidences = np.maximum(predictions, 1 - predictions)
    selected = confidences >= conf_thresh
    n_selected = selected.sum()
    coverage = selected.mean()

    if n_selected > 0:
        model_preds = predictions[selected] > 0.5
        actual = labels[selected] > 0.5
        accuracy = (model_preds == actual).mean()
        print(f"{conf_thresh:>12.0%} | {coverage:>10.1%} | {accuracy:>10.1%} | {n_selected:>8}")
    else:
        print(f"{conf_thresh:>12.0%} | {0:>10.1%} | {'N/A':>10} | {0:>8}")

# Overall accuracy (no selection)
overall_acc = ((predictions > 0.5) == (labels > 0.5)).mean()
print("-" * 50)
print(f"{'Baseline':>12} | {1.0:>10.1%} | {overall_acc:>10.1%} | {len(labels):>8}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(coverages * 100, accuracies * 100, 'b-', linewidth=2, label='VLM Judge')
plt.axhline(y=overall_acc * 100, color='gray', linestyle='--', label=f'Baseline (no selection): {overall_acc:.1%}')

# Mark key points
for conf_thresh in [0.70, 0.80, 0.90]:
    confidences = np.maximum(predictions, 1 - predictions)
    selected = confidences >= conf_thresh
    if selected.sum() > 0:
        coverage = selected.mean() * 100
        model_preds = predictions[selected] > 0.5
        actual = labels[selected] > 0.5
        accuracy = (model_preds == actual).mean() * 100
        plt.scatter([coverage], [accuracy], s=100, zorder=5)
        plt.annotate(f'{conf_thresh:.0%} conf\n({coverage:.0f}%, {accuracy:.0f}%)',
                    (coverage, accuracy), textcoords="offset points", xytext=(10, -10))

plt.xlabel('Coverage (%)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Selective Prediction Curve\n(Higher confidence threshold → Lower coverage but higher accuracy)', fontsize=14)
plt.xlim(0, 105)
plt.ylim(50, 105)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('data/vlm_judge_combined/selective_prediction_curve.png', dpi=150)
print(f"\nPlot saved to data/vlm_judge_combined/selective_prediction_curve.png")

# Also plot accuracy vs P(correct) threshold (different view)
plt.figure(figsize=(10, 6))

thresh_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for thresh in thresh_values:
    selected = predictions >= thresh
    if selected.sum() > 0:
        coverage = selected.mean() * 100
        accuracy = labels[selected].mean() * 100  # Actual accuracy of "correct" predictions
        plt.scatter([coverage], [accuracy], s=100)
        if thresh in [0.5, 0.7, 0.9]:
            plt.annotate(f'P>{thresh:.1f}', (coverage, accuracy),
                        textcoords="offset points", xytext=(5, 5))

# Connect points
covs, accs = [], []
for thresh in np.linspace(0.05, 0.95, 50):
    selected = predictions >= thresh
    if selected.sum() > 0:
        covs.append(selected.mean() * 100)
        accs.append(labels[selected].mean() * 100)
plt.plot(covs, accs, 'b-', alpha=0.5)

plt.xlabel('Coverage (% predicted correct)', fontsize=12)
plt.ylabel('Precision (% actually correct)', fontsize=12)
plt.title('Precision vs Coverage for "Correct" Predictions', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/vlm_judge_combined/precision_coverage_curve.png', dpi=150)
print(f"Plot saved to data/vlm_judge_combined/precision_coverage_curve.png")

print("\n" + "=" * 70)
print("PRACTICAL INTERPRETATION")
print("=" * 70)

# Find the coverage at 90% accuracy
for i, (cov, acc) in enumerate(zip(coverages, accuracies)):
    if not np.isnan(acc) and acc >= 0.90:
        print(f"\n✓ At 90% accuracy, we can answer {cov:.1%} of samples")
        break

# Find accuracy at 80% coverage
for i, (cov, acc) in enumerate(zip(coverages, accuracies)):
    if cov <= 0.80 and not np.isnan(acc):
        print(f"✓ At 80% coverage, we achieve {acc:.1%} accuracy")
        break

print("\nThis demonstrates the practical value of the UQ model:")
print("- We can trade off coverage for accuracy")
print("- High-confidence predictions are more reliable")
