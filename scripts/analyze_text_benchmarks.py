#!/usr/bin/env python3
"""
Analyze VLM judge predictions per text benchmark.
Computes within-benchmark AUROC, mean P(correct), and base rate.
"""
import sys
import json
import re
import io
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
import tensorflow as tf
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


PROMPT_TEMPLATE = """Question: {question}
Answer: {response}
Is the answer correct? (i) No (ii) Yes"""


@dataclass
class UQTrainingSample:
    question_id: str
    benchmark: str
    prompt: str
    response: str
    is_correct: bool
    dataset_index: Optional[int] = None
    has_image: bool = False


def load_text_samples(path: Path) -> list[UQTrainingSample]:
    samples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            sample = UQTrainingSample(
                question_id=data["id"],
                benchmark=data.get("benchmark", "unknown"),
                prompt=data["input"] if isinstance(data["input"], str) else json.dumps(data["input"]),
                response=data["model_response"],
                is_correct=data["correct"] == 1,
                dataset_index=None,
                has_image=False,
            )
            samples.append(sample)
    return samples


def get_p_correct(model, processor, image, question, response, device):
    prompt = PROMPT_TEMPLATE.format(question=question, response=response)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        min_pixels=256 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]

    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]

    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()


def main():
    print("=" * 80)
    print("PER-BENCHMARK TEXT ANALYSIS")
    print("=" * 80)

    # Config
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    checkpoint_path = Path("data/vlm_judge_combined/checkpoint-788")

    # Load test data
    print("\nLoading text test data...")
    text_test_path = Path("data/finetune/test_v2.jsonl")
    text_test = load_text_samples(text_test_path)
    print(f"Loaded {len(text_test)} samples")

    # Count benchmarks
    benchmarks = defaultdict(list)
    for s in text_test:
        benchmarks[s.benchmark].append(s)

    print(f"\nBenchmarks found: {len(benchmarks)}")
    for b, samples in sorted(benchmarks.items(), key=lambda x: -len(x[1])):
        n_correct = sum(1 for s in samples if s.is_correct)
        print(f"  {b}: {len(samples)} samples ({n_correct} correct, {100*n_correct/len(samples):.1f}%)")

    # Load model
    print(f"\nLoading base model {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    model.eval()

    device = next(model.parameters()).device
    fallback_image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    # Evaluate each sample
    print("\n" + "=" * 50)
    print("RUNNING INFERENCE")
    print("=" * 50)

    results = defaultdict(lambda: {"preds": [], "labels": []})

    for sample in tqdm(text_test, desc="Evaluating"):
        try:
            p_correct = get_p_correct(
                model, processor, fallback_image,
                sample.prompt[:500], sample.response[:300], device
            )
        except Exception as e:
            print(f"Error on {sample.question_id}: {e}")
            p_correct = 0.5

        results[sample.benchmark]["preds"].append(p_correct)
        results[sample.benchmark]["labels"].append(float(sample.is_correct))

    # Compute per-benchmark statistics
    print("\n" + "=" * 80)
    print("PER-BENCHMARK RESULTS")
    print("=" * 80)

    print(f"\n{'Benchmark':<15} | {'N':>5} | {'Within-AUROC':>12} | {'Mean P(corr)':>12} | {'Base Rate':>10} | {'Δ(P-Base)':>10}")
    print("-" * 80)

    all_results = []
    for benchmark in sorted(results.keys()):
        data = results[benchmark]
        preds = np.array(data["preds"])
        labels = np.array(data["labels"])

        n = len(labels)
        mean_p = float(preds.mean())
        base_rate = float(labels.mean())
        delta = mean_p - base_rate

        # Within-benchmark AUROC (only if both classes present)
        if len(set(labels)) >= 2:
            auroc = roc_auc_score(labels, preds)
        else:
            auroc = float('nan')

        all_results.append({
            "benchmark": benchmark,
            "n": n,
            "within_auroc": auroc,
            "mean_p_correct": mean_p,
            "base_rate": base_rate,
            "delta": delta,
        })

        auroc_str = f"{auroc:.3f}" if not np.isnan(auroc) else "N/A"
        print(f"{benchmark:<15} | {n:>5} | {auroc_str:>12} | {mean_p:>12.3f} | {base_rate:>10.3f} | {delta:>+10.3f}")

    # Summary statistics
    valid_aurocs = [r["within_auroc"] for r in all_results if not np.isnan(r["within_auroc"])]

    print("-" * 80)
    print(f"\n{'SUMMARY':^80}")
    print("-" * 80)
    print(f"Mean within-benchmark AUROC: {np.mean(valid_aurocs):.3f}")
    print(f"Median within-benchmark AUROC: {np.median(valid_aurocs):.3f}")
    print(f"Min within-benchmark AUROC: {np.min(valid_aurocs):.3f}")
    print(f"Max within-benchmark AUROC: {np.max(valid_aurocs):.3f}")

    # Interpretation
    mean_auroc = np.mean(valid_aurocs)
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if mean_auroc < 0.55:
        print("⚠️  Mean within-AUROC < 0.55: Model likely learned BENCHMARK PRIORS only")
        print("    The model predicts based on which benchmark, not answer correctness.")
    elif mean_auroc < 0.65:
        print("⚠️  Mean within-AUROC 0.55-0.65: Model has WEAK within-benchmark discrimination")
        print("    Some learning of answer correctness, but benchmark priors dominate.")
    elif mean_auroc < 0.70:
        print("✓  Mean within-AUROC 0.65-0.70: Model has MODERATE discrimination")
        print("    Model learned both benchmark priors AND some answer correctness.")
    else:
        print("✓  Mean within-AUROC > 0.70: Model has GOOD within-benchmark discrimination")
        print("    Model genuinely learned to judge individual answers!")

    # Check for benchmark prior learning
    print("\n" + "-" * 40)
    print("Benchmark prior check (|Mean P - Base Rate|):")
    print("-" * 40)

    for r in sorted(all_results, key=lambda x: abs(x["delta"])):
        status = "✓" if abs(r["delta"]) < 0.15 else "⚠️"
        print(f"  {status} {r['benchmark']:<15}: Δ = {r['delta']:+.3f}")

    # Save results
    output_path = Path("data/vlm_judge_combined/text_benchmark_analysis.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
