#!/usr/bin/env python3
"""
Evaluate VLM Judge on TEXT benchmarks (no image input).
Tests if vision-trained model transfers to text-only UQ.
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# Prompt template (same as VLM training)
PROMPT_TEMPLATE = """Question: {question}
Answer: {response}
Is the answer correct? (i) No (ii) Yes"""


def load_text_test_data(test_path: Path, max_samples: int = None) -> list:
    """Load text UQ test data."""
    samples = []
    with open(test_path) as f:
        for line in f:
            data = json.loads(line)
            samples.append({
                "id": data["id"],
                "benchmark": data.get("benchmark", "unknown"),
                "question": data["input"] if isinstance(data["input"], str) else json.dumps(data["input"]),
                "response": data["model_response"],
                "is_correct": data["correct"] == 1,
            })
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def get_p_correct(model, processor, question: str, response: str, device, dummy_image: Image.Image) -> float:
    """Get P(correct) from model - using dummy image for text-only input."""
    prompt = PROMPT_TEMPLATE.format(question=question[:1000], response=response[:500])

    # Build message with dummy image (model expects image input)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=[dummy_image],
        return_tensors="pt",
        padding=True,
        min_pixels=256 * 28 * 28,
        max_pixels=256 * 28 * 28,  # Minimal image processing
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]

    # Get token IDs for "i" and "ii"
    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]

    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()  # P(correct) = P("ii")


def main():
    print("=" * 70)
    print("EVALUATING VLM JUDGE ON TEXT BENCHMARKS")
    print("=" * 70)
    print("\nTests if vision-trained model transfers to text-only UQ\n")

    # Paths
    lora_path = Path("data/vlm_judge_lora")
    test_path = Path("data/finetune/test_v2.jsonl")

    # Load test data
    print("Loading text test data...")
    test_samples = load_text_test_data(test_path)
    print(f"Loaded {len(test_samples)} test samples")

    # Count correct/incorrect
    n_correct = sum(1 for s in test_samples if s["is_correct"])
    print(f"Class balance: {n_correct} correct ({100*n_correct/len(test_samples):.1f}%), "
          f"{len(test_samples)-n_correct} incorrect ({100*(len(test_samples)-n_correct)/len(test_samples):.1f}%)")

    # Load model
    print("\nLoading Qwen3-VL-8B + LoRA adapter...")
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    model.eval()

    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    # Create dummy image (gray 224x224)
    dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    # Evaluate
    print("\n" + "=" * 50)
    print("RUNNING EVALUATION")
    print("=" * 50)

    all_preds = []
    all_labels = []
    per_benchmark = defaultdict(lambda: {"preds": [], "labels": []})

    for i, sample in enumerate(tqdm(test_samples, desc="Evaluating")):
        try:
            p_correct = get_p_correct(
                model, processor,
                sample["question"],
                sample["response"],
                device,
                dummy_image
            )
        except Exception as e:
            if i < 5:
                print(f"  Error on sample {i}: {e}")
            p_correct = 0.5

        all_preds.append(p_correct)
        all_labels.append(float(sample["is_correct"]))

        # Extract benchmark name from ID
        bench = sample["id"].split("_")[0] if "_" in sample["id"] else "unknown"
        per_benchmark[bench]["preds"].append(p_correct)
        per_benchmark[bench]["labels"].append(float(sample["is_correct"]))

    # Compute metrics
    results = {
        "auroc": roc_auc_score(all_labels, all_preds),
        "auprc": average_precision_score(all_labels, all_preds),
        "n_samples": len(all_labels),
        "n_correct": sum(all_labels),
        "base_rate": sum(all_labels) / len(all_labels),
    }

    # Per-benchmark
    results["per_benchmark"] = {}
    for bench, data in sorted(per_benchmark.items()):
        if len(set(data["labels"])) < 2:
            continue
        results["per_benchmark"][bench] = {
            "auroc": roc_auc_score(data["labels"], data["preds"]),
            "n_samples": len(data["labels"]),
        }

    # ECE and Brier
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    results["brier"] = float(np.mean((preds - labels) ** 2))

    # ECE
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for j in range(n_bins):
        in_bin = (preds >= bin_boundaries[j]) & (preds < bin_boundaries[j + 1])
        if in_bin.sum() == 0:
            continue
        bin_conf = preds[in_bin].mean()
        bin_acc = labels[in_bin].mean()
        ece += in_bin.sum() / len(preds) * abs(bin_conf - bin_acc)
    results["ece"] = float(ece)

    # Print results
    print("\n" + "=" * 70)
    print("VLM JUDGE ON TEXT RESULTS")
    print("=" * 70)
    print(f"\nOverall AUROC: {results['auroc']:.4f}")
    print(f"Overall AUPRC: {results['auprc']:.4f}")
    print(f"ECE: {results['ece']:.4f}")
    print(f"Brier: {results['brier']:.4f}")

    print("\nPer-benchmark AUROC:")
    for bench, data in sorted(results["per_benchmark"].items(), key=lambda x: -x[1]["auroc"]):
        print(f"  {bench}: {data['auroc']:.4f} (n={data['n_samples']})")

    # Compare to baselines
    print("\n" + "-" * 50)
    print("COMPARISON")
    print("-" * 50)
    print(f"VLM Judge on TEXT: {results['auroc']:.4f}")
    print(f"VLM Judge on VISION: 0.766")
    print(f"Probe baseline (vision): 0.704")

    # Save results
    output_path = lora_path / "text_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
