#!/usr/bin/env python3
"""
Evaluate combined VLM judge model on vision and text test sets.
Uses checkpoint-788 (trained for 2 full epochs on combined data).
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
from sklearn.metrics import roc_auc_score, average_precision_score
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


def parse_question_id(question_id: str) -> Optional[int]:
    match = re.search(r'_(\d+)$', question_id)
    if match:
        return int(match.group(1))
    return None


def load_erqa_from_tfrecord(tfrecord_path: str) -> list:
    feature_description = {
        'answer': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.VarLenFeature(tf.string),
        'question_type': tf.io.VarLenFeature(tf.string),
        'visual_indices': tf.io.VarLenFeature(tf.int64),
        'question': tf.io.FixedLenFeature([], tf.string)
    }

    samples = []
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    for example_proto in dataset:
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        images_encoded = tf.sparse.to_dense(parsed['image/encoded']).numpy()

        if len(images_encoded) > 0:
            img_bytes = images_encoded[0]
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                img = None
        else:
            img = None

        samples.append({
            "image": img,
            "question": parsed['question'].numpy().decode('utf-8'),
            "answer": parsed['answer'].numpy().decode('utf-8'),
        })

    return samples


class ERQADataset:
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_benchmark_dataset(benchmark: str):
    if benchmark == "vsr":
        return load_dataset("cambridgeltl/vsr_random", split="test")
    elif benchmark == "mmmu":
        return load_dataset("MMMU/MMMU", "Art", split="validation")
    elif benchmark == "charxiv":
        return load_dataset("princeton-nlp/CharXiv", split="validation")
    elif benchmark == "hallusionbench":
        return load_dataset("lmms-lab/HallusionBench", split="image")
    elif benchmark == "erqa":
        tfrecord_path = Path("data/erqa_repo/data/erqa.tfrecord")
        if tfrecord_path.exists():
            samples = load_erqa_from_tfrecord(str(tfrecord_path))
            return ERQADataset(samples)
        return None
    else:
        return None


def get_image_from_dataset(ds, idx: int, benchmark: str) -> Optional[Image.Image]:
    try:
        if idx >= len(ds):
            return None
        row = ds[idx]
        img = row.get("image")
        if img is None:
            return None
        if isinstance(img, Image.Image):
            return img.convert("RGB") if img.mode != "RGB" else img
        return None
    except Exception:
        return None


def load_vision_samples(base_dir: Path) -> list[UQTrainingSample]:
    samples = []
    for bench_dir in base_dir.iterdir():
        if not bench_dir.is_dir() or bench_dir.name == "smoke_test":
            continue
        benchmark = bench_dir.name
        for pt_file in bench_dir.glob("*.pt"):
            try:
                data = torch.load(pt_file, weights_only=False)
                question_id = data.get("question_id", "")
                idx = parse_question_id(question_id)
                if idx is None:
                    continue
                sample = UQTrainingSample(
                    question_id=question_id,
                    benchmark=benchmark,
                    prompt=data.get("prompt", ""),
                    response=data.get("response", ""),
                    is_correct=data.get("is_correct", False),
                    dataset_index=idx,
                    has_image=True,
                )
                samples.append(sample)
            except Exception:
                continue
    return samples


def load_text_samples(path: Path) -> list[UQTrainingSample]:
    samples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            sample = UQTrainingSample(
                question_id=data["id"],
                benchmark=data.get("benchmark", "text"),
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


def evaluate_model(
    model,
    processor,
    test_samples: list[UQTrainingSample],
    datasets_cache: dict,
    device: torch.device,
    fallback_image: Image.Image,
    desc: str = "Evaluating",
) -> dict:
    model.eval()

    all_preds = []
    all_labels = []
    per_benchmark = defaultdict(lambda: {"preds": [], "labels": []})

    for sample in tqdm(test_samples, desc=desc):
        if sample.has_image and sample.benchmark in datasets_cache:
            ds = datasets_cache.get(sample.benchmark)
            image = get_image_from_dataset(ds, sample.dataset_index, sample.benchmark)
            if image is None:
                image = fallback_image
        else:
            image = fallback_image

        try:
            p_correct = get_p_correct(
                model, processor, image,
                sample.prompt[:500], sample.response[:300], device
            )
        except Exception as e:
            print(f"Error on {sample.question_id}: {e}")
            p_correct = 0.5

        all_preds.append(p_correct)
        all_labels.append(float(sample.is_correct))
        per_benchmark[sample.benchmark]["preds"].append(p_correct)
        per_benchmark[sample.benchmark]["labels"].append(float(sample.is_correct))

    results = {
        "auroc": roc_auc_score(all_labels, all_preds),
        "auprc": average_precision_score(all_labels, all_preds),
        "n_samples": len(all_labels),
        "n_correct": sum(all_labels),
        "base_rate": sum(all_labels) / len(all_labels),
    }

    results["per_benchmark"] = {}
    for bench, data in per_benchmark.items():
        if len(set(data["labels"])) < 2:
            continue
        results["per_benchmark"][bench] = {
            "auroc": roc_auc_score(data["labels"], data["preds"]),
            "n_samples": len(data["labels"]),
        }

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    results["brier"] = float(np.mean((preds - labels) ** 2))

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

    return results


def main():
    print("=" * 70)
    print("EVALUATING COMBINED VLM JUDGE (checkpoint-788)")
    print("=" * 70)

    # Config
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    checkpoint_path = Path("data/vlm_judge_combined/checkpoint-788")
    output_dir = Path("data/vlm_judge_combined")

    # Load test data
    print("\nLoading test data...")

    # Vision test
    vision_dir = Path("data/features")
    vision_samples = load_vision_samples(vision_dir)
    print(f"Loaded {len(vision_samples)} vision samples")

    test_ids_path = Path("data/probe_results/test_ids.json")
    with open(test_ids_path) as f:
        test_ids_data = json.load(f)
        vision_test_ids = set(item["question_id"] for item in test_ids_data)

    vision_test = [s for s in vision_samples if s.question_id in vision_test_ids]
    print(f"Vision test: {len(vision_test)} samples")

    # Text test
    text_test_path = Path("data/finetune/test_v2.jsonl")
    text_test = load_text_samples(text_test_path)
    print(f"Text test: {len(text_test)} samples")

    # Load model
    print(f"\nLoading base model {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    print(f"Loading LoRA adapter from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    model.eval()

    device = next(model.parameters()).device
    fallback_image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    # Load benchmark datasets for evaluation
    print("\nLoading benchmark datasets...")
    vision_benchmarks = list(set(s.benchmark for s in vision_test if s.has_image))
    datasets_cache = {}
    for bench in vision_benchmarks:
        print(f"  Loading {bench}...")
        datasets_cache[bench] = load_benchmark_dataset(bench)

    # Evaluate on vision test
    print("\n" + "=" * 50)
    print("EVALUATING ON VISION TEST SET")
    print("=" * 50)
    vision_results = evaluate_model(
        model, processor, vision_test, datasets_cache, device, fallback_image, "Vision"
    )
    print(f"\nVision Test Results:")
    print(f"  AUROC: {vision_results['auroc']:.4f}")
    print(f"  AUPRC: {vision_results['auprc']:.4f}")
    print(f"  Brier: {vision_results['brier']:.4f}")
    print(f"  ECE:   {vision_results['ece']:.4f}")
    print(f"\nPer-benchmark AUROC:")
    for bench, data in vision_results["per_benchmark"].items():
        print(f"  {bench}: {data['auroc']:.3f} (n={data['n_samples']})")

    # Evaluate on text test
    print("\n" + "=" * 50)
    print("EVALUATING ON TEXT TEST SET")
    print("=" * 50)
    text_results = evaluate_model(
        model, processor, text_test, {}, device, fallback_image, "Text"
    )
    print(f"\nText Test Results:")
    print(f"  AUROC: {text_results['auroc']:.4f}")
    print(f"  AUPRC: {text_results['auprc']:.4f}")
    print(f"  Brier: {text_results['brier']:.4f}")
    print(f"  ECE:   {text_results['ece']:.4f}")
    print(f"\nPer-benchmark AUROC:")
    for bench, data in text_results["per_benchmark"].items():
        print(f"  {bench}: {data['auroc']:.3f} (n={data['n_samples']})")

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print("\n| Model | Vision AUROC | Text AUROC |")
    print("|-------|--------------|------------|")
    print(f"| Vision-only trained   | 0.766        | 0.811      |")
    print(f"| Combined (2 epochs)   | {vision_results['auroc']:.3f}        | {text_results['auroc']:.3f}      |")

    delta_vision = vision_results['auroc'] - 0.766
    delta_text = text_results['auroc'] - 0.811
    print(f"\nDelta from vision-only:")
    print(f"  Vision: {delta_vision:+.3f}")
    print(f"  Text:   {delta_text:+.3f}")

    # Save results
    results = {
        "checkpoint": "checkpoint-788",
        "epochs_trained": 2,
        "vision": vision_results,
        "text": text_results,
        "comparison": {
            "vision_only_vision_auroc": 0.766,
            "vision_only_text_auroc": 0.811,
            "combined_vision_auroc": vision_results['auroc'],
            "combined_text_auroc": text_results['auroc'],
            "delta_vision": delta_vision,
            "delta_text": delta_text,
        }
    }

    results_path = output_dir / "combined_checkpoint788_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
