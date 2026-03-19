#!/usr/bin/env python3
"""Ablation: Truncation length and response scrambling.

Tests:
1. Truncation: How does AUROC vary with response truncation length?
   (200, 400, 800, 1600, 3000 chars)
2. Scramble: Shuffle words in the response. If AUROC drops to ~0.5,
   the model reads content. If it stays high, it uses surface features.

Usage:
    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/ablation_truncation_scramble.py --smoke_test

    # Full run
    CUDA_VISIBLE_DEVICES=0 python scripts/ablation_truncation_scramble.py

    # Truncation only
    CUDA_VISIBLE_DEVICES=0 python scripts/ablation_truncation_scramble.py --mode truncation

    # Scramble only
    CUDA_VISIBLE_DEVICES=0 python scripts/ablation_truncation_scramble.py --mode scramble
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_CHECKPOINT = "uq_models/best_v2_r32_combined"
IMAGE_CACHE_DIR = Path("data/training_images")
OUTPUT_DIR = Path("data/ablations/truncation_scramble")

VLM_BENCHMARKS = {
    "charxiv", "mmmu", "mmstar", "hallusionbench", "mathverse",
    "mathvision", "mathvista", "realworldqa", "vizwiz", "hle_multimodal", "mmvet",
}
EXCLUDED_BENCHMARKS = {
    "triviaqa", "babilong", "vsr", "aokvqa", "erqa",
    "tutorbench", "healthbench", "arc", "oolong",
}
TARGET_CONFIGS = {
    "gpt5mini": {"data_dir": "runs/gpt5_mini_combined", "mode": "combined"},
    "gpt52": {"prefix": "gpt52_high_", "mode": "prefixed"},
    "qwen35": {"prefix": "qwen35_397b_", "mode": "prefixed"},
}

PROMPT_COMBINED = """Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes"""


# ============================================================
# DATA LOADING
# ============================================================

def extract_question_text(input_data) -> str:
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, dict):
        for key in ["question", "query", "query_cot", "prompt", "text"]:
            if key in input_data and input_data[key]:
                val = input_data[key]
                if isinstance(val, str):
                    return val
        if "messages" in input_data:
            for msg in input_data["messages"]:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        texts = [p.get("text", "") for p in content
                                 if isinstance(p, dict) and "text" in p]
                        return " ".join(texts)
        clean = {k: v for k, v in input_data.items() if k != "images"}
        return json.dumps(clean)[:2000]
    return str(input_data)[:2000]


def load_predictions(pred_file: Path, benchmark: str) -> list:
    samples = []
    with open(pred_file) as f:
        for line in f:
            try:
                pred = json.loads(line)
            except json.JSONDecodeError:
                continue
            score = pred.get("score", {})
            correct = score.get("correct", -1) if isinstance(score, dict) else score
            if correct not in (0, 1):
                continue
            question = extract_question_text(pred.get("input", {}))
            response = pred.get("response_text", "")
            if not response:
                response = str(pred.get("prediction", {}).get("answer", ""))
            if not question or not response:
                continue
            samples.append({
                "id": str(pred.get("id", "")),
                "benchmark": benchmark,
                "question": question[:2000],
                "response": response[:5000],  # keep more for truncation ablation
                "is_correct": int(correct == 1),
                "has_image": benchmark in VLM_BENCHMARKS,
            })
    return samples


def load_all_test_samples(test_ids: set):
    all_samples = []
    for target, config in TARGET_CONFIGS.items():
        samples = []
        if config["mode"] == "combined":
            data_path = Path(config["data_dir"])
            if not data_path.exists():
                continue
            for bench_dir in sorted(data_path.iterdir()):
                if not bench_dir.is_dir():
                    continue
                benchmark = bench_dir.name
                if benchmark in EXCLUDED_BENCHMARKS:
                    continue
                pred_file = bench_dir / "predictions.jsonl"
                if not pred_file.exists():
                    continue
                samples.extend(load_predictions(pred_file, benchmark))
        else:
            prefix = config["prefix"]
            for run_dir in sorted(Path("runs").iterdir()):
                if not run_dir.name.startswith(prefix):
                    continue
                benchmark = run_dir.name[len(prefix):]
                if benchmark in EXCLUDED_BENCHMARKS or "backup" in run_dir.name:
                    continue
                pred_file = run_dir / "predictions.jsonl"
                if not pred_file.exists():
                    continue
                samples.extend(load_predictions(pred_file, benchmark))

        test_samples = [s for s in samples
                       if f"{s['benchmark']}_{s['id']}" in test_ids or s["id"] in test_ids]
        for s in test_samples:
            s["target_model"] = target
        print(f"  {target}: {len(test_samples)} test samples")
        all_samples.extend(test_samples)
    return all_samples


# ============================================================
# SCRAMBLE
# ============================================================

def scramble_words(text: str, seed: int = 42) -> str:
    """Shuffle words in text while preserving punctuation attachment."""
    rng = random.Random(seed)
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def scramble_sentences(text: str, seed: int = 42) -> str:
    """Shuffle sentences while keeping words within sentences intact."""
    import re
    rng = random.Random(seed)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    rng.shuffle(sentences)
    return " ".join(sentences)


# ============================================================
# INFERENCE
# ============================================================

def load_image_for_sample(sample: dict) -> Image.Image:
    fallback = Image.new('RGB', (224, 224), color='gray')
    if not sample["has_image"]:
        return fallback
    cache_path = IMAGE_CACHE_DIR / sample["benchmark"] / f"{sample['id']}.jpg"
    if cache_path.exists():
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            return fallback
    return fallback


def score_sample(model, processor, sample, device, q_len=1500, r_len=800,
                 response_override=None, target_model=""):
    """Score a single sample with the combined template."""
    question = sample["question"][:q_len]
    response = (response_override or sample["response"])[:r_len]

    prompt = PROMPT_COMBINED.format(
        question=question,
        response=response,
        benchmark=sample["benchmark"],
        source_model=target_model,
    )
    image = load_image_for_sample(sample)
    is_vlm = sample["has_image"]
    min_px = 256 * 28 * 28
    max_px = 512 * 28 * 28 if is_vlm else 256 * 28 * 28

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True,
        min_pixels=min_px, max_pixels=max_px,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()


def compute_metrics(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "n": len(labels)}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "n": len(labels),
    }


def run_condition(model, processor, device, samples, condition_name,
                  output_dir, r_len=800, q_len=1500, response_transform=None):
    """Score all samples under a given condition."""
    print(f"\n{'='*70}")
    print(f"CONDITION: {condition_name} (r_len={r_len}, q_len={q_len})")
    print(f"{'='*70}")

    labels, scores = [], []
    per_benchmark = defaultdict(lambda: {"labels": [], "scores": []})
    per_model = defaultdict(lambda: {"labels": [], "scores": []})
    t0 = time.time()

    preds_path = output_dir / f"{condition_name}_predictions.jsonl"
    with open(preds_path, "w") as f_out:
        for i, sample in enumerate(samples):
            response_override = None
            if response_transform:
                response_override = response_transform(sample["response"])

            try:
                p = score_sample(
                    model, processor, sample, device,
                    q_len=q_len, r_len=r_len,
                    response_override=response_override,
                    target_model=sample["target_model"],
                )
            except Exception as e:
                if i < 5:
                    print(f"  Error on sample {i}: {e}")
                p = 0.5

            label = sample["is_correct"]
            labels.append(label)
            scores.append(p)
            per_benchmark[sample["benchmark"]]["labels"].append(label)
            per_benchmark[sample["benchmark"]]["scores"].append(p)
            per_model[sample["target_model"]]["labels"].append(label)
            per_model[sample["target_model"]]["scores"].append(p)

            f_out.write(json.dumps({
                "id": sample["id"], "benchmark": sample["benchmark"],
                "target_model": sample["target_model"],
                "is_correct": label, "p_correct": round(p, 6),
            }) + "\n")

            if (i + 1) % 200 == 0 or (i + 1) == len(samples):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                interim = ""
                if len(set(labels)) > 1:
                    interim = f", AUROC={roc_auc_score(labels, scores):.4f}"
                print(f"  [{i+1}/{len(samples)}] {rate:.1f}/s, {elapsed:.0f}s{interim}")
                f_out.flush()

    overall = compute_metrics(labels, scores)
    bench_metrics = {b: compute_metrics(d["labels"], d["scores"])
                     for b, d in sorted(per_benchmark.items())}
    model_metrics = {m: compute_metrics(d["labels"], d["scores"])
                     for m, d in sorted(per_model.items())}

    result = {
        "condition": condition_name,
        "overall": overall,
        "per_benchmark": bench_metrics,
        "per_model": model_metrics,
        "config": {"q_len": q_len, "r_len": r_len,
                    "transform": response_transform.__name__ if response_transform else None},
    }

    result_path = output_dir / f"{condition_name}_results.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  AUROC: {overall['auroc']:.4f} | N: {overall['n']}")
    for m, v in sorted(model_metrics.items()):
        print(f"    {m}: {v['auroc']:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--mode", default="both", choices=["truncation", "scramble", "both"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test IDs
    split_info_path = Path(args.checkpoint) / "split_info.json"
    with open(split_info_path) as f:
        split_info = json.load(f)
    test_ids = set(split_info["test_ids"])
    print(f"Loaded {len(test_ids)} test IDs")

    # Load samples
    print("\nLoading test samples...")
    all_samples = load_all_test_samples(test_ids)

    if args.smoke_test:
        from collections import Counter
        counts = Counter()
        filtered = []
        for s in all_samples:
            if counts[s["target_model"]] < 5:
                filtered.append(s)
                counts[s["target_model"]] += 1
        all_samples = filtered

    print(f"Total: {len(all_samples)} samples")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()
    print("Model loaded.")

    all_results = {}

    # --- TRUNCATION ABLATION ---
    if args.mode in ("truncation", "both"):
        truncation_lengths = [200, 400, 800, 1600, 3000]
        for r_len in truncation_lengths:
            cond = f"trunc_r{r_len}"
            result = run_condition(
                model, processor, device, all_samples, cond, output_dir,
                r_len=r_len, q_len=1500,
            )
            all_results[cond] = result

    # --- SCRAMBLE ABLATION ---
    if args.mode in ("scramble", "both"):
        # Word-level scramble
        result = run_condition(
            model, processor, device, all_samples, "scramble_words", output_dir,
            r_len=800, q_len=1500,
            response_transform=scramble_words,
        )
        all_results["scramble_words"] = result

        # Sentence-level scramble
        result = run_condition(
            model, processor, device, all_samples, "scramble_sentences", output_dir,
            r_len=800, q_len=1500,
            response_transform=scramble_sentences,
        )
        all_results["scramble_sentences"] = result

    # Summary
    summary = {
        "description": "Truncation length and response scrambling ablations",
        "checkpoint": args.checkpoint,
        "n_samples": len(all_samples),
        "results": {k: {"auroc": v["overall"]["auroc"], "auprc": v["overall"]["auprc"]}
                    for k, v in all_results.items()},
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for cond, result in all_results.items():
        print(f"  {cond:<25} AUROC={result['overall']['auroc']:.4f}")

    # Interpretation
    if "scramble_words" in all_results:
        scramble_auroc = all_results["scramble_words"]["overall"]["auroc"]
        # Compare to r_len=800 if available, else to first truncation result
        ref_key = "trunc_r800" if "trunc_r800" in all_results else list(all_results.keys())[0]
        ref_auroc = all_results[ref_key]["overall"]["auroc"]
        drop = ref_auroc - scramble_auroc
        print(f"\n  Scramble drop (vs {ref_key}): {drop:+.4f}")
        if drop > 0.15:
            print("  STRONG: Model heavily relies on response content.")
        elif drop > 0.05:
            print("  MODERATE: Model uses content but also surface features.")
        else:
            print("  WEAK: Model may rely on surface features (length, formatting).")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
