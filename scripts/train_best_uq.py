#!/usr/bin/env python3
"""Train the best unified UQ model on ALL data (text + VLM with real images).

This trains a single Qwen3-VL-8B-Instruct model via LoRA on:
  - All GPT-5-mini benchmark predictions (~4157 samples)
  - All GPT-5.2 benchmark predictions (~4108 samples)
  - All Qwen3.5 benchmark predictions (~3077 samples)

VLM benchmarks get real images. Text benchmarks get gray placeholders.
The result is the BEST possible UQ calibrator for all modalities.

Usage:
    # Full training (4 GPUs, ~2-3 hours)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_best_uq.py \
        --output_dir uq_models/best_unified

    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/train_best_uq.py \
        --output_dir uq_models/best_unified_smoke --smoke_test

    # Prepare data only (download images, create splits)
    python scripts/train_best_uq.py --prepare_only
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# Data sources
DATA_SOURCES = {
    "gpt5mini": {"dir": "runs/gpt5_mini_combined", "type": "combined"},
    "gpt52": {"dir": "runs", "prefix": "gpt52_high_", "type": "prefixed"},
    "qwen35": {"dir": "runs", "prefix": "qwen35_397b_", "type": "prefixed"},
}

# VLM benchmarks that have images
VLM_BENCHMARKS = {
    "charxiv", "mmmu", "mmstar", "hallusionbench", "mathverse",
    "mathvision", "mathvista", "realworldqa", "vizwiz", "hle_multimodal", "mmvet",
}

# Exclude broken/trivial benchmarks
EXCLUDED = {
    "triviaqa", "babilong", "vsr", "aokvqa", "erqa", "tutorbench",
    "healthbench", "arc", "oolong",
}

IMAGE_CACHE_DIR = Path("data/training_images")

PROMPT_TEMPLATE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""


# ============================================================
# DATA LOADING
# ============================================================

@dataclass
class Sample:
    id: str
    benchmark: str
    source_model: str
    question: str
    response: str
    is_correct: bool
    has_image: bool


def extract_question_text(input_data) -> str:
    """Extract question text from input field."""
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


def load_combined_dir(combined_dir: str, source_model: str, max_per_benchmark=None):
    """Load predictions from a combined directory (like gpt5_mini_combined)."""
    samples = []
    combined_path = Path(combined_dir)

    for bench_dir in sorted(combined_path.iterdir()):
        if not bench_dir.is_dir():
            continue
        benchmark = bench_dir.name
        if benchmark in EXCLUDED:
            continue

        pred_file = bench_dir / "predictions.jsonl"
        if not pred_file.exists():
            continue

        bench_samples = []
        with open(pred_file) as f:
            for line in f:
                try:
                    pred = json.loads(line)
                except json.JSONDecodeError:
                    continue

                score = pred.get("score", -1)
                if isinstance(score, dict):
                    correct = score.get("correct", -1)
                else:
                    correct = score
                if correct not in (0, 1):
                    continue

                question = extract_question_text(pred.get("input", {}))
                response = pred.get("response_text", "") or str(pred.get("prediction", ""))
                if not question or not response:
                    continue

                bench_samples.append(Sample(
                    id=str(pred.get("id", "")),
                    benchmark=benchmark,
                    source_model=source_model,
                    question=question[:2000],
                    response=response[:1000],
                    is_correct=bool(correct == 1),
                    has_image=benchmark in VLM_BENCHMARKS,
                ))

        if max_per_benchmark and len(bench_samples) > max_per_benchmark:
            np.random.seed(42)
            idx = np.random.choice(len(bench_samples), max_per_benchmark, replace=False)
            bench_samples = [bench_samples[i] for i in idx]

        samples.extend(bench_samples)

    return samples


def load_prefixed_runs(runs_dir: str, prefix: str, source_model: str, max_per_benchmark=None):
    """Load predictions from prefixed run directories."""
    samples = []
    runs_path = Path(runs_dir)

    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.name.startswith(prefix):
            continue
        benchmark = run_dir.name[len(prefix):]
        if benchmark in EXCLUDED:
            continue

        pred_file = run_dir / "predictions.jsonl"
        if not pred_file.exists():
            continue

        bench_samples = []
        with open(pred_file) as f:
            for line in f:
                try:
                    pred = json.loads(line)
                except json.JSONDecodeError:
                    continue

                score = pred.get("score", {})
                if isinstance(score, dict):
                    correct = score.get("correct", -1)
                else:
                    correct = score
                if correct not in (0, 1):
                    continue

                question = extract_question_text(pred.get("input", {}))
                response = pred.get("response_text", "")
                if not response:
                    response = str(pred.get("prediction", {}).get("answer", ""))
                if not question or not response:
                    continue

                bench_samples.append(Sample(
                    id=str(pred.get("id", "")),
                    benchmark=benchmark,
                    source_model=source_model,
                    question=question[:2000],
                    response=response[:1000],
                    is_correct=bool(correct == 1),
                    has_image=benchmark in VLM_BENCHMARKS,
                ))

        if max_per_benchmark and len(bench_samples) > max_per_benchmark:
            np.random.seed(42)
            idx = np.random.choice(len(bench_samples), max_per_benchmark, replace=False)
            bench_samples = [bench_samples[i] for i in idx]

        samples.extend(bench_samples)

    return samples


def load_all_samples(max_per_benchmark=None):
    """Load samples from all data sources."""
    all_samples = []

    for model_name, config in DATA_SOURCES.items():
        print(f"\nLoading {model_name}...")
        if config["type"] == "combined":
            samples = load_combined_dir(config["dir"], model_name, max_per_benchmark)
        else:
            samples = load_prefixed_runs(
                config["dir"], config["prefix"], model_name, max_per_benchmark
            )
        print(f"  {model_name}: {len(samples)} samples")
        all_samples.extend(samples)

    return all_samples


# ============================================================
# IMAGE DOWNLOADING
# ============================================================

def decode_data_url_to_pil(data_url: str):
    if not data_url or not data_url.startswith("data:"):
        return None
    try:
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None


def download_benchmark_images(benchmark: str, needed_ids: set, cache_dir: Path):
    """Download and cache images for a benchmark."""
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    already_cached = 0
    downloaded = 0
    failed = 0

    # Check what's already cached
    remaining_ids = set()
    for sid in needed_ids:
        cache_path = cache_dir / f"{sid}.jpg"
        if cache_path.exists():
            already_cached += 1
        else:
            remaining_ids.add(sid)

    if not remaining_ids:
        print(f"  {benchmark}: {already_cached} already cached, 0 to download")
        return already_cached

    print(f"  {benchmark}: {already_cached} cached, {len(remaining_ids)} to download...")
    t0 = time.time()

    try:
        if benchmark == "charxiv":
            ds = load_dataset("princeton-nlp/CharXiv", split="validation")
            for idx, row in enumerate(ds):
                sid = f"charxiv_{idx}"
                if sid in remaining_ids:
                    img = row.get("image")
                    if isinstance(img, Image.Image):
                        img.convert("RGB").save(cache_dir / f"{sid}.jpg")
                        downloaded += 1
                    else:
                        failed += 1

        elif benchmark == "mmmu":
            SUBJECTS = [
                "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
                "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
                "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
                "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage",
                "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy",
                "Physics", "Psychology", "Public_Health", "Sociology"
            ]
            for mmmu_split in ["dev", "validation"]:
                for subject in SUBJECTS:
                    if downloaded + failed >= len(remaining_ids):
                        break
                    try:
                        ds = load_dataset("MMMU/MMMU", subject, split=mmmu_split)
                    except Exception:
                        continue
                    for row in ds:
                        sid = row.get("id", "")
                        if sid in remaining_ids:
                            for i in range(1, 8):
                                img = row.get(f"image_{i}")
                                if isinstance(img, Image.Image):
                                    img.convert("RGB").save(cache_dir / f"{sid}.jpg")
                                    downloaded += 1
                                    break
                            else:
                                failed += 1

        elif benchmark == "mmstar":
            ds = load_dataset("Lin-Chen/MMStar", split="val")
            for idx, row in enumerate(ds):
                sid = str(row.get("index", idx))
                if sid in remaining_ids:
                    img = row.get("image")
                    if isinstance(img, Image.Image):
                        img.convert("RGB").save(cache_dir / f"{sid}.jpg")
                        downloaded += 1

        elif benchmark == "hallusionbench":
            ds = load_dataset("lmms-lab/HallusionBench", split="image")
            for idx, row in enumerate(ds):
                sid = str(row.get("id", idx))
                if sid in remaining_ids:
                    img = row.get("image")
                    if isinstance(img, Image.Image):
                        img.convert("RGB").save(cache_dir / f"{sid}.jpg")
                        downloaded += 1

        elif benchmark == "mathverse":
            ds = load_dataset("AI4Math/MathVerse", "testmini", split="testmini")
            for idx, row in enumerate(ds):
                sid = str(row.get("sample_index", idx))
                if sid in remaining_ids:
                    img = row.get("image") or row.get("decoded_image")
                    if isinstance(img, Image.Image):
                        img.convert("RGB").save(cache_dir / f"{sid}.jpg")
                        downloaded += 1

        elif benchmark == "mathvision":
            try:
                ds = load_dataset("MathLLMs/MathVision", split="testmini")
            except Exception:
                ds = load_dataset("MathLLMs/MathVision", split="test")
            for idx, row in enumerate(ds):
                sid = str(row.get("id", idx))
                if sid in remaining_ids:
                    img = row.get("decoded_image") or row.get("image")
                    if isinstance(img, Image.Image):
                        img.convert("RGB").save(cache_dir / f"{sid}.jpg")
                        downloaded += 1

        elif benchmark == "mathvista":
            ds = load_dataset("AI4Math/MathVista", split="testmini")
            for idx, row in enumerate(ds):
                sid = str(row.get("pid", row.get("id", "")))
                if sid in remaining_ids:
                    img = row.get("decoded_image") or row.get("image")
                    if isinstance(img, Image.Image):
                        img.convert("RGB").save(cache_dir / f"{sid}.jpg")
                        downloaded += 1

        elif benchmark == "realworldqa":
            ds = load_dataset("xai-org/RealworldQA", split="test")
            for idx, row in enumerate(ds):
                sid = str(idx)
                if sid in remaining_ids:
                    img = row.get("image")
                    if isinstance(img, Image.Image):
                        img.convert("RGB").save(cache_dir / f"{sid}.jpg")
                        downloaded += 1

        elif benchmark == "vizwiz":
            ds = load_dataset("lmms-lab/VizWiz-VQA", split="val")
            for idx, row in enumerate(ds):
                sid = str(idx)
                if sid in remaining_ids:
                    img = row.get("image")
                    if isinstance(img, Image.Image):
                        img.convert("RGB").save(cache_dir / f"{sid}.jpg")
                        downloaded += 1

        elif benchmark == "hle_multimodal":
            ds = load_dataset("cais/hle", split="test")
            for idx, row in enumerate(ds):
                sid = f"hle_{idx}"
                if sid in remaining_ids:
                    img_data = row.get("image", "")
                    if img_data and isinstance(img_data, str) and img_data.startswith("data:"):
                        img = decode_data_url_to_pil(img_data)
                        if img is not None:
                            img.convert("RGB").save(cache_dir / f"{sid}.jpg")
                            downloaded += 1

        elif benchmark == "mmvet":
            ds = load_dataset("lmms-lab/MMVet", split="test")
            for idx, row in enumerate(ds):
                sid = str(row.get("question_id", idx))
                if sid in remaining_ids:
                    img = row.get("image")
                    if isinstance(img, Image.Image):
                        img.convert("RGB").save(cache_dir / f"{sid}.jpg")
                        downloaded += 1

    except Exception as e:
        print(f"  WARNING: {benchmark} download error: {e}")

    elapsed = time.time() - t0
    total = already_cached + downloaded
    print(f"  {benchmark}: {downloaded} downloaded, {failed} failed ({elapsed:.1f}s). "
          f"Total cached: {total}/{len(needed_ids)}")
    return total


def prepare_all_images(samples: list):
    """Download and cache all images needed for VLM benchmarks."""
    needed = defaultdict(set)
    for s in samples:
        if s.has_image:
            needed[s.benchmark].add(s.id)

    if not needed:
        print("No VLM samples found.")
        return

    total_needed = sum(len(ids) for ids in needed.values())
    print(f"\nPreparing images for {total_needed} VLM samples across {len(needed)} benchmarks...")

    total_cached = 0
    for bench, ids in sorted(needed.items()):
        cache_dir = IMAGE_CACHE_DIR / bench
        cached = download_benchmark_images(bench, ids, cache_dir)
        total_cached += cached

    print(f"\nImage preparation complete: {total_cached}/{total_needed} images available")


# ============================================================
# TRAINING DATASET
# ============================================================

PROMPT_TEMPLATES = {
    "baseline": PROMPT_TEMPLATE,
    "combined": """Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes""",
}

TRUNCATION_LENGTHS = {
    "baseline": (500, 300),
    "combined": (1500, 800),
}


class UnifiedUQDataset(torch.utils.data.Dataset):
    """Unified dataset for text + VLM UQ training."""

    def __init__(self, samples: list[Sample], processor, max_length=2048,
                 prompt_variant="baseline"):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.prompt_variant = prompt_variant
        self.prompt_template = PROMPT_TEMPLATES[prompt_variant]
        q_len, r_len = TRUNCATION_LENGTHS[prompt_variant]
        self.q_truncation = q_len
        self.r_truncation = r_len
        self.fallback_image = Image.new('RGB', (336, 336), color='gray')
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 512 * 28 * 28
        # Dynamic assistant token lookup (varies across model families)
        assistant_ids = self.processor.tokenizer.encode("assistant", add_special_tokens=False)
        self.assistant_token = assistant_ids[-1] if assistant_ids else 77091

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        if sample.has_image:
            cache_path = IMAGE_CACHE_DIR / sample.benchmark / f"{sample.id}.jpg"
            if cache_path.exists():
                try:
                    image = Image.open(cache_path).convert("RGB")
                except Exception:
                    image = self.fallback_image
            else:
                image = self.fallback_image
            min_px = self.min_pixels
            max_px = self.max_pixels
        else:
            image = self.fallback_image
            min_px = 256 * 28 * 28
            max_px = 256 * 28 * 28

        # Target
        target = "ii" if sample.is_correct else "i"

        fmt_kwargs = {
            "question": sample.question[:self.q_truncation],
            "response": sample.response[:self.r_truncation],
        }
        if self.prompt_variant == "combined":
            fmt_kwargs["benchmark"] = sample.benchmark
            fmt_kwargs["source_model"] = sample.source_model
        prompt = self.prompt_template.format(**fmt_kwargs)

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
            {"role": "assistant", "content": target},
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            min_pixels=min_px,
            max_pixels=max_px,
        )

        # Labels - mask everything except the answer token
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        # Find the target answer token after the last assistant marker
        # Works for both Qwen3-VL (assistant\nii) and Qwen3.5 (assistant\n<think>\n\n</think>\n\nii)
        target_token_id = self.processor.tokenizer.encode(target, add_special_tokens=False)[-1]
        assistant_positions = (input_ids == self.assistant_token).nonzero(as_tuple=True)[0]
        if len(assistant_positions) > 0:
            search_start = assistant_positions[-1].item()
            # Find target token after the assistant marker
            answer_pos = None
            for pos in range(search_start, len(input_ids)):
                if input_ids[pos].item() == target_token_id:
                    answer_pos = pos
                    break
            if answer_pos is not None:
                labels[:] = -100
                labels[answer_pos] = input_ids[answer_pos]
            else:
                # Fallback: use original offset logic
                answer_pos = search_start + 2
                labels[:] = -100
                if answer_pos < len(labels):
                    labels[answer_pos] = input_ids[answer_pos]
        else:
            labels[:-3] = -100

        result = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            if isinstance(pv, list):
                result["pixel_values"] = pv[0] if len(pv) > 0 else pv
            else:
                result["pixel_values"] = pv.squeeze(0) if pv.dim() > 3 else pv

        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]

        return result


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, processor, test_samples: list[Sample], device,
                   prompt_variant="baseline"):
    """Evaluate trained model on test set."""
    template = PROMPT_TEMPLATES[prompt_variant]
    q_len, r_len = TRUNCATION_LENGTHS[prompt_variant]
    model.eval()
    fallback = Image.new('RGB', (224, 224), color='gray')

    all_preds, all_labels = [], []
    per_benchmark = defaultdict(lambda: {"preds": [], "labels": []})

    for i, sample in enumerate(test_samples):
        if i % 100 == 0:
            print(f"  Eval {i}/{len(test_samples)}...")

        # Load image
        if sample.has_image:
            cache_path = IMAGE_CACHE_DIR / sample.benchmark / f"{sample.id}.jpg"
            if cache_path.exists():
                try:
                    image = Image.open(cache_path).convert("RGB")
                except Exception:
                    image = fallback
            else:
                image = fallback
        else:
            image = fallback

        fmt_kwargs = {
            "question": sample.question[:q_len],
            "response": sample.response[:r_len],
        }
        if prompt_variant == "combined":
            fmt_kwargs["benchmark"] = sample.benchmark
            fmt_kwargs["source_model"] = sample.source_model
        prompt = template.format(**fmt_kwargs)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[image], return_tensors="pt", padding=True,
            min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
            token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
            probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
            p_correct = probs[1].item()
        except Exception as e:
            if i < 5:
                print(f"  Error: {e}")
            p_correct = 0.5

        all_preds.append(p_correct)
        all_labels.append(float(sample.is_correct))
        per_benchmark[sample.benchmark]["preds"].append(p_correct)
        per_benchmark[sample.benchmark]["labels"].append(float(sample.is_correct))

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    results = {
        "auroc": float(roc_auc_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
        "auprc": float(average_precision_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
        "brier": float(brier_score_loss(labels, preds)),
        "ece": 0.0,
        "n_samples": len(labels),
        "n_correct": int(sum(labels)),
    }

    # ECE
    bin_boundaries = np.linspace(0, 1, 11)
    for j in range(10):
        in_bin = (preds >= bin_boundaries[j]) & (preds < bin_boundaries[j + 1])
        if in_bin.sum() == 0:
            continue
        results["ece"] += float((in_bin.sum() / len(preds)) * abs(labels[in_bin].mean() - preds[in_bin].mean()))

    # Per-benchmark
    results["per_benchmark"] = {}
    for bench, data in sorted(per_benchmark.items()):
        bp, bl = np.array(data["preds"]), np.array(data["labels"])
        entry = {"n_samples": len(bl), "accuracy": float(bl.mean()), "is_vlm": bench in VLM_BENCHMARKS}
        if len(set(bl)) > 1:
            entry["auroc"] = float(roc_auc_score(bl, bp))
        results["per_benchmark"][bench] = entry

    # VLM vs text aggregate
    vlm_p, vlm_l, txt_p, txt_l = [], [], [], []
    for bench, data in per_benchmark.items():
        if bench in VLM_BENCHMARKS:
            vlm_p.extend(data["preds"])
            vlm_l.extend(data["labels"])
        else:
            txt_p.extend(data["preds"])
            txt_l.extend(data["labels"])
    if vlm_l and len(set(vlm_l)) > 1:
        results["vlm_auroc"] = float(roc_auc_score(vlm_l, vlm_p))
    if txt_l and len(set(txt_l)) > 1:
        results["text_auroc"] = float(roc_auc_score(txt_l, txt_p))

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train best unified UQ model")
    parser.add_argument("--output_dir", type=str, default="uq_models/best_unified")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--prepare_only", action="store_true",
                        help="Only prepare data and download images, don't train")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="LoRA alpha (default: 2*lora_r)")
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Max training samples (for size ablation)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_variant", choices=["baseline", "combined"],
                        default="baseline",
                        help="Prompt template: baseline (v1) or combined (v2)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None,
                        help="Override base model (e.g., Qwen/Qwen3.5-9B)")
    parser.add_argument("--split_info", type=str, default=None,
                        help="Path to split_info.json to reuse exact train/test split")
    args = parser.parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r

    max_per_bench = 5 if args.smoke_test else None

    base_model = args.base_model or MODEL_NAME

    print("=" * 70)
    print("TRAINING BEST UNIFIED UQ MODEL")
    print("=" * 70)
    print(f"Model: {base_model}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}, LR: {args.learning_rate}, LoRA r: {args.lora_r}")
    print()

    # --- Step 1: Load all data ---
    print("STEP 1: Loading all prediction data...")
    all_samples = load_all_samples(max_per_benchmark=max_per_bench)

    # Stats
    by_model = defaultdict(int)
    by_bench = defaultdict(int)
    n_vlm = 0
    for s in all_samples:
        by_model[s.source_model] += 1
        by_bench[s.benchmark] += 1
        if s.has_image:
            n_vlm += 1

    print(f"\nTotal: {len(all_samples)} samples ({n_vlm} VLM, {len(all_samples)-n_vlm} text)")
    for m, c in sorted(by_model.items()):
        print(f"  {m}: {c}")
    print(f"Benchmarks: {len(by_bench)}")
    n_correct = sum(1 for s in all_samples if s.is_correct)
    print(f"Correct: {n_correct} ({100*n_correct/len(all_samples):.1f}%)")

    # --- Step 2: Download images ---
    print(f"\nSTEP 2: Preparing images...")
    prepare_all_images(all_samples)

    # --- Step 3: Train/test split ---
    if args.split_info:
        print(f"\nSTEP 3: Loading existing split from {args.split_info}...")
        with open(args.split_info) as f:
            existing_split = json.load(f)

        test_id_set = set(existing_split["test_ids"])
        train_id_set = set(existing_split["train_ids"])

        train_samples, test_samples = [], []
        unmatched = 0
        for s in all_samples:
            if s.id in test_id_set:
                test_samples.append(s)
            elif s.id in train_id_set:
                train_samples.append(s)
            else:
                train_samples.append(s)
                unmatched += 1

        print(f"  Loaded split: {len(train_samples)} train, {len(test_samples)} test")
        if unmatched:
            print(f"  ({unmatched} new samples added to training set)")
        # Verify no leakage
        test_ids_in_train = set(s.id for s in test_samples) & set(s.id for s in train_samples)
        if test_ids_in_train:
            print(f"  WARNING: {len(test_ids_in_train)} IDs appear in both train and test!")
    else:
        print(f"\nSTEP 3: Creating train/test split ({1-args.test_fraction:.0%}/{args.test_fraction:.0%})...")
        # Stratify by benchmark for balanced split
        strat_key = [s.benchmark for s in all_samples]
        strat_counts = defaultdict(int)
        for k in strat_key:
            strat_counts[k] += 1
        # Replace rare strata (< 3 samples) with "other" to avoid split errors
        min_test = max(2, int(len(all_samples) * args.test_fraction * 0.5))
        strat_key_safe = [k if strat_counts[k] >= 3 else "other" for k in strat_key]
        # If still too many classes for test size, fall back to no stratification
        n_classes = len(set(strat_key_safe))
        n_test = max(1, int(len(all_samples) * args.test_fraction))

        # Group samples by question ID so all source-model variants of the same
        # question end up in the same split (prevents question-level leakage).
        from collections import OrderedDict
        qid_to_indices = OrderedDict()
        for i, s in enumerate(all_samples):
            qid_to_indices.setdefault(s.id, []).append(i)

        unique_qids = list(qid_to_indices.keys())
        # Stratify by the benchmark of the first sample for each question
        qid_strat = []
        for qid in unique_qids:
            bench = all_samples[qid_to_indices[qid][0]].benchmark
            qid_strat.append(bench if strat_counts.get(bench, 0) >= 3 else "other")

        try:
            train_qids, test_qids = train_test_split(
                unique_qids, test_size=args.test_fraction, random_state=42, stratify=qid_strat
            )
        except ValueError:
            train_qids, test_qids = train_test_split(
                unique_qids, test_size=args.test_fraction, random_state=42
            )

        train_qid_set = set(train_qids)
        test_qid_set = set(test_qids)
        train_samples = [all_samples[i] for qid in train_qids for i in qid_to_indices[qid]]
        test_samples = [all_samples[i] for qid in test_qids for i in qid_to_indices[qid]]

        # Verify zero question-level overlap
        overlap = train_qid_set & test_qid_set
        assert len(overlap) == 0, f"Question-level leakage: {len(overlap)} shared question IDs!"
        print(f"  Question-level split: {len(train_qids)} train questions, {len(test_qids)} test questions, 0 overlap")

    # Subsample training data for size ablation
    if args.max_train_samples and args.max_train_samples < len(train_samples):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(train_samples), args.max_train_samples, replace=False)
        train_samples = [train_samples[i] for i in sorted(idx)]
        print(f"Subsampled to {len(train_samples)} training samples (seed={args.seed})")

    n_train_vlm = sum(1 for s in train_samples if s.has_image)
    n_test_vlm = sum(1 for s in test_samples if s.has_image)
    print(f"Train: {len(train_samples)} ({n_train_vlm} VLM, {len(train_samples)-n_train_vlm} text)")
    print(f"Test:  {len(test_samples)} ({n_test_vlm} VLM, {len(test_samples)-n_test_vlm} text)")

    # Save split info
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Unique question IDs (no duplicates across source models)
    train_question_ids = sorted(set(s.id for s in train_samples))
    test_question_ids = sorted(set(s.id for s in test_samples))
    question_overlap = set(train_question_ids) & set(test_question_ids)
    if question_overlap:
        print(f"  FATAL: {len(question_overlap)} question IDs in both train and test!")
        sys.exit(1)

    split_info = {
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "n_train_vlm": n_train_vlm,
        "n_test_vlm": n_test_vlm,
        "n_train_questions": len(train_question_ids),
        "n_test_questions": len(test_question_ids),
        "question_overlap": len(question_overlap),
        "split_method": "question_level",
        "train_ids": [s.id for s in train_samples],
        "test_ids": [s.id for s in test_samples],
        "train_question_ids": train_question_ids,
        "test_question_ids": test_question_ids,
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f)

    if args.prepare_only:
        print("\n--prepare_only: Stopping before training.")
        return

    # --- Step 4: Load model ---
    print(f"\nSTEP 4: Loading {base_model}...")
    from transformers import (
        AutoModelForImageTextToText, AutoProcessor, AutoConfig,
        TrainingArguments, Trainer, TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    num_gpus = torch.cuda.device_count()
    max_memory = {i: "78GiB" for i in range(num_gpus)}
    print(f"Using {num_gpus} GPUs")

    # Determine model class: AutoModel doesn't always dispatch new model types
    model_cls = AutoModelForImageTextToText
    if "qwen3.5" in base_model.lower() or "qwen3_5" in base_model.lower():
        try:
            from transformers import Qwen3_5ForConditionalGeneration
            model_cls = Qwen3_5ForConditionalGeneration
            print(f"  Using Qwen3_5ForConditionalGeneration")
        except ImportError:
            print(f"  WARNING: Qwen3_5ForConditionalGeneration not available, using AutoModel")

    model = model_cls.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Step 5: Create datasets ---
    print("\nSTEP 5: Creating datasets...")
    train_dataset = UnifiedUQDataset(train_samples, processor,
                                     prompt_variant=args.prompt_variant)

    # Collator
    def collate_fn(batch):
        max_len = max(x["input_ids"].size(0) for x in batch)
        pad_token_id = processor.tokenizer.pad_token_id or 0

        input_ids, attention_mask, labels_list = [], [], []
        for x in batch:
            seq_len = x["input_ids"].size(0)
            pad_len = max_len - seq_len
            input_ids.append(torch.cat([
                torch.full((pad_len,), pad_token_id, dtype=x["input_ids"].dtype),
                x["input_ids"]
            ]))
            attention_mask.append(torch.cat([
                torch.zeros(pad_len, dtype=x["attention_mask"].dtype),
                x["attention_mask"]
            ]))
            labels_list.append(torch.cat([
                torch.full((pad_len,), -100, dtype=x["labels"].dtype),
                x["labels"]
            ]))

        result = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels_list),
        }
        if "pixel_values" in batch[0]:
            result["pixel_values"] = torch.cat([x["pixel_values"] for x in batch], dim=0)
        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in batch])
        return result

    # --- Step 6: Train ---
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        bf16_full_eval=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
    )

    class MemoryCleanupCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 100 == 0:
                torch.cuda.empty_cache()
            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[MemoryCleanupCallback()],
    )

    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)
    eff_batch = args.batch_size * args.grad_accum * max(num_gpus, 1)
    print(f"Epochs: {args.epochs}, Effective batch: {eff_batch}")
    print(f"Steps/epoch: ~{len(train_dataset) // eff_batch}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save
    print(f"\nSaving to {output_dir}...")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    # --- Step 7: Evaluate ---
    print("\n" + "=" * 50)
    print("EVALUATING ON TEST SET")
    print("=" * 50)

    device = next(model.parameters()).device
    results = evaluate_model(model, processor, test_samples, device,
                             prompt_variant=args.prompt_variant)

    print(f"\nOverall AUROC: {results['auroc']:.4f}")
    print(f"VLM AUROC:     {results.get('vlm_auroc', 'N/A')}")
    print(f"Text AUROC:    {results.get('text_auroc', 'N/A')}")
    print(f"ECE:           {results['ece']:.4f}")
    print(f"Brier:         {results['brier']:.4f}")

    print("\nPer-benchmark:")
    for bench, data in sorted(results["per_benchmark"].items(),
                               key=lambda x: x[1].get("auroc", 0), reverse=True):
        auroc = data.get("auroc", "N/A")
        vlm_tag = " [VLM]" if data.get("is_vlm") else ""
        if isinstance(auroc, float):
            print(f"  {bench:<20} AUROC={auroc:.3f} (n={data['n_samples']}){vlm_tag}")
        else:
            print(f"  {bench:<20} {auroc} (n={data['n_samples']}){vlm_tag}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
