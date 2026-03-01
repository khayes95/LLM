#!/usr/bin/env python3
"""Cross-model evaluation of VLM judge on existing prediction data.

Evaluates the VSR-fixed VLM judge (Qwen3-VL-8B + LoRA) on responses
from a target model. Supports real images for VLM benchmarks (default)
or gray placeholder images for text-only evaluation.

Supports four data sources:
  1. Qwen3-VL-30B predictions (from runs/)
  2. GPT-5-mini combined dataset (from runs/gpt5_mini_combined/)
  3. GPT-5.2 high-reasoning predictions (from runs/gpt52_high_*)
  4. Qwen3.5-397B predictions (from runs/qwen35_397b_*)

Usage:
    # With real images (default)
    CUDA_VISIBLE_DEVICES=0 python scripts/vlm_judge_cross_model_eval.py \
        --target gpt52 --output data/cross_model/vlm_judge_images_on_gpt52.json

    # Text-only (gray placeholder)
    CUDA_VISIBLE_DEVICES=0 python scripts/vlm_judge_cross_model_eval.py \
        --target gpt52 --no_images

    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/vlm_judge_cross_model_eval.py \
        --target gpt52 --smoke_test
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# CONFIG
# ============================================================

VLM_JUDGE_CHECKPOINT = "data/vlm_judge_vsr_fixed"
VLM_JUDGE_BASE = "Qwen/Qwen3-VL-8B-Instruct"
GPT5_COMBINED = "runs/gpt5_mini_combined"

# Best Qwen3-VL-30B run directories (Jan 3 + overnight Feb 21)
QWEN3_RUNS = {
    "gpqa": "runs/20260103_170932_full_Qwen3_VL_30B_A3B_Thinking/gpqa",
    "hallusionbench": "runs/20260103_170932_full_Qwen3_VL_30B_A3B_Thinking/hallusionbench",
    "chembench": "runs/20260103_170932_full_Qwen3_VL_30B_A3B_Thinking/chembench",
    "mmmu": "runs/20260103_170933_full_Qwen3_VL_30B_A3B_Thinking/mmmu",
    "hle": "runs/20260103_170933_full_Qwen3_VL_30B_A3B_Thinking/hle",
    "mathvision": "runs/20260103_130359_full_Qwen3_VL_30B_A3B_Thinking/mathvision",
    "mathverse": "runs/20260103_130359_full_Qwen3_VL_30B_A3B_Thinking/mathverse",
    "mathvista": "runs/20260103_130359_full_Qwen3_VL_30B_A3B_Thinking/mathvista",
    "mmstar": "runs/20260103_130359_full_Qwen3_VL_30B_A3B_Thinking/mmstar",
    "realworldqa": "runs/20260103_130359_full_Qwen3_VL_30B_A3B_Thinking/realworldqa",
    "vizwiz": "runs/20260103_130359_full_Qwen3_VL_30B_A3B_Thinking/vizwiz",
    "hle_multimodal": "runs/20260103_130359_full_Qwen3_VL_30B_A3B_Thinking/hle_multimodal",
    "charxiv": "runs/20260103_214937_charxiv_Qwen3_VL_30B_A3B_Thinking/charxiv",
    "mmvet": "runs/20260103_214252_mmvet_Qwen3_VL_30B_A3B_Thinking/mmvet",
    # Overnight Feb 21 runs (missing benchmarks)
    "omnimath": "runs/qwen3vl_30b_overnight/omnimath",
    "bbeh": "runs/qwen3vl_30b_overnight/bbeh",
    "livebench": "runs/qwen3vl_30b_overnight/livebench",
    "simpleqa": "runs/qwen3vl_30b_overnight/simpleqa",
}

# Benchmarks to exclude (too easy, broken, etc.)
EXCLUDED = {"triviaqa", "babilong", "vsr", "aokvqa", "erqa", "tutorbench", "healthbench", "arc", "oolong"}

# VLM benchmarks (have images in HuggingFace datasets)
VLM_BENCHMARKS = {
    "charxiv", "mmmu", "mmstar", "hallusionbench", "mathverse",
    "mathvision", "mathvista", "realworldqa", "vizwiz", "hle_multimodal", "mmvet",
}

PROMPT_TEMPLATE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""


# ============================================================
# IMAGE LOADING FROM HUGGINGFACE
# ============================================================

def decode_data_url_to_pil(data_url: str):
    """Decode a data:image/... URL to a PIL Image."""
    if not data_url or not data_url.startswith("data:"):
        return None
    try:
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None


def load_benchmark_images(benchmark: str, needed_ids: set) -> dict:
    """Load images from HuggingFace dataset for a given benchmark.

    Returns dict mapping sample_id -> PIL.Image
    """
    from datasets import load_dataset

    images = {}
    print(f"  Loading {benchmark} images from HuggingFace...")
    t0 = time.time()

    try:
        if benchmark == "charxiv":
            ds = load_dataset("princeton-nlp/CharXiv", split="validation")
            for idx, row in enumerate(ds):
                sid = f"charxiv_{idx}"
                if sid in needed_ids:
                    img = row.get("image")
                    if isinstance(img, Image.Image):
                        images[sid] = img.convert("RGB")

        elif benchmark == "mmmu":
            MMMU_SUBJECTS = [
                "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
                "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
                "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
                "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage",
                "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy",
                "Physics", "Psychology", "Public_Health", "Sociology"
            ]
            # IDs contain the split prefix (e.g., "dev_Basic_Medical_Science_4")
            # Try both dev and validation splits
            for mmmu_split in ["dev", "validation"]:
                for subject in MMMU_SUBJECTS:
                    if len(images) >= len(needed_ids):
                        break
                    try:
                        ds = load_dataset("MMMU/MMMU", subject, split=mmmu_split)
                    except Exception:
                        continue
                    for row in ds:
                        sid = row.get("id", "")
                        if sid in needed_ids:
                            # MMMU can have multiple images; use first available
                            for i in range(1, 8):
                                img = row.get(f"image_{i}")
                                if isinstance(img, Image.Image):
                                    images[sid] = img.convert("RGB")
                                    break

        elif benchmark == "mmstar":
            ds = load_dataset("Lin-Chen/MMStar", split="val")
            for idx, row in enumerate(ds):
                sid = str(row.get("index", idx))
                if sid in needed_ids:
                    img = row.get("image")
                    if isinstance(img, Image.Image):
                        images[sid] = img.convert("RGB")

        elif benchmark == "hallusionbench":
            ds = load_dataset("lmms-lab/HallusionBench", split="image")
            for idx, row in enumerate(ds):
                sid = str(row.get("id", idx))
                if sid in needed_ids:
                    img = row.get("image")
                    if isinstance(img, Image.Image):
                        images[sid] = img.convert("RGB")

        elif benchmark == "mathverse":
            ds = load_dataset("AI4Math/MathVerse", "testmini", split="testmini")
            for idx, row in enumerate(ds):
                sid = str(row.get("sample_index", idx))
                if sid in needed_ids:
                    img = row.get("image") or row.get("decoded_image")
                    if isinstance(img, Image.Image):
                        images[sid] = img.convert("RGB")

        elif benchmark == "mathvision":
            try:
                ds = load_dataset("MathLLMs/MathVision", split="testmini")
            except Exception:
                ds = load_dataset("MathLLMs/MathVision", split="test")
            for idx, row in enumerate(ds):
                sid = str(row.get("id", idx))
                if sid in needed_ids:
                    img = row.get("decoded_image") or row.get("image")
                    if isinstance(img, Image.Image):
                        images[sid] = img.convert("RGB")

        elif benchmark == "mathvista":
            ds = load_dataset("AI4Math/MathVista", split="testmini")
            for idx, row in enumerate(ds):
                sid = str(row.get("pid", row.get("id", "")))
                if sid in needed_ids:
                    img = row.get("decoded_image") or row.get("image")
                    if isinstance(img, Image.Image):
                        images[sid] = img.convert("RGB")

        elif benchmark == "realworldqa":
            ds = load_dataset("xai-org/RealworldQA", split="test")
            for idx, row in enumerate(ds):
                sid = str(idx)
                if sid in needed_ids:
                    img = row.get("image")
                    if isinstance(img, Image.Image):
                        images[sid] = img.convert("RGB")

        elif benchmark == "vizwiz":
            ds = load_dataset("lmms-lab/VizWiz-VQA", split="val")
            for idx, row in enumerate(ds):
                sid = str(idx)
                if sid in needed_ids:
                    img = row.get("image")
                    if isinstance(img, Image.Image):
                        images[sid] = img.convert("RGB")

        elif benchmark == "hle_multimodal":
            ds = load_dataset("cais/hle", split="test")
            for idx, row in enumerate(ds):
                sid = f"hle_{idx}"
                if sid in needed_ids:
                    img_data = row.get("image", "")
                    if img_data and isinstance(img_data, str) and img_data.startswith("data:"):
                        img = decode_data_url_to_pil(img_data)
                        if img is not None:
                            images[sid] = img.convert("RGB")

        elif benchmark == "mmvet":
            ds = load_dataset("lmms-lab/MMVet", split="test")
            for idx, row in enumerate(ds):
                sid = str(row.get("question_id", idx))
                if sid in needed_ids:
                    img = row.get("image")
                    if isinstance(img, Image.Image):
                        images[sid] = img.convert("RGB")

    except Exception as e:
        print(f"  WARNING: Failed to load {benchmark} images: {e}")

    elapsed = time.time() - t0
    print(f"  Loaded {len(images)}/{len(needed_ids)} images for {benchmark} ({elapsed:.1f}s)")
    return images


def load_all_images(samples: list) -> dict:
    """Load images for all VLM benchmark samples.

    Returns dict mapping (benchmark, sample_id) -> PIL.Image
    """
    # Collect needed IDs per benchmark
    needed = defaultdict(set)
    for s in samples:
        if s["benchmark"] in VLM_BENCHMARKS:
            needed[s["benchmark"]].add(s["id"])

    if not needed:
        print("No VLM benchmark samples found — all text-only.")
        return {}

    print(f"\nLoading real images for {sum(len(v) for v in needed.values())} VLM samples "
          f"across {len(needed)} benchmarks...")

    image_cache = {}
    for bench, ids in sorted(needed.items()):
        bench_images = load_benchmark_images(bench, ids)
        for sid, img in bench_images.items():
            image_cache[(bench, sid)] = img

    total = sum(len(v) for v in needed.values())
    found = len(image_cache)
    print(f"Image loading complete: {found}/{total} images loaded "
          f"({100*found/total:.0f}% coverage)")
    return image_cache


# ============================================================
# DATA LOADING
# ============================================================

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
        return json.dumps(clean)[:1000]
    return str(input_data)[:1000]


def load_gpt5_mini_samples(max_per_benchmark=None):
    """Load GPT-5-mini predictions as cross-model target."""
    samples = []
    stats = {}
    combined_dir = Path(GPT5_COMBINED)

    for bench_dir in sorted(combined_dir.iterdir()):
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

                bench_samples.append({
                    "id": str(pred.get("id", "")),
                    "benchmark": benchmark,
                    "question": question[:2000],
                    "response": response[:1000],
                    "is_correct": bool(correct == 1),
                })

        if max_per_benchmark and len(bench_samples) > max_per_benchmark:
            np.random.seed(42)
            indices = np.random.choice(len(bench_samples), max_per_benchmark, replace=False)
            bench_samples = [bench_samples[i] for i in indices]

        n_correct = sum(1 for s in bench_samples if s["is_correct"])
        stats[benchmark] = {
            "used": len(bench_samples),
            "correct": n_correct,
            "accuracy": n_correct / len(bench_samples) if bench_samples else 0,
        }
        print(f"  {benchmark}: {len(bench_samples)} samples ({n_correct} correct)")
        samples.extend(bench_samples)

    return samples, stats


def load_qwen3vl_samples(max_per_benchmark=None):
    """Load Qwen3-VL-30B predictions, enriched with GPT-5-mini questions."""
    samples = []
    stats = {}

    for benchmark, run_path in sorted(QWEN3_RUNS.items()):
        if benchmark in EXCLUDED:
            continue

        pred_file = Path(run_path) / "predictions.jsonl"
        if not pred_file.exists():
            print(f"  {benchmark}: not found at {run_path}")
            continue

        # Load GPT-5-mini questions for ID matching
        gpt5_questions = {}
        gpt5_pred_file = Path(GPT5_COMBINED) / benchmark / "predictions.jsonl"
        if gpt5_pred_file.exists():
            with open(gpt5_pred_file) as f:
                for line in f:
                    try:
                        p = json.loads(line)
                        sid = str(p.get("id", ""))
                        q = extract_question_text(p.get("input", {}))
                        if q:
                            gpt5_questions[sid] = q
                    except json.JSONDecodeError:
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

                sid = str(pred.get("id", ""))
                question = gpt5_questions.get(sid, "")
                if not question:
                    continue

                response = pred.get("response_text", "") or str(pred.get("prediction", ""))

                bench_samples.append({
                    "id": sid,
                    "benchmark": benchmark,
                    "question": question[:2000],
                    "response": response[:1000],
                    "is_correct": bool(correct == 1),
                })

        if max_per_benchmark and len(bench_samples) > max_per_benchmark:
            np.random.seed(42)
            indices = np.random.choice(len(bench_samples), max_per_benchmark, replace=False)
            bench_samples = [bench_samples[i] for i in indices]

        n_correct = sum(1 for s in bench_samples if s["is_correct"])
        stats[benchmark] = {
            "used": len(bench_samples),
            "correct": n_correct,
            "accuracy": n_correct / len(bench_samples) if bench_samples else 0,
        }
        print(f"  {benchmark}: {len(bench_samples)} samples ({n_correct} correct)")
        samples.extend(bench_samples)

    return samples, stats


def load_prefixed_run_samples(prefix: str, runs_dir: str = "runs", max_per_benchmark=None):
    """Load predictions from runs matching a prefix (e.g. gpt52_high_, qwen35_397b_).

    These runs store input/response/score directly in predictions.jsonl.
    """
    samples = []
    stats = {}
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

                bench_samples.append({
                    "id": str(pred.get("id", "")),
                    "benchmark": benchmark,
                    "question": question[:2000],
                    "response": response[:1000],
                    "is_correct": bool(correct == 1),
                })

        if max_per_benchmark and len(bench_samples) > max_per_benchmark:
            np.random.seed(42)
            indices = np.random.choice(len(bench_samples), max_per_benchmark, replace=False)
            bench_samples = [bench_samples[i] for i in indices]

        n_correct = sum(1 for s in bench_samples if s["is_correct"])
        stats[benchmark] = {
            "used": len(bench_samples),
            "correct": n_correct,
            "accuracy": n_correct / len(bench_samples) if bench_samples else 0,
        }
        print(f"  {benchmark}: {len(bench_samples)} samples ({n_correct} correct)")
        samples.extend(bench_samples)

    return samples, stats


# ============================================================
# VLM JUDGE EVALUATION
# ============================================================

def load_vlm_judge(checkpoint_path: str):
    """Load trained VLM judge with LoRA adapter."""
    print(f"Loading VLM judge from {checkpoint_path}...")
    processor = AutoProcessor.from_pretrained(VLM_JUDGE_BASE, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        VLM_JUDGE_BASE,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    print("VLM judge loaded.")
    return model, processor


def get_p_correct(model, processor, question: str, response: str, image: Image.Image) -> float:
    """Get P(correct) from VLM judge."""
    prompt = PROMPT_TEMPLATE.format(question=question[:500], response=response[:300])

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()


def evaluate(model, processor, samples, image_cache=None, output_path=None):
    """Run evaluation and compute metrics.

    Args:
        image_cache: dict mapping (benchmark, id) -> PIL.Image for real images.
                     If None, uses gray placeholder for all samples.
    """
    gray_image = Image.new("RGB", (224, 224), color="gray")
    all_preds = []
    all_labels = []
    per_benchmark = defaultdict(lambda: {"preds": [], "labels": []})
    n_real_images = 0
    n_gray_images = 0

    for i, sample in enumerate(samples):
        if i % 50 == 0:
            print(f"  Evaluating {i}/{len(samples)} "
                  f"(real_img={n_real_images}, gray={n_gray_images})...")

        # Select image: real if available, gray otherwise
        image = gray_image
        if image_cache is not None:
            real_img = image_cache.get((sample["benchmark"], sample["id"]))
            if real_img is not None:
                image = real_img
                n_real_images += 1
            else:
                n_gray_images += 1
        else:
            n_gray_images += 1

        try:
            p_correct = get_p_correct(
                model, processor,
                sample["question"], sample["response"], image
            )
        except Exception as e:
            print(f"  Error on sample {i} ({sample['benchmark']}/{sample['id']}): {e}")
            p_correct = 0.5

        all_preds.append(p_correct)
        all_labels.append(float(sample["is_correct"]))
        per_benchmark[sample["benchmark"]]["preds"].append(p_correct)
        per_benchmark[sample["benchmark"]]["labels"].append(float(sample["is_correct"]))

        # Save intermediate results every 200 samples
        if output_path and (i + 1) % 200 == 0:
            _save_intermediate(all_preds, all_labels, output_path, i + 1)

    print(f"  Done: {n_real_images} real images, {n_gray_images} gray placeholders")
    results = compute_metrics(all_preds, all_labels, per_benchmark)
    results["n_real_images"] = n_real_images
    results["n_gray_images"] = n_gray_images
    return results


def _save_intermediate(preds, labels, output_path, n_done):
    """Save intermediate results."""
    try:
        interim = {
            "n_evaluated": n_done,
            "auroc": roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5,
        }
        interim_path = Path(output_path).with_suffix(".interim.json")
        with open(interim_path, "w") as f:
            json.dump(interim, f, indent=2)
    except Exception:
        pass


def compute_metrics(all_preds, all_labels, per_benchmark):
    """Compute all evaluation metrics."""
    preds = np.array(all_preds)
    labels = np.array(all_labels)

    results = {
        "auroc": float(roc_auc_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
        "auprc": float(average_precision_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
        "brier": float(brier_score_loss(labels, preds)),
        "n_samples": len(labels),
        "n_correct": int(sum(labels)),
        "base_rate": float(sum(labels) / len(labels)),
    }

    # ECE
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for j in range(n_bins):
        in_bin = (preds >= bin_boundaries[j]) & (preds < bin_boundaries[j + 1])
        if in_bin.sum() == 0:
            continue
        ece += (in_bin.sum() / len(preds)) * abs(labels[in_bin].mean() - preds[in_bin].mean())
    results["ece"] = float(ece)

    # Per-benchmark
    results["per_benchmark"] = {}
    for bench, data in sorted(per_benchmark.items()):
        bp = np.array(data["preds"])
        bl = np.array(data["labels"])
        entry = {
            "n_samples": len(bl),
            "n_correct": int(sum(bl)),
            "accuracy": float(sum(bl) / len(bl)) if len(bl) > 0 else 0,
            "is_vlm": bench in VLM_BENCHMARKS,
        }
        if len(set(bl)) > 1:
            entry["auroc"] = float(roc_auc_score(bl, bp))
        else:
            entry["auroc"] = None
            entry["note"] = "Only one class present"
        results["per_benchmark"][bench] = entry

    # Compute VLM-only and text-only aggregate AUROC
    vlm_preds, vlm_labels = [], []
    txt_preds, txt_labels = [], []
    for bench, data in per_benchmark.items():
        if bench in VLM_BENCHMARKS:
            vlm_preds.extend(data["preds"])
            vlm_labels.extend(data["labels"])
        else:
            txt_preds.extend(data["preds"])
            txt_labels.extend(data["labels"])

    if vlm_labels and len(set(vlm_labels)) > 1:
        results["vlm_auroc"] = float(roc_auc_score(vlm_labels, vlm_preds))
        results["vlm_n"] = len(vlm_labels)
    if txt_labels and len(set(txt_labels)) > 1:
        results["text_auroc"] = float(roc_auc_score(txt_labels, txt_preds))
        results["text_n"] = len(txt_labels)

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="VLM judge cross-model evaluation")
    parser.add_argument("--target", choices=["gpt5mini", "qwen3vl", "gpt52", "qwen35"],
                        required=True, help="Target model data to evaluate on")
    parser.add_argument("--checkpoint", default=VLM_JUDGE_CHECKPOINT,
                        help="VLM judge checkpoint path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (auto-generated if not set)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run on 5 samples per benchmark")
    parser.add_argument("--max_per_benchmark", type=int, default=None)
    parser.add_argument("--no_images", action="store_true",
                        help="Use gray placeholder instead of real images (text-only mode)")
    parser.add_argument("--vlm_only", action="store_true",
                        help="Only evaluate on VLM benchmarks (skip text-only)")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_per_benchmark = 5

    use_images = not args.no_images
    mode_str = "with real images" if use_images else "text-only (gray placeholder)"

    if args.output is None:
        suffix = "_images" if use_images else "_gray"
        args.output = f"data/cross_model/vlm_judge{suffix}_on_{args.target}.json"

    TARGET_NAMES = {
        "gpt5mini": "GPT-5-mini",
        "qwen3vl": "Qwen3-VL-30B",
        "gpt52": "GPT-5.2 (high reasoning)",
        "qwen35": "Qwen3.5-397B-A17B-FP8",
    }
    target_name = TARGET_NAMES[args.target]

    print("=" * 70)
    print(f"VLM JUDGE CROSS-MODEL EVALUATION")
    print("=" * 70)
    print(f"Judge: {args.checkpoint}")
    print(f"Source (training): InternVL3-78B")
    print(f"Target (testing): {target_name}")
    print(f"Mode: {mode_str}")
    print()

    # Load data
    print(f"Loading {target_name} predictions...")
    if args.target == "gpt5mini":
        samples, data_stats = load_gpt5_mini_samples(args.max_per_benchmark)
    elif args.target == "qwen3vl":
        samples, data_stats = load_qwen3vl_samples(args.max_per_benchmark)
    elif args.target == "gpt52":
        samples, data_stats = load_prefixed_run_samples(
            "gpt52_high_", max_per_benchmark=args.max_per_benchmark)
    elif args.target == "qwen35":
        samples, data_stats = load_prefixed_run_samples(
            "qwen35_397b_", max_per_benchmark=args.max_per_benchmark)

    # Filter to VLM-only if requested
    if args.vlm_only:
        samples = [s for s in samples if s["benchmark"] in VLM_BENCHMARKS]
        print(f"VLM-only mode: filtered to {len(samples)} VLM benchmark samples")

    print(f"\nTotal: {len(samples)} samples")
    n_correct = sum(1 for s in samples if s["is_correct"])
    n_vlm = sum(1 for s in samples if s["benchmark"] in VLM_BENCHMARKS)
    n_text = len(samples) - n_vlm
    print(f"Correct: {n_correct} ({100*n_correct/len(samples):.1f}%)")
    print(f"VLM benchmarks: {n_vlm} samples, Text benchmarks: {n_text} samples")

    if len(samples) == 0:
        print("ERROR: No samples loaded!")
        sys.exit(1)

    # Load images if enabled
    image_cache = None
    if use_images:
        image_cache = load_all_images(samples)

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load VLM judge
    model, processor = load_vlm_judge(args.checkpoint)

    # Run evaluation
    print(f"\nEvaluating on {len(samples)} {target_name} responses ({mode_str})...")
    results = evaluate(model, processor, samples, image_cache=image_cache,
                       output_path=args.output)

    # Add metadata
    results["source_model"] = "InternVL3-78B"
    results["target_model"] = target_name
    results["judge"] = args.checkpoint
    results["judge_type"] = f"VLM ({mode_str})"
    results["use_images"] = use_images
    results["data_stats"] = data_stats

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    print()
    print("=" * 70)
    print(f"RESULTS ({mode_str})")
    print("=" * 70)
    print(f"Overall AUROC: {results['auroc']:.4f}")
    print(f"Overall AUPRC: {results['auprc']:.4f}")
    print(f"ECE: {results['ece']:.4f}")
    print(f"Brier: {results['brier']:.4f}")
    print(f"Base rate: {results['base_rate']:.2%}")
    if "vlm_auroc" in results:
        print(f"VLM-only AUROC: {results['vlm_auroc']:.4f} (n={results['vlm_n']})")
    if "text_auroc" in results:
        print(f"Text-only AUROC: {results['text_auroc']:.4f} (n={results['text_n']})")
    print()

    print("Per-benchmark:")
    for bench, data in sorted(results["per_benchmark"].items(),
                               key=lambda x: x[1].get("auroc") or 0,
                               reverse=True):
        auroc = data.get("auroc")
        vlm_tag = " [VLM]" if data.get("is_vlm") else ""
        if auroc is not None:
            print(f"  {bench:<20} AUROC={auroc:.3f} (n={data['n_samples']}, "
                  f"acc={data['accuracy']:.1%}){vlm_tag}")
        else:
            print(f"  {bench:<20} N/A ({data.get('note', '')}){vlm_tag}")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
