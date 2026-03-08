#!/usr/bin/env python3
"""Score missing heatmap gap predictions and append to scored files.

Scores arc_agi/GPT5-mini and hle_multimodal/GPT5.2 predictions with
the v2 calibrator, appends to scored_test_only_v2/ files, then
regenerates the per-benchmark breakdown.

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/score_heatmap_gaps.py
    CUDA_VISIBLE_DEVICES=4 python scripts/score_heatmap_gaps.py --smoke_test
"""
import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CHECKPOINT = "uq_models/best_v2_r32_combined"
SCORED_DIR = Path("data/use_cases/scored_test_only_v2")
IMAGE_CACHE = Path("data/training_images")

PROMPT_TEMPLATE = """Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes"""

VLM_BENCHMARKS = {
    "charxiv", "mmmu", "mmstar", "hallusionbench", "mathverse",
    "mathvision", "mathvista", "realworldqa", "vizwiz", "hle_multimodal", "mmvet",
}

Q_TRUNC = 1500
R_TRUNC = 800

NEW_PREDS = [
    {
        "pred_file": "runs/gpt5_mini_combined/arc_agi/predictions.jsonl",
        "target": "gpt5mini",
        "source_model": "GPT-5-mini",
        "benchmark": "arc_agi",
    },
    {
        "pred_file": "runs/gpt52_high_hle_multimodal/predictions.jsonl",
        "target": "gpt52",
        "source_model": "GPT-5.2",
        "benchmark": "hle_multimodal",
    },
]


def load_predictions(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_image(sample, benchmark):
    if benchmark not in VLM_BENCHMARKS:
        return None
    sid = str(sample.get("id", "unknown"))
    for ext in [".png", ".jpg", ".jpeg"]:
        img_path = IMAGE_CACHE / benchmark / f"{sid}{ext}"
        if img_path.exists():
            return Image.open(img_path).convert("RGB")
    meta = sample.get("meta", {})
    if isinstance(meta, dict):
        for key in ["image_path", "image_file"]:
            p = meta.get(key)
            if p and os.path.exists(str(p)):
                return Image.open(str(p)).convert("RGB")
    return None


def score_sample(model, processor, question, response, benchmark, source_model, image=None):
    prompt_text = PROMPT_TEMPLATE.format(
        benchmark=benchmark,
        source_model=source_model,
        question=question[:Q_TRUNC],
        response=response[:R_TRUNC],
    )

    if image is not None:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]
    else:
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt_text},
        ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process images manually
    if image is not None:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        image_inputs, video_inputs = None, None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]

    yes_ids = processor.tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = processor.tokenizer.encode("No", add_special_tokens=False)
    yes_id = yes_ids[0]
    no_id = no_ids[0]

    yes_logit = logits[yes_id].item()
    no_logit = logits[no_id].item()

    probs = F.softmax(torch.tensor([no_logit, yes_logit]), dim=0)
    p_correct = probs[1].item()

    return p_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    ckpt_path = Path(CHECKPOINT)
    subdirs = sorted(ckpt_path.glob("checkpoint-*"))
    lora_path = str(subdirs[-1]) if subdirs else str(ckpt_path)
    print(f"LoRA adapter: {lora_path}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    print("Model loaded.")

    for config in NEW_PREDS:
        pred_file = config["pred_file"]
        target = config["target"]
        source_model = config["source_model"]
        benchmark = config["benchmark"]

        print(f"\n{'='*60}")
        print(f"Scoring {benchmark} / {target} from {pred_file}")
        print(f"{'='*60}")

        preds = load_predictions(pred_file)
        print(f"  Loaded {len(preds)} predictions")

        if args.smoke_test:
            preds = preds[:3]
            print(f"  Smoke test: using {len(preds)} samples")

        # Load existing scored file to check for duplicates
        scored_file = SCORED_DIR / f"{target}_scored.jsonl"
        existing_bench_ids = set()
        if scored_file.exists():
            with open(scored_file) as f:
                for line in f:
                    d = json.loads(line)
                    if d.get("benchmark") == benchmark:
                        existing_bench_ids.add(str(d.get("id")))
        print(f"  Already scored for {benchmark}: {len(existing_bench_ids)}")

        new_scored = []
        for i, pred in enumerate(preds):
            sid = str(pred.get("id", f"unknown_{i}"))
            if sid in existing_bench_ids:
                continue

            question = pred.get("input", "")
            if not isinstance(question, str):
                question = json.dumps(question)
            response = pred.get("response_text", "")
            if not isinstance(response, str):
                response = json.dumps(response)
            score = pred.get("score", {})
            is_correct = score.get("correct", 0) if isinstance(score, dict) else 0

            if not question or not response:
                continue

            image = get_image(pred, benchmark)
            has_image = image is not None

            if benchmark in VLM_BENCHMARKS and not has_image:
                image = Image.new("RGB", (224, 224), (128, 128, 128))
                has_image = True

            try:
                p_correct = score_sample(model, processor, question, response,
                                         benchmark, source_model, image)
            except Exception as e:
                print(f"  ERROR scoring sample {sid}: {e}")
                continue

            scored_entry = {
                "id": sid,
                "benchmark": benchmark,
                "source_model": source_model,
                "question": question[:2000],
                "response": response[:2000],
                "is_correct": int(is_correct),
                "p_correct": p_correct,
                "has_image": has_image,
            }
            new_scored.append(scored_entry)

            if (i + 1) % 10 == 0:
                print(f"  Scored {i+1}/{len(preds)} ({len(new_scored)} new)")

        print(f"  Total new samples scored: {len(new_scored)}")

        if new_scored:
            with open(scored_file, "a") as f:
                for entry in new_scored:
                    f.write(json.dumps(entry) + "\n")
            print(f"  Appended {len(new_scored)} to {scored_file}")

            labels = [s["is_correct"] for s in new_scored]
            scores = [s["p_correct"] for s in new_scored]
            if len(set(labels)) > 1:
                from sklearn.metrics import roc_auc_score
                auroc = roc_auc_score(labels, scores)
                print(f"  AUROC on new samples: {auroc:.4f}")

    print("\nDone scoring gaps.")


if __name__ == "__main__":
    main()
