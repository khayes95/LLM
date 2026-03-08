#!/usr/bin/env python3
"""Evaluate finetuned FineGRAIN models on new (unseen) T2I models.

After the 5-fold CV produces checkpoints, this script:
1. Loads each fold's checkpoint
2. Scores the 15 judge-labeled models (which vary in actual quality)
3. Measures cross-model transfer: does training on SD3 variants help detect
   failures in Flux2/GPT-Image/Gemini/etc?
4. Compares finetuned vs zero-shot (v2 baseline) on the same data

Also computes:
- Selective evaluation: at what coverage can we achieve 90%/95% accuracy?
- Model ranking correlation with judge labels
- Per-failure-mode transfer analysis

Usage:
    # Use best fold checkpoint
    python scripts/finegrain_transfer_eval.py --checkpoint data/finegrain_uq/exp1_human_cv/fold_flux/checkpoint-best

    # Compare all folds
    python scripts/finegrain_transfer_eval.py --cv_dir data/finegrain_uq/exp1_human_cv
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy.stats import spearmanr
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ============================================================
# PATHS
# ============================================================
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
ZERO_SHOT_CHECKPOINT = "uq_models/best_v2_r32_combined"
FINEGRAIN_DEV = Path("/scratch/khayes/finegrain_dev")
VLM_EVAL_DIR = FINEGRAIN_DEV / "data/results/vlm_evaluation"
IMAGE_DIR = FINEGRAIN_DEV / "data/results/images"
OUTPUT_DIR = Path("data/finegrain_uq/transfer_eval")

# Models with judge labels (excluding the 5 human-labeled models)
NEW_MODELS = [
    "flux2_dev", "flux2_pro", "gemini_image", "gemini_image_native",
    "gpt_image15", "hidream", "nano_banana2", "qwen",
    "sd1", "sd2", "seedream", "wan22",
]


def load_judge_data(models=None, max_per_model=None):
    """Load judge-labeled data for new models."""
    if models is None:
        models = NEW_MODELS

    samples = []
    for model in models:
        judge_path = VLM_EVAL_DIR / f"metadata_{model}_tailored_vllm_judged.json"
        if not judge_path.exists():
            print(f"  Skipping {model} (no judge file)")
            continue

        with open(judge_path) as f:
            data = json.load(f)

        count = 0
        for item in data:
            judge_eval = item.get("llm_evaluation_tailored", {})
            label = judge_eval.get("boolean")
            if label is None:
                continue

            # Build image path
            fm_slug = item["failure_mode"].lower().replace(" ", "_")
            img_path = IMAGE_DIR / model / fm_slug / f"{item['index']}.png"

            samples.append({
                "model": model,
                "index": item["index"],
                "prompt": item["prompt"],
                "failure_mode": item["failure_mode"],
                "judge_label": int(label),  # 1=failure, 0=compliant
                "judge_score": judge_eval.get("score"),
                "image_path": str(img_path),
            })
            count += 1
            if max_per_model and count >= max_per_model:
                break

        print(f"  {model}: {count} samples")

    print(f"Total: {len(samples)} judge-labeled samples", flush=True)
    return samples


def score_samples(model, processor, samples, device="cuda:0"):
    """Score samples with the UQ model."""
    scores = []
    for i, s in enumerate(samples):
        img_path = s["image_path"]

        # Load image (or gray placeholder if missing)
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (128, 128, 128))
        else:
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        prompt = s["prompt"]
        fm = s["failure_mode"]
        question = f"Does this image accurately depict the prompt: '{prompt}'? Regarding: {fm}"
        answer = "Yes, the image correctly depicts the prompt."

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"Question: {question}\n\nAnswer: {answer}\n\nIs the answer correct? (i) No (ii) Yes"},
            ]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[img],
            return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]
            p_yes = torch.softmax(torch.tensor([logits[0, no_id].item(), logits[0, yes_id].item()]), dim=0)[1].item()

        scores.append(p_yes)

        if (i + 1) % 10 == 0:
            print(f"    Scored {i + 1}/{len(samples)}...", flush=True)

    return np.array(scores)


def evaluate_scores(scores, labels, model_names=None):
    """Compute metrics from scores and labels."""
    labels = np.array(labels)
    failure_scores = 1 - scores  # Higher = more likely failure

    results = {}

    # Overall
    if len(np.unique(labels)) >= 2:
        results["overall_auroc"] = float(roc_auc_score(labels, failure_scores))

    # Selective prediction
    sel_results = []
    sorted_idx = np.argsort(np.abs(scores - 0.5))[::-1]  # Most confident first
    for coverage in [0.1, 0.25, 0.5, 0.75, 1.0]:
        n = max(1, int(len(labels) * coverage))
        sel = sorted_idx[:n]
        preds = (failure_scores[sel] >= 0.5).astype(int)
        acc = accuracy_score(labels[sel], preds)
        sel_results.append({"coverage": coverage, "accuracy": float(acc), "n": n})
    results["selective_prediction"] = sel_results

    # Per-model if available
    if model_names is not None:
        per_model = defaultdict(lambda: {"scores": [], "labels": []})
        for s, l, m in zip(scores, labels, model_names):
            per_model[m]["scores"].append(s)
            per_model[m]["labels"].append(l)

        model_results = {}
        for m, data in per_model.items():
            y = np.array(data["labels"])
            p = 1 - np.array(data["scores"])
            if len(np.unique(y)) >= 2:
                model_results[m] = {
                    "auroc": float(roc_auc_score(y, p)),
                    "n_samples": len(y),
                    "failure_rate": float(y.mean()),
                }
        results["per_model"] = model_results

        # Model ranking
        mean_scores = {m: np.mean(d["scores"]) for m, d in per_model.items()}
        failure_rates = {m: np.mean(d["labels"]) for m, d in per_model.items()}
        models_with_both = [m for m in mean_scores if m in failure_rates and len(per_model[m]["labels"]) > 10]
        if len(models_with_both) >= 3:
            rho, p = spearmanr(
                [mean_scores[m] for m in models_with_both],
                [1 - failure_rates[m] for m in models_with_both]  # Higher score → more compliant
            )
            results["ranking_spearman_rho"] = float(rho)
            results["ranking_spearman_p"] = float(p)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Single checkpoint to evaluate")
    parser.add_argument("--cv_dir", help="CV directory with fold_*/checkpoint-best/")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_per_model", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_per_model = 5 if args.smoke_test else args.max_per_model

    print("Loading judge-labeled data...")
    samples = load_judge_data(max_per_model=max_per_model)
    if not samples:
        print("No samples loaded!")
        return

    labels = [s["judge_label"] for s in samples]
    model_names = [s["model"] for s in samples]

    all_results = {}

    # Evaluate zero-shot baseline
    print(f"\n=== Zero-shot baseline ({ZERO_SHOT_CHECKPOINT}) ===", flush=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    print("  Processor loaded", flush=True)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16
    ).to(args.device)
    print("  Base model loaded", flush=True)
    zs_model = PeftModel.from_pretrained(base_model, ZERO_SHOT_CHECKPOINT)
    zs_model.eval()
    print("  LoRA adapter loaded", flush=True)

    zs_scores = score_samples(zs_model, processor, samples, device=args.device)
    zs_results = evaluate_scores(zs_scores, labels, model_names)
    all_results["zero_shot"] = zs_results
    print(f"  Zero-shot AUROC: {zs_results.get('overall_auroc', 'N/A')}", flush=True)

    # Save intermediate results
    with open(output_dir / "transfer_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Intermediate results saved", flush=True)

    del zs_model, base_model
    torch.cuda.empty_cache()

    # Evaluate finetuned checkpoints
    if args.checkpoint:
        checkpoints = [("finetuned", args.checkpoint)]
    elif args.cv_dir:
        cv_dir = Path(args.cv_dir)
        checkpoints = []
        for fold_dir in sorted(cv_dir.glob("fold_*")):
            ckpt = fold_dir / "checkpoint-best"
            if ckpt.exists():
                fold_name = fold_dir.name
                checkpoints.append((fold_name, str(ckpt)))
    else:
        print("Specify --checkpoint or --cv_dir")
        return

    for name, ckpt_path in checkpoints:
        print(f"\n=== {name} ({ckpt_path}) ===", flush=True)
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16
        ).to(args.device)
        ft_model = PeftModel.from_pretrained(base_model, ckpt_path)
        ft_model.eval()

        ft_scores = score_samples(ft_model, processor, samples, device=args.device)
        ft_results = evaluate_scores(ft_scores, labels, model_names)
        all_results[name] = ft_results
        print(f"  {name} AUROC: {ft_results.get('overall_auroc', 'N/A')}")

        # Improvement over zero-shot
        if "overall_auroc" in zs_results and "overall_auroc" in ft_results:
            delta = ft_results["overall_auroc"] - zs_results["overall_auroc"]
            print(f"  Delta vs zero-shot: {delta:+.4f}", flush=True)

        # Save intermediate results
        with open(output_dir / "transfer_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        del ft_model, base_model
        torch.cuda.empty_cache()

    # Save
    with open(output_dir / "transfer_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'transfer_results.json'}")

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Method':<25} {'AUROC':>8} {'Rank ρ':>8}")
    print("-" * 45)
    for name, r in all_results.items():
        auroc = r.get("overall_auroc", float("nan"))
        rho = r.get("ranking_spearman_rho", float("nan"))
        print(f"{name:<25} {auroc:>8.4f} {rho:>8.3f}")


if __name__ == "__main__":
    main()
