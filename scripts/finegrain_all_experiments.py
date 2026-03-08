#!/usr/bin/env python3
"""Comprehensive FineGRAIN x UQ experiments.

Runs multiple experiments on the 3,750 human-labeled FineGRAIN samples:

1. BASELINES: CLIPScore, ImageReward (manual), BLIP-IQA
2. PROMPT-ONLY: UQ model with gray placeholder (no image)
3. CAPTION-BASED: UQ model scoring Molmo captions as "answers"
4. ENSEMBLE: Combine UQ + CLIP + other signals

Usage:
    # Smoke test
    python scripts/finegrain_all_experiments.py --smoke_test --experiment all

    # Individual experiments
    python scripts/finegrain_all_experiments.py --experiment baselines
    python scripts/finegrain_all_experiments.py --experiment prompt_only
    python scripts/finegrain_all_experiments.py --experiment caption_based
    python scripts/finegrain_all_experiments.py --experiment ensemble
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from scipy.stats import spearmanr

# ============================================================
# PATHS
# ============================================================
FINEGRAIN_IMAGE_DIR = Path("/scratch/khayes/diff/t2i-finegrain/images")
FINEGRAIN_METADATA = Path("/scratch/khayes/diff/t2i-finegrain/metadata.csv")
VLM_EVAL_DIR = Path("/scratch/khayes/finegrain_dev/data/results/vlm_evaluation")
UQ_SCORED = Path("data/finegrain_uq/scored_samples.jsonl")
OUTPUT_DIR = Path("data/finegrain_uq/experiments")

LABELED_MODELS = ["flux", "sd3.5_large", "sd3.5_medium", "sd3_m", "sd3_xl"]

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_CHECKPOINT = "uq_models/best_v2_r32_combined"


# ============================================================
# DATA LOADING
# ============================================================

def load_metadata(max_examples=None):
    """Load FineGRAIN metadata with human labels."""
    import random
    samples = []
    with open(FINEGRAIN_METADATA) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] not in LABELED_MODELS:
                continue
            if row.get("human_labels") in (None, "", "nan"):
                continue
            sample = {
                "file_name": row["file_name"],
                "prompt_id": int(row["prompt_id"]),
                "prompt_text": row["prompt_text"],
                "failure_mode": row["failure_mode"],
                "model": row["model"],
                "human_label": int(float(row["human_labels"])),  # 1=failure, 0=compliant
            }
            # Build image path
            img_path = FINEGRAIN_IMAGE_DIR / row["model"] / f"{int(row['prompt_id']):05d}.png"
            sample["image_path"] = str(img_path)
            samples.append(sample)

    if max_examples and max_examples < len(samples):
        # Stratified sampling to ensure both classes represented
        random.seed(42)
        pos = [s for s in samples if s["human_label"] == 1]
        neg = [s for s in samples if s["human_label"] == 0]
        n_pos = max(1, max_examples // 2)
        n_neg = max_examples - n_pos
        random.shuffle(pos)
        random.shuffle(neg)
        samples = pos[:n_pos] + neg[:n_neg]
        random.shuffle(samples)

    print(f"Loaded {len(samples)} human-labeled samples "
          f"({sum(s['human_label'] == 1 for s in samples)} failures, "
          f"{sum(s['human_label'] == 0 for s in samples)} compliant)")
    return samples


def load_uq_scores():
    """Load existing UQ model scores from scored_samples.jsonl."""
    scores = {}
    with open(UQ_SCORED) as f:
        for line in f:
            s = json.loads(line)
            key = (s["model"], s["prompt_id"])
            scores[key] = s.get("p_compliant", s.get("p_correct", 0.5))
    print(f"Loaded {len(scores)} UQ scores")
    return scores


def load_molmo_captions():
    """Load Molmo VLM captions from the judged JSON files.

    Captions are keyed by (model, prompt_text_prefix) for robust matching,
    since index in caption files is per-failure-mode, not global prompt_id.
    """
    captions = {}
    # Also build a prompt-text-based index for matching
    caption_by_prompt = {}

    for model in LABELED_MODELS:
        for pattern in [
            f"metadata_{model}_tailored_vllm_judged.json",
            f"metadata_{model}_tailored_vllm.json",
        ]:
            path = VLM_EVAL_DIR / pattern
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                for item in data:
                    caption = item.get("molmo_tailored_caption", "")
                    if not caption:
                        continue
                    prompt = item.get("prompt", "")
                    failure_mode = item.get("failure_mode", "")
                    entry = {
                        "caption": caption,
                        "prompt": prompt,
                        "failure_mode": failure_mode,
                        "judge_label": item.get("llm_evaluation_tailored", {}).get("boolean"),
                        "judge_score": item.get("llm_evaluation_tailored", {}).get("score"),
                        "judge_reasoning": item.get("llm_evaluation_tailored", {}).get("reasoning", ""),
                    }
                    # Key by (model, prompt_prefix, failure_mode) for robust matching
                    key = (model, prompt[:80], failure_mode)
                    caption_by_prompt[key] = entry
                break

    print(f"Loaded {len(caption_by_prompt)} Molmo captions (keyed by prompt+failure_mode)")
    return caption_by_prompt


# ============================================================
# EXPERIMENT 1: BASELINES (CLIPScore, BLIP-2)
# ============================================================

def run_baselines(samples, device="cuda:0", batch_size=32):
    """Run CLIPScore and BLIP-2 ITC baselines."""
    from transformers import CLIPModel, CLIPProcessor

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: CLIPScore Baseline")
    print("=" * 70)

    # Load CLIP
    print("Loading CLIP ViT-L/14...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()

    clip_scores = []
    labels = []
    skipped = 0

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        images = []
        texts = []
        batch_labels = []

        for s in batch:
            img_path = s["image_path"]
            if not os.path.exists(img_path):
                skipped += 1
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                texts.append(s["prompt_text"][:77])  # CLIP max token length
                batch_labels.append(s["human_label"])
            except Exception as e:
                skipped += 1
                continue

        if not images:
            continue

        with torch.no_grad():
            inputs = clip_processor(text=texts, images=images, return_tensors="pt",
                                    padding=True, truncation=True).to(device)
            outputs = clip_model(**inputs)
            # Cosine similarity between image and text embeddings
            img_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            similarities = (img_embeds * txt_embeds).sum(dim=-1).cpu().numpy()

        clip_scores.extend(similarities.tolist())
        labels.extend(batch_labels)

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i + len(batch)}/{len(samples)} samples...")

    clip_scores = np.array(clip_scores)
    labels = np.array(labels)

    # Higher CLIP score = more aligned = more likely compliant
    # For AUROC predicting failure, we use 1 - clip_score
    failure_scores = 1 - clip_scores
    auroc = roc_auc_score(labels, failure_scores)
    # Also try raw clip_score predicting compliance (label=0)
    auroc_compliance = roc_auc_score(1 - labels, clip_scores)

    print(f"\nCLIPScore Results:")
    print(f"  Samples scored: {len(labels)} (skipped: {skipped})")
    print(f"  AUROC (predicting failure): {auroc:.4f}")
    print(f"  AUROC (predicting compliance): {auroc_compliance:.4f}")
    print(f"  Mean score (compliant): {clip_scores[labels == 0].mean():.4f}")
    print(f"  Mean score (failure): {clip_scores[labels == 1].mean():.4f}")

    del clip_model
    torch.cuda.empty_cache()

    return {
        "clip_scores": clip_scores.tolist(),
        "labels": labels.tolist(),
        "auroc_failure": float(auroc),
        "auroc_compliance": float(auroc_compliance),
        "n_samples": len(labels),
        "n_skipped": skipped,
        "mean_compliant": float(clip_scores[labels == 0].mean()),
        "mean_failure": float(clip_scores[labels == 1].mean()),
    }


def run_blip2_itc(samples, device="cuda:0", batch_size=16):
    """Run BLIP-2 image-text matching score."""
    from transformers import Blip2Processor, Blip2ForImageTextRetrieval

    print("\n" + "=" * 70)
    print("EXPERIMENT 1b: BLIP-2 ITM Baseline")
    print("=" * 70)

    print("Loading BLIP-2...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-itm-vit-g")
    model = Blip2ForImageTextRetrieval.from_pretrained(
        "Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16
    ).to(device)
    model.eval()

    itm_scores = []
    labels = []
    skipped = 0

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        images = []
        texts = []
        batch_labels = []

        for s in batch:
            if not os.path.exists(s["image_path"]):
                skipped += 1
                continue
            try:
                img = Image.open(s["image_path"]).convert("RGB")
                images.append(img)
                texts.append(s["prompt_text"][:200])
                batch_labels.append(s["human_label"])
            except Exception:
                skipped += 1
                continue

        if not images:
            continue

        with torch.no_grad():
            inputs = processor(images=images, text=texts, return_tensors="pt",
                               padding=True, truncation=True).to(device)
            outputs = model(**inputs, use_image_text_matching_head=True)
            # ITM logits: [batch, 2] (no-match, match)
            itm_logits = outputs.logits_per_image
            match_probs = torch.softmax(itm_logits, dim=-1)[:, 1].cpu().numpy()

        itm_scores.extend(match_probs.tolist())
        labels.extend(batch_labels)

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i + len(batch)}/{len(samples)} samples...")

    itm_scores = np.array(itm_scores)
    labels = np.array(labels)

    failure_scores = 1 - itm_scores
    auroc = roc_auc_score(labels, failure_scores)

    print(f"\nBLIP-2 ITM Results:")
    print(f"  Samples scored: {len(labels)} (skipped: {skipped})")
    print(f"  AUROC (predicting failure): {auroc:.4f}")
    print(f"  Mean score (compliant): {itm_scores[labels == 0].mean():.4f}")
    print(f"  Mean score (failure): {itm_scores[labels == 1].mean():.4f}")

    del model
    torch.cuda.empty_cache()

    return {
        "itm_scores": itm_scores.tolist(),
        "labels": labels.tolist(),
        "auroc_failure": float(auroc),
        "n_samples": len(labels),
        "mean_compliant": float(itm_scores[labels == 0].mean()),
        "mean_failure": float(itm_scores[labels == 1].mean()),
    }


# ============================================================
# EXPERIMENT 2: PROMPT-ONLY PREDICTION (no image)
# ============================================================

def run_prompt_only(samples, device="cuda:0", checkpoint=DEFAULT_CHECKPOINT):
    """Score with UQ model using gray placeholder instead of real image."""
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Prompt-Only Prediction (Gray Placeholder)")
    print("=" * 70)

    print(f"Loading model from {checkpoint}...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device
    )
    model = PeftModel.from_pretrained(base_model, checkpoint)
    model.eval()

    # Create gray placeholder (same as text benchmarks in training)
    gray_img = Image.new("RGB", (224, 224), (128, 128, 128))

    scores = []
    labels = []

    for i, s in enumerate(samples):
        prompt_text = s["prompt_text"]
        failure_mode = s["failure_mode"]

        question = f"Does this image accurately depict the prompt: '{prompt_text}'? Regarding: {failure_mode}"
        answer = "Yes, the image correctly depicts the prompt."

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": gray_img},
                {"type": "text", "text": f"Question: {question}\n\nAnswer: {answer}\n\nIs the answer correct? (i) No (ii) Yes"},
            ]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[gray_img],
            return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]

            # Get token IDs for Yes/No
            yes_tokens = processor.tokenizer.encode("Yes", add_special_tokens=False)
            no_tokens = processor.tokenizer.encode("No", add_special_tokens=False)
            yes_id = yes_tokens[0]
            no_id = no_tokens[0]

            yes_logit = logits[0, yes_id].item()
            no_logit = logits[0, no_id].item()

            p_yes = torch.softmax(torch.tensor([no_logit, yes_logit]), dim=0)[1].item()

        scores.append(p_yes)  # P(compliant)
        labels.append(s["human_label"])

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples...")

    scores = np.array(scores)
    labels = np.array(labels)

    # Failure score = 1 - P(compliant)
    failure_scores = 1 - scores
    auroc = roc_auc_score(labels, failure_scores)

    print(f"\nPrompt-Only Results:")
    print(f"  AUROC (predicting failure): {auroc:.4f}")
    print(f"  Mean P(compliant) for compliant: {scores[labels == 0].mean():.4f}")
    print(f"  Mean P(compliant) for failure: {scores[labels == 1].mean():.4f}")

    del model, base_model
    torch.cuda.empty_cache()

    return {
        "scores": scores.tolist(),
        "labels": labels.tolist(),
        "auroc": float(auroc),
        "n_samples": len(labels),
        "mean_compliant": float(scores[labels == 0].mean()),
        "mean_failure": float(scores[labels == 1].mean()),
    }


# ============================================================
# EXPERIMENT 3: CAPTION-BASED SCORING
# ============================================================

def run_caption_based(samples, device="cuda:0", checkpoint=DEFAULT_CHECKPOINT):
    """Score using Molmo captions as 'answers' instead of images."""
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Caption-Based Scoring")
    print("=" * 70)

    captions = load_molmo_captions()
    print(f"Matched captions for {len(captions)} samples")

    print(f"Loading model from {checkpoint}...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device
    )
    model = PeftModel.from_pretrained(base_model, checkpoint)
    model.eval()

    gray_img = Image.new("RGB", (224, 224), (128, 128, 128))

    scores = []
    labels = []
    skipped = 0

    for i, s in enumerate(samples):
        key = (s["model"], s["prompt_text"][:80], s["failure_mode"])
        if key not in captions:
            skipped += 1
            continue

        cap_data = captions[key]
        caption = cap_data["caption"]
        prompt_text = s["prompt_text"]
        failure_mode = s["failure_mode"]

        # Frame as: question is the prompt requirement, answer is the caption description
        question = f"The prompt asked for: '{prompt_text}'. Regarding the aspect: {failure_mode}."
        answer = f"The generated image shows: {caption}"

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": gray_img},
                {"type": "text", "text": f"Question: {question}\n\nAnswer: {answer}\n\nIs the answer correct? (i) No (ii) Yes"},
            ]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[gray_img],
            return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]

            yes_tokens = processor.tokenizer.encode("Yes", add_special_tokens=False)
            no_tokens = processor.tokenizer.encode("No", add_special_tokens=False)
            yes_id = yes_tokens[0]
            no_id = no_tokens[0]

            yes_logit = logits[0, yes_id].item()
            no_logit = logits[0, no_id].item()

            p_yes = torch.softmax(torch.tensor([no_logit, yes_logit]), dim=0)[1].item()

        scores.append(p_yes)
        labels.append(s["human_label"])

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(samples)} (skipped {skipped})...")

    scores = np.array(scores)
    labels = np.array(labels)

    failure_scores = 1 - scores
    auroc = roc_auc_score(labels, failure_scores)

    print(f"\nCaption-Based Results:")
    print(f"  AUROC (predicting failure): {auroc:.4f}")
    print(f"  Samples scored: {len(labels)} (skipped: {skipped})")
    print(f"  Mean P(compliant) for compliant: {scores[labels == 0].mean():.4f}")
    print(f"  Mean P(compliant) for failure: {scores[labels == 1].mean():.4f}")

    del model, base_model
    torch.cuda.empty_cache()

    return {
        "scores": scores.tolist(),
        "labels": labels.tolist(),
        "auroc": float(auroc),
        "n_samples": len(labels),
        "n_skipped": skipped,
        "mean_compliant": float(scores[labels == 0].mean()),
        "mean_failure": float(scores[labels == 1].mean()),
    }


# ============================================================
# EXPERIMENT 4: ENSEMBLE
# ============================================================

def run_ensemble(samples, baseline_results, prompt_only_results=None, caption_results=None):
    """Combine multiple signals with logistic regression."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Ensemble Combinations")
    print("=" * 70)

    # Load existing UQ scores
    uq_scores_map = load_uq_scores()

    # Build aligned arrays
    uq_scores = []
    clip_scores = []
    labels = []
    has_prompt_only = prompt_only_results is not None
    has_caption = caption_results is not None
    prompt_scores = []
    caption_scores_list = []

    n_clip = len(baseline_results.get("clip_scores", []))

    for i, s in enumerate(samples):
        key = (s["model"], s["prompt_id"])
        if key not in uq_scores_map:
            continue
        if i >= n_clip:
            continue

        uq_scores.append(uq_scores_map[key])
        clip_scores.append(baseline_results["clip_scores"][i])
        labels.append(s["human_label"])

        if has_prompt_only and i < len(prompt_only_results.get("scores", [])):
            prompt_scores.append(prompt_only_results["scores"][i])
        if has_caption and i < len(caption_results.get("scores", [])):
            caption_scores_list.append(caption_results["scores"][i])

    uq_scores = np.array(uq_scores)
    clip_scores = np.array(clip_scores)
    labels = np.array(labels)

    results = {}

    # Individual AUROCs for reference
    uq_auroc = roc_auc_score(labels, 1 - uq_scores)
    clip_auroc = roc_auc_score(labels, 1 - clip_scores)
    results["uq_only_auroc"] = float(uq_auroc)
    results["clip_only_auroc"] = float(clip_auroc)
    print(f"  UQ-only AUROC: {uq_auroc:.4f}")
    print(f"  CLIP-only AUROC: {clip_auroc:.4f}")

    # Ensemble: UQ + CLIP
    X_uq_clip = np.column_stack([uq_scores, clip_scores])
    lr = LogisticRegression(max_iter=1000)
    ensemble_probs = cross_val_predict(lr, X_uq_clip, labels, cv=5, method="predict_proba")[:, 1]
    ensemble_auroc = roc_auc_score(labels, ensemble_probs)
    results["uq_clip_ensemble_auroc"] = float(ensemble_auroc)
    print(f"  UQ + CLIP ensemble AUROC: {ensemble_auroc:.4f}")

    # Ensemble with prompt-only
    if has_prompt_only and len(prompt_scores) == len(labels):
        prompt_scores = np.array(prompt_scores)
        prompt_auroc = roc_auc_score(labels, 1 - prompt_scores)
        results["prompt_only_auroc"] = float(prompt_auroc)
        print(f"  Prompt-only AUROC: {prompt_auroc:.4f}")

        X_all3 = np.column_stack([uq_scores, clip_scores, prompt_scores])
        probs_3 = cross_val_predict(LogisticRegression(max_iter=1000), X_all3, labels, cv=5, method="predict_proba")[:, 1]
        auroc_3 = roc_auc_score(labels, probs_3)
        results["uq_clip_prompt_ensemble_auroc"] = float(auroc_3)
        print(f"  UQ + CLIP + Prompt ensemble AUROC: {auroc_3:.4f}")

    # Ensemble with caption
    if has_caption and len(caption_scores_list) == len(labels):
        caption_arr = np.array(caption_scores_list)
        caption_auroc = roc_auc_score(labels, 1 - caption_arr)
        results["caption_only_auroc"] = float(caption_auroc)
        print(f"  Caption-only AUROC: {caption_auroc:.4f}")

        X_all4 = np.column_stack([uq_scores, clip_scores, caption_arr])
        probs_4 = cross_val_predict(LogisticRegression(max_iter=1000), X_all4, labels, cv=5, method="predict_proba")[:, 1]
        auroc_4 = roc_auc_score(labels, probs_4)
        results["uq_clip_caption_ensemble_auroc"] = float(auroc_4)
        print(f"  UQ + CLIP + Caption ensemble AUROC: {auroc_4:.4f}")

    # Kitchen sink ensemble
    features = [uq_scores, clip_scores]
    feature_names = ["uq", "clip"]
    if has_prompt_only and len(prompt_scores) == len(labels):
        features.append(np.array(prompt_scores))
        feature_names.append("prompt")
    if has_caption and len(caption_scores_list) == len(labels):
        features.append(np.array(caption_scores_list))
        feature_names.append("caption")

    if len(features) > 2:
        X_all = np.column_stack(features)
        probs_all = cross_val_predict(LogisticRegression(max_iter=1000), X_all, labels, cv=5, method="predict_proba")[:, 1]
        auroc_all = roc_auc_score(labels, probs_all)
        results["kitchen_sink_auroc"] = float(auroc_all)
        results["kitchen_sink_features"] = feature_names
        print(f"  Kitchen sink ({'+'.join(feature_names)}) AUROC: {auroc_all:.4f}")

    # Per-failure-mode analysis of best ensemble
    print(f"\n  Per-failure-mode breakdown (UQ+CLIP ensemble):")
    fm_map = defaultdict(lambda: {"scores": [], "labels": []})
    for i, s in enumerate(samples[:len(labels)]):
        fm_map[s["failure_mode"]]["scores"].append(ensemble_probs[i])
        fm_map[s["failure_mode"]]["labels"].append(labels[i])

    fm_results = {}
    for fm, data in sorted(fm_map.items()):
        y = np.array(data["labels"])
        p = np.array(data["scores"])
        if len(np.unique(y)) < 2:
            continue
        fm_auroc = roc_auc_score(y, p)
        fm_results[fm] = {"auroc": float(fm_auroc), "n": len(y)}
        print(f"    {fm}: AUROC={fm_auroc:.3f} (n={len(y)})")

    results["per_failure_mode"] = fm_results
    results["n_samples"] = len(labels)

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["baselines", "prompt_only", "caption_based",
                                                   "ensemble", "all"], default="all")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_examples = 100 if args.smoke_test else None
    samples = load_metadata(max_examples=max_examples)

    all_results = {"config": {"smoke_test": args.smoke_test, "n_samples": len(samples)}}
    start_time = time.time()

    # ---- Baselines ----
    baseline_results = None
    if args.experiment in ("baselines", "all"):
        baseline_results = run_baselines(samples, device=args.device)
        all_results["clipscore"] = baseline_results

        try:
            blip_results = run_blip2_itc(samples, device=args.device)
            all_results["blip2_itm"] = blip_results
        except Exception as e:
            print(f"BLIP-2 failed: {e}")
            all_results["blip2_itm"] = {"error": str(e)}

        # Save intermediate
        with open(output_dir / "baselines.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nBaseline results saved to {output_dir / 'baselines.json'}")

    # ---- Prompt-only ----
    prompt_only_results = None
    if args.experiment in ("prompt_only", "all"):
        prompt_only_results = run_prompt_only(samples, device=args.device, checkpoint=args.checkpoint)
        all_results["prompt_only"] = prompt_only_results

        with open(output_dir / "prompt_only.json", "w") as f:
            json.dump({"prompt_only": prompt_only_results}, f, indent=2)
        print(f"\nPrompt-only results saved to {output_dir / 'prompt_only.json'}")

    # ---- Caption-based ----
    caption_results = None
    if args.experiment in ("caption_based", "all"):
        caption_results = run_caption_based(samples, device=args.device, checkpoint=args.checkpoint)
        all_results["caption_based"] = caption_results

        with open(output_dir / "caption_based.json", "w") as f:
            json.dump({"caption_based": caption_results}, f, indent=2)
        print(f"\nCaption-based results saved to {output_dir / 'caption_based.json'}")

    # ---- Ensemble ----
    if args.experiment in ("ensemble", "all"):
        # Load baselines if not already computed
        if baseline_results is None:
            baseline_path = output_dir / "baselines.json"
            if baseline_path.exists():
                with open(baseline_path) as f:
                    saved = json.load(f)
                baseline_results = saved.get("clipscore", {})
            else:
                print("No baseline results found. Run baselines first.")
                return

        if prompt_only_results is None:
            po_path = output_dir / "prompt_only.json"
            if po_path.exists():
                with open(po_path) as f:
                    prompt_only_results = json.load(f).get("prompt_only")

        if caption_results is None:
            cap_path = output_dir / "caption_based.json"
            if cap_path.exists():
                with open(cap_path) as f:
                    caption_results = json.load(f).get("caption_based")

        ensemble_results = run_ensemble(samples, baseline_results,
                                         prompt_only_results, caption_results)
        all_results["ensemble"] = ensemble_results

    # ---- Save all ----
    total_time = time.time() - start_time
    all_results["total_time_s"] = total_time

    with open(output_dir / "all_experiments.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETE in {total_time:.0f}s")
    print(f"Results saved to {output_dir / 'all_experiments.json'}")
    print(f"{'=' * 70}")

    # Summary table
    print(f"\n{'Method':<35} {'AUROC':>8}")
    print("-" * 45)
    if "clipscore" in all_results:
        print(f"{'CLIPScore':<35} {all_results['clipscore']['auroc_failure']:>8.4f}")
    if "blip2_itm" in all_results and "auroc_failure" in all_results.get("blip2_itm", {}):
        print(f"{'BLIP-2 ITM':<35} {all_results['blip2_itm']['auroc_failure']:>8.4f}")
    if "prompt_only" in all_results:
        print(f"{'UQ Prompt-Only (no image)':<35} {all_results['prompt_only']['auroc']:>8.4f}")
    if "caption_based" in all_results:
        print(f"{'UQ Caption-Based (Molmo)':<35} {all_results['caption_based']['auroc']:>8.4f}")
    print(f"{'UQ + Image (existing)':<35} {'0.7359':>8}")
    if "ensemble" in all_results:
        for k, v in all_results["ensemble"].items():
            if k.endswith("_auroc") and isinstance(v, float):
                name = k.replace("_auroc", "").replace("_", " + ").title()
                print(f"{'Ensemble: ' + name:<35} {v:>8.4f}")


if __name__ == "__main__":
    main()
