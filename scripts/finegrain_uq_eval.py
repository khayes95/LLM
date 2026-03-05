#!/usr/bin/env python3
"""Score FineGRAIN T2I failure mode detection with the UQ calibrator.

This script uses our Qwen3-VL-8B UQ model to directly evaluate whether a
text-to-image model's output matches its prompt, treating it as a visual QA task.

The FineGRAIN benchmark provides:
  - T2I-generated images
  - Original prompts
  - Failure mode labels (human-annotated)
  - 27 failure mode categories

Our UQ model sees:
  - Image: the T2I generated image
  - Question: "Does this image accurately depict the prompt: '{prompt}'?
               Regarding: {failure_mode}"
  - Answer: "Yes, the image correctly depicts the prompt."
  - Model outputs P(answer is correct) -- i.e., P(image is compliant)

We evaluate:
  1. AUROC of P(compliant) vs human labels (0=compliant, 1=failure)
  2. Selective prediction: abstain on uncertain samples
  3. Per-failure-mode breakdown
  4. Per-T2I-model breakdown
  5. Comparison with FineGRAIN's VLM+LLM pipeline (67.4% accuracy)

Usage:
    # Smoke test (10 samples)
    CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_eval.py --smoke_test

    # Full evaluation (3800 samples, ~1.5 hours on 1 GPU)
    CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_eval.py

    # With v2 checkpoint
    CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_eval.py \
        --checkpoint uq_models/best_v2_r32_combined --prompt_variant combined
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
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, average_precision_score,
)
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_CHECKPOINT = "uq_models/best_v2_r32_combined"

# FineGRAIN image directory (original 5 models with human labels)
FINEGRAIN_IMAGE_DIR = Path("/scratch/khayes/diff/t2i-finegrain/images")
FINEGRAIN_METADATA = Path("/scratch/khayes/diff/t2i-finegrain/metadata.csv")

# Models with both images and human labels
LABELED_MODELS = ["flux", "sd3.5_large", "sd3.5_medium", "sd3_m", "sd3_xl"]

# Model directory name mapping (metadata model names -> image dir names)
MODEL_DIR_MAP = {
    "flux": "flux",
    "sd3.5_large": "sd3.5_large",
    "sd3.5_medium": "sd3.5_medium",
    "sd3_m": "sd3_m",
    "sd3_xl": "sd3_xl",
}

PROMPT_TEMPLATES = {
    "baseline": """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes""",

    "combined": """Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes""",

    # FineGRAIN-specific prompts
    "finegrain_direct": """Question: A text-to-image model was given this prompt: "{prompt}"

Look at the generated image. Does it accurately depict what was requested?

Specifically, check for the failure mode: "{failure_mode}" — {fm_description}

Answer: {answer}

Is the answer correct? (i) No (ii) Yes""",

    "finegrain_qa": """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes""",
}

TRUNCATION_LENGTHS = {
    "baseline": (500, 300),
    "combined": (1500, 800),
    "finegrain_direct": (1500, 800),
    "finegrain_qa": (500, 300),
}

# Failure mode descriptions from FineGRAIN
FAILURE_MODE_DESCRIPTIONS = {
    "Counts or Multiple Objects": "The model struggles with generating a precise number of distinct objects in a scene.",
    "Colour attribute binding": "The model fails to bind the correct color to the specified object.",
    "Shape attribute binding": "The model fails to bind the correct shape to the specified object.",
    "Texture attribute binding": "The model fails to bind the correct texture to the specified object.",
    "Scaling": "The model fails to correctly scale objects relative to each other.",
    "Perspective": "The model fails to render the correct perspective or viewpoint.",
    "Spatial Relation": "The model fails to position objects in the correct spatial relationship.",
    "Physics": "The model generates images that violate physical laws.",
    "Negation": "The model fails to handle absence or negation specifications.",
    "Human Action": "The model fails to depict human actions correctly.",
    "Human Anatomy Moving": "The model generates anatomically incorrect humans in motion.",
    "Anatomical limb and torso accuracy": "The model generates anatomically incorrect limbs or torsos.",
    "Emotional conveyance": "The model fails to convey the specified emotion.",
    "Text-based": "The model fails to render text correctly in the image.",
    "Short Text Specific": "The model fails to render short text strings correctly.",
    "Long text specific": "The model fails to render long text strings correctly.",
    "Blending Different Styles": "The model fails to blend different artistic styles.",
    "FG-BG relations": "The model fails to correctly render foreground-background relationships.",
    "Background and Foreground Mismatch": "The model produces mismatched backgrounds and foregrounds.",
    "Opposite of Normal Relation": "The model fails with reversed or opposite relationships.",
    "Surreal": "The model fails to generate surreal or impossible scenarios.",
    "(Visual Reasoning) Cause-and-effect Relations": "The model fails to depict cause-and-effect relationships.",
    "Tense and aspect variation": "The model fails to depict temporal aspects correctly.",
    "Tense+Text Rendering + Style": "The model fails with combined tense, text rendering, and style requirements.",
    "Action and motion representation": "The model fails to represent action and motion.",
    "Social Relations": "The model fails to depict social relationships between figures.",
    "Depicting abstract concepts": "The model fails to represent abstract concepts visually.",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_finegrain_data(metadata_path, image_dir, models=None, max_samples=None):
    """Load FineGRAIN metadata and map to image paths.

    Returns list of dicts with keys:
        prompt_id, prompt_text, failure_mode, model, image_path, human_label
    """
    df = pd.read_csv(metadata_path)

    # Filter to models with images and labels
    if models is None:
        models = LABELED_MODELS
    df = df[df["model"].isin(models)]

    # Remove NaN labels
    df = df[df["human_labels"].notna()]

    samples = []
    for _, row in df.iterrows():
        model_dir = MODEL_DIR_MAP.get(row["model"], row["model"])
        img_path = image_dir / model_dir / f"{int(row['prompt_id']):05d}.png"

        if not img_path.exists():
            continue

        samples.append({
            "prompt_id": int(row["prompt_id"]),
            "prompt_text": row["prompt_text"],
            "failure_mode": row["failure_mode"],
            "model": row["model"],
            "image_path": str(img_path),
            "human_label": int(row["human_labels"]),  # 0=compliant, 1=failure
        })

    if max_samples and len(samples) > max_samples:
        # Stratified sampling: keep class balance
        rng = np.random.RandomState(42)
        pos = [s for s in samples if s["human_label"] == 1]
        neg = [s for s in samples if s["human_label"] == 0]
        n_pos = min(len(pos), max_samples // 2)
        n_neg = min(len(neg), max_samples - n_pos)
        samples = (
            list(rng.choice(pos, n_pos, replace=False))
            + list(rng.choice(neg, n_neg, replace=False))
        )
        rng.shuffle(samples)

    return samples


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(checkpoint_path):
    """Load Qwen3-VL + LoRA model."""
    print(f"Loading base model: {BASE_MODEL}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Find best checkpoint
    ckpt = Path(checkpoint_path)
    subdirs = sorted(ckpt.glob("checkpoint-*"))
    lora_path = str(subdirs[-1]) if subdirs else str(ckpt)
    print(f"Loading LoRA adapter: {lora_path}")

    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    return model, processor


# ============================================================
# INFERENCE
# ============================================================

def get_p_correct(model, processor, question, response, image, device,
                  prompt_template, q_len=500, r_len=300,
                  benchmark="", source_model=""):
    """Get P(correct) for a QA pair using the UQ model."""
    prompt = prompt_template.format(
        question=question[:q_len],
        response=response[:r_len],
        benchmark=benchmark,
        source_model=source_model,
    )

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True,
        min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()  # P(Yes, correct)


def score_finegrain_sample(model, processor, device, sample, prompt_variant):
    """Score a single FineGRAIN sample.

    We frame the task as:
        Question: "Does this image accurately depict: {prompt}?
                   Check for failure mode: {failure_mode}"
        Answer: "Yes, the image correctly depicts the prompt."

    P(correct) = P(the image is compliant)
    High P(correct) → model thinks image matches prompt → no failure
    Low P(correct)  → model thinks image doesn't match → failure present
    """
    image = Image.open(sample["image_path"]).convert("RGB")

    fm_desc = FAILURE_MODE_DESCRIPTIONS.get(
        sample["failure_mode"],
        "A specific generation failure."
    )

    if prompt_variant == "finegrain_direct":
        # Use FineGRAIN-specific prompt
        template = PROMPT_TEMPLATES["finegrain_direct"]
        prompt = template.format(
            prompt=sample["prompt_text"][:800],
            failure_mode=sample["failure_mode"],
            fm_description=fm_desc,
            answer="Yes, the image correctly depicts the prompt without this failure mode.",
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[image], return_tensors="pt", padding=True,
            min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
        token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
        probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
        p_correct = probs[1].item()

    else:
        # Map to standard QA format
        question = (
            f"A text-to-image model was asked to generate an image with this prompt: "
            f"\"{sample['prompt_text'][:600]}\"\n\n"
            f"Check for the failure mode \"{sample['failure_mode']}\": {fm_desc}\n\n"
            f"Does the generated image accurately depict the prompt?"
        )
        response = "Yes, the image correctly depicts the prompt without exhibiting the specified failure mode."

        q_len, r_len = TRUNCATION_LENGTHS.get(prompt_variant, (500, 300))
        p_correct = get_p_correct(
            model, processor, question, response, image, device,
            prompt_template=PROMPT_TEMPLATES.get(prompt_variant, PROMPT_TEMPLATES["baseline"]),
            q_len=q_len, r_len=r_len,
            benchmark="finegrain",
            source_model=sample["model"],
        )

    return p_correct


# ============================================================
# METRICS
# ============================================================

def compute_metrics(labels, scores, prefix=""):
    """Compute comprehensive metrics for binary classification."""
    results = {}

    labels = np.array(labels)
    scores = np.array(scores)

    # Note: human_label=0 means compliant (positive for us), 1 means failure
    # Our scores: high P(correct) = compliant, low P(correct) = failure
    # For AUROC: we want to detect failures, so invert
    # AUROC of (1-scores) vs labels, OR equivalently scores vs (1-labels)
    failure_scores = 1.0 - scores  # Higher = more likely failure

    if len(set(labels)) > 1:
        results[f"{prefix}auroc"] = roc_auc_score(labels, failure_scores)
        results[f"{prefix}auprc"] = average_precision_score(labels, failure_scores)
    else:
        results[f"{prefix}auroc"] = float("nan")
        results[f"{prefix}auprc"] = float("nan")

    # Binary predictions at threshold 0.5
    preds = (failure_scores >= 0.5).astype(int)
    results[f"{prefix}accuracy"] = accuracy_score(labels, preds)
    if labels.sum() > 0 and (1 - labels).sum() > 0:
        results[f"{prefix}f1"] = f1_score(labels, preds)
        results[f"{prefix}precision"] = precision_score(labels, preds, zero_division=0)
        results[f"{prefix}recall"] = recall_score(labels, preds, zero_division=0)

    # Optimal threshold
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        t_preds = (failure_scores >= t).astype(int)
        if labels.sum() > 0:
            t_f1 = f1_score(labels, t_preds, zero_division=0)
            if t_f1 > best_f1:
                best_f1, best_thresh = t_f1, t
    results[f"{prefix}best_f1"] = best_f1
    results[f"{prefix}best_threshold"] = best_thresh

    # Selective prediction: only judge samples where model is confident
    # Sort by confidence (abs distance from 0.5)
    confidence = np.abs(failure_scores - 0.5)
    sorted_idx = np.argsort(-confidence)  # Most confident first

    coverages = [0.25, 0.50, 0.75, 1.0]
    for cov in coverages:
        n = int(len(labels) * cov)
        if n == 0:
            continue
        sel_idx = sorted_idx[:n]
        sel_preds = (failure_scores[sel_idx] >= best_thresh).astype(int)
        sel_labels = labels[sel_idx]
        if len(set(sel_labels)) > 1:
            sel_acc = accuracy_score(sel_labels, sel_preds)
            results[f"{prefix}selective_acc@{int(cov*100)}%"] = sel_acc

    # Calibration: mean predicted prob vs actual failure rate
    results[f"{prefix}mean_failure_score"] = float(failure_scores.mean())
    results[f"{prefix}actual_failure_rate"] = float(labels.mean())
    results[f"{prefix}n_samples"] = len(labels)
    results[f"{prefix}n_failures"] = int(labels.sum())

    return results


def compute_bootstrap_ci(labels, scores, n_bootstrap=1000, alpha=0.05):
    """Compute bootstrap confidence interval for AUROC."""
    labels = np.array(labels)
    scores = np.array(scores)
    failure_scores = 1.0 - scores

    if len(set(labels)) < 2:
        return {"auroc_mean": float("nan"), "auroc_ci_low": float("nan"),
                "auroc_ci_high": float("nan")}

    rng = np.random.RandomState(42)
    aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(labels), len(labels), replace=True)
        if len(set(labels[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(labels[idx], failure_scores[idx]))

    aurocs = np.array(aurocs)
    return {
        "auroc_mean": float(aurocs.mean()),
        "auroc_ci_low": float(np.percentile(aurocs, 100 * alpha / 2)),
        "auroc_ci_high": float(np.percentile(aurocs, 100 * (1 - alpha / 2))),
        "n_bootstrap": len(aurocs),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate UQ model on FineGRAIN benchmark")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to UQ model checkpoint")
    parser.add_argument("--prompt_variant", default="combined",
                        choices=list(PROMPT_TEMPLATES.keys()),
                        help="Prompt template to use")
    parser.add_argument("--output_dir", default="data/finegrain_uq",
                        help="Output directory for results")
    parser.add_argument("--image_dir", default=str(FINEGRAIN_IMAGE_DIR),
                        help="Path to FineGRAIN images")
    parser.add_argument("--metadata", default=str(FINEGRAIN_METADATA),
                        help="Path to FineGRAIN metadata CSV")
    parser.add_argument("--models", nargs="+", default=None,
                        help="T2I models to evaluate (default: all labeled)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run on 10 samples only")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=" * 70)
    print("FineGRAIN x UQ Evaluation")
    print("=" * 70)

    max_samples = 10 if args.smoke_test else args.max_samples
    samples = load_finegrain_data(
        Path(args.metadata), Path(args.image_dir),
        models=args.models, max_samples=max_samples,
    )

    n_fail = sum(s["human_label"] == 1 for s in samples)
    n_ok = sum(s["human_label"] == 0 for s in samples)
    print(f"Loaded {len(samples)} samples: {n_ok} compliant, {n_fail} failures")
    print(f"Failure modes: {len(set(s['failure_mode'] for s in samples))}")
    print(f"T2I models: {sorted(set(s['model'] for s in samples))}")

    if not samples:
        print("ERROR: No samples loaded!")
        sys.exit(1)

    # Load model
    print(f"\nLoading UQ model from {args.checkpoint}...")
    model, processor = load_model(args.checkpoint)
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    # Score all samples
    print(f"\nScoring {len(samples)} samples with prompt_variant={args.prompt_variant}...")
    scored_path = output_dir / "scored_samples.jsonl"
    start_time = time.time()

    all_scores = []
    all_labels = []

    with open(scored_path, "w") as f_out:
        for i, sample in enumerate(samples):
            try:
                p_correct = score_finegrain_sample(
                    model, processor, device, sample, args.prompt_variant
                )
            except Exception as e:
                print(f"  ERROR on sample {i}: {e}")
                p_correct = 0.5

            all_scores.append(p_correct)
            all_labels.append(sample["human_label"])

            record = {
                "idx": i,
                "prompt_id": sample["prompt_id"],
                "model": sample["model"],
                "failure_mode": sample["failure_mode"],
                "human_label": sample["human_label"],
                "p_compliant": p_correct,
                "prompt_text": sample["prompt_text"][:200],
            }
            f_out.write(json.dumps(record) + "\n")

            if (i + 1) % 50 == 0 or (i + 1) == len(samples):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(samples) - i - 1) / rate if rate > 0 else 0
                cur_auroc = ""
                if len(set(all_labels)) > 1:
                    cur_auroc = f" | AUROC={roc_auc_score(all_labels, [1-s for s in all_scores]):.4f}"
                print(f"  [{i+1}/{len(samples)}] {rate:.1f} samples/s, "
                      f"ETA {eta/60:.1f}min{cur_auroc}")

    total_time = time.time() - start_time
    print(f"\nScoring complete in {total_time:.1f}s ({len(samples)/total_time:.1f} samples/s)")

    # Compute metrics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    results = {"config": {
        "checkpoint": args.checkpoint,
        "prompt_variant": args.prompt_variant,
        "n_samples": len(samples),
        "total_time_s": total_time,
    }}

    # Overall metrics
    overall = compute_metrics(all_labels, all_scores, prefix="overall_")
    results["overall"] = overall
    print(f"\nOverall ({len(samples)} samples):")
    print(f"  AUROC:     {overall['overall_auroc']:.4f}")
    print(f"  Best F1:   {overall['overall_best_f1']:.4f} (threshold={overall['overall_best_threshold']:.2f})")
    print(f"  Accuracy:  {overall['overall_accuracy']:.4f}")
    for k, v in overall.items():
        if "selective" in k:
            print(f"  {k}: {v:.4f}")

    # Bootstrap CI
    if not args.smoke_test:
        ci = compute_bootstrap_ci(all_labels, all_scores)
        results["bootstrap_ci"] = ci
        print(f"  AUROC CI:  [{ci['auroc_ci_low']:.4f}, {ci['auroc_ci_high']:.4f}]")

    # Per-model breakdown
    print(f"\nPer T2I Model:")
    results["per_model"] = {}
    for model_name in sorted(set(s["model"] for s in samples)):
        idx = [i for i, s in enumerate(samples) if s["model"] == model_name]
        m_labels = [all_labels[i] for i in idx]
        m_scores = [all_scores[i] for i in idx]
        m_metrics = compute_metrics(m_labels, m_scores)
        results["per_model"][model_name] = m_metrics
        print(f"  {model_name:20s}: AUROC={m_metrics.get('auroc', float('nan')):.4f}, "
              f"n={len(idx)}, n_fail={sum(m_labels)}")

    # Per-failure-mode breakdown
    print(f"\nPer Failure Mode:")
    results["per_failure_mode"] = {}
    for fm in sorted(set(s["failure_mode"] for s in samples)):
        idx = [i for i, s in enumerate(samples) if s["failure_mode"] == fm]
        fm_labels = [all_labels[i] for i in idx]
        fm_scores = [all_scores[i] for i in idx]
        fm_metrics = compute_metrics(fm_labels, fm_scores)
        results["per_failure_mode"][fm] = fm_metrics
        auroc_str = f"{fm_metrics.get('auroc', float('nan')):.3f}"
        print(f"  {fm:50s}: AUROC={auroc_str}, n={len(idx)}, "
              f"n_fail={sum(fm_labels)}, fail_rate={sum(fm_labels)/len(idx):.2f}")

    # Comparison with FineGRAIN baseline (67.4% accuracy)
    print(f"\n{'='*70}")
    print("COMPARISON WITH FINEGRAIN VLM+LLM PIPELINE")
    print(f"{'='*70}")
    print(f"FineGRAIN (Molmo+Llama3) accuracy: 67.4% (reported in paper)")
    print(f"Our UQ model accuracy:             {overall['overall_accuracy']*100:.1f}%")
    print(f"Our UQ model AUROC:                {overall['overall_auroc']:.4f}")
    print(f"Our UQ model best F1:              {overall['overall_best_f1']:.4f}")

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    print(f"Scored samples saved to {scored_path}")


if __name__ == "__main__":
    main()
