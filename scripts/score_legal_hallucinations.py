#!/usr/bin/env python3
"""Score Stanford RegLab legal hallucination dataset with the v2 calibrator.

Dataset: reglab/legal_hallucinations (745K examples)
- 4 LLMs: GPT-4, GPT-3.5, PaLM 2, Llama 2
- 11 legal tasks: case_existence, citation_retrieval, quotation, etc.
- Binary label: hallucination (True/False)
- This is GENUINELY out-of-distribution — calibrator never saw legal data.

We sample a balanced subset per LLM × task for tractable GPU scoring.

Usage:
    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/score_legal_hallucinations.py --smoke_test

    # Full run (~2000 samples, ~1-2h on 1 GPU)
    CUDA_VISIBLE_DEVICES=0 python scripts/score_legal_hallucinations.py
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CHECKPOINT = "uq_models/best_v2_r32_combined"

PROMPT_TEMPLATE = """Benchmark: legal_hallucinations
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes"""

Q_LEN = 1500
R_LEN = 800

DATASET_PATH = "data/legal_hallucinations/dataset.csv"

# Per LLM × task, sample this many (balanced correct/incorrect)
SAMPLES_PER_CELL = 25  # ~2000 total across 4 LLMs × ~11 tasks


# ============================================================
# DATA LOADING
# ============================================================

def load_and_sample(csv_path, samples_per_cell=SAMPLES_PER_CELL, max_total=None, seed=42):
    """Load dataset and create a stratified sample."""
    df = pd.read_csv(csv_path, low_memory=False)
    # Skip duplicate header row
    df = df[df['id'] != 'id'].reset_index(drop=True)

    # Clean columns
    df['hallucination'] = df['hallucination'].astype(str).str.strip().str.lower()
    df['is_correct'] = (df['hallucination'] == 'false').astype(int)

    # Drop rows with missing outputs
    df = df.dropna(subset=['query', 'llm_output']).reset_index(drop=True)

    print(f"Total dataset: {len(df)} rows")
    print(f"Accuracy: {df['is_correct'].mean():.3f}")
    print(f"LLMs: {df['llm'].unique()}")
    print(f"Tasks: {df['task'].unique()}")

    # Stratified sampling: per LLM × task, sample balanced correct/incorrect
    rng = np.random.RandomState(seed)
    sampled = []

    for llm in sorted(df['llm'].unique()):
        for task in sorted(df['task'].unique()):
            subset = df[(df['llm'] == llm) & (df['task'] == task)]
            if len(subset) < 4:
                continue

            correct = subset[subset['is_correct'] == 1]
            incorrect = subset[subset['is_correct'] == 0]

            n_each = samples_per_cell // 2
            # Take what's available, balanced
            n_correct = min(n_each, len(correct))
            n_incorrect = min(n_each, len(incorrect))

            if n_correct > 0:
                sampled.append(correct.sample(n=n_correct, random_state=rng))
            if n_incorrect > 0:
                sampled.append(incorrect.sample(n=n_incorrect, random_state=rng))

    result = pd.concat(sampled, ignore_index=True)
    result = result.sample(frac=1, random_state=rng).reset_index(drop=True)  # Shuffle

    if max_total and len(result) > max_total:
        result = result.head(max_total)

    print(f"\nSampled: {len(result)} rows")
    print(f"Sample accuracy: {result['is_correct'].mean():.3f}")
    for llm in sorted(result['llm'].unique()):
        sub = result[result['llm'] == llm]
        print(f"  {llm}: {len(sub)} samples (acc={sub['is_correct'].mean():.3f})")

    return result


# ============================================================
# INFERENCE
# ============================================================

def get_p_correct(model, processor, question, response, device, source_model=""):
    """Extract P(correct) from calibrator logits."""
    prompt = PROMPT_TEMPLATE.format(
        question=question[:Q_LEN],
        response=response[:R_LEN],
        source_model=source_model,
    )

    image = Image.new('RGB', (224, 224), color='gray')
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

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--output_dir", default="data/legal_hallucinations/scored")
    parser.add_argument("--samples_per_cell", type=int, default=SAMPLES_PER_CELL)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.smoke_test:
        args.samples_per_cell = 2
        args.max_examples = 10

    # Load and sample data
    df = load_and_sample(args.dataset, args.samples_per_cell, args.max_examples)

    # Load model
    print(f"\nLoading base model: {BASE_MODEL}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    print(f"Loading LoRA from: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()
    device = next(model.parameters()).device
    print(f"Model on device: {device}")

    # Score each sample
    scored = []
    output_path = os.path.join(args.output_dir, "legal_hallucinations_scored.jsonl")
    t0 = time.time()

    # Map LLM names to source_model strings
    llm_to_source = {
        "GPT 4": "gpt-4",
        "GPT 3.5": "gpt-3.5-turbo",
        "PaLM 2": "palm-2",
        "Llama 2": "llama-2-70b",
    }

    for i, (_, row) in enumerate(df.iterrows()):
        source_model = llm_to_source.get(row['llm'], row['llm'])
        p = get_p_correct(
            model, processor,
            row['query'], row['llm_output'],
            device, source_model=source_model,
        )

        record = {
            "id": str(row['id']),
            "benchmark": "legal_hallucinations",
            "task": row['task'],
            "llm": row['llm'],
            "court_level": row.get('court_level', ''),
            "is_correct": int(row['is_correct']),
            "p_correct": float(p),
            "query_preview": str(row['query'])[:200],
            "output_preview": str(row['llm_output'])[:200],
            "correct_answer": str(row.get('example_correct_answer', ''))[:200],
        }
        scored.append(record)

        # Write incrementally
        with open(output_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(df)}] {rate:.1f} samples/sec | "
                  f"elapsed={elapsed:.0f}s")

    # Analysis
    print(f"\n{'='*60}")
    print("LEGAL HALLUCINATION DETECTION RESULTS")
    print(f"{'='*60}")

    labels = np.array([s['is_correct'] for s in scored])
    scores = np.array([s['p_correct'] for s in scored])

    if len(set(labels)) >= 2:
        overall_auroc = roc_auc_score(labels, scores)
        print(f"\nOverall AUROC: {overall_auroc:.3f}")
    else:
        overall_auroc = float("nan")
        print("\nOverall AUROC: N/A (single class)")

    print(f"Total samples: {len(scored)}")
    print(f"Base accuracy: {labels.mean():.3f}")

    # Per-LLM AUROC
    print(f"\n{'LLM':>15} {'N':>6} {'Acc':>6} {'AUROC':>8}")
    per_llm = {}
    for llm in sorted(set(s['llm'] for s in scored)):
        mask = np.array([s['llm'] == llm for s in scored])
        sub_labels = labels[mask]
        sub_scores = scores[mask]
        if len(set(sub_labels)) >= 2:
            auroc = roc_auc_score(sub_labels, sub_scores)
        else:
            auroc = float("nan")
        auroc_str = f"{auroc:.3f}" if not np.isnan(auroc) else "N/A"
        print(f"{llm:>15} {mask.sum():>6} {sub_labels.mean():>6.3f} {auroc_str:>8}")
        per_llm[llm] = {"n": int(mask.sum()), "accuracy": float(sub_labels.mean()),
                        "auroc": float(auroc) if not np.isnan(auroc) else None}

    # Per-task AUROC
    print(f"\n{'Task':>25} {'N':>6} {'Acc':>6} {'AUROC':>8}")
    per_task = {}
    for task in sorted(set(s['task'] for s in scored)):
        mask = np.array([s['task'] == task for s in scored])
        sub_labels = labels[mask]
        sub_scores = scores[mask]
        if len(set(sub_labels)) >= 2:
            auroc = roc_auc_score(sub_labels, sub_scores)
        else:
            auroc = float("nan")
        auroc_str = f"{auroc:.3f}" if not np.isnan(auroc) else "N/A"
        print(f"{task:>25} {mask.sum():>6} {sub_labels.mean():>6.3f} {auroc_str:>8}")
        per_task[task] = {"n": int(mask.sum()), "accuracy": float(sub_labels.mean()),
                         "auroc": float(auroc) if not np.isnan(auroc) else None}

    # Save summary
    summary = {
        "description": "Legal hallucination detection — calibrator on Stanford RegLab dataset",
        "note": "Genuinely OOD — calibrator never saw legal data during training",
        "checkpoint": args.checkpoint,
        "dataset": "reglab/legal_hallucinations",
        "total_samples": len(scored),
        "base_accuracy": float(labels.mean()),
        "overall_auroc": float(overall_auroc) if not np.isnan(overall_auroc) else None,
        "per_llm": per_llm,
        "per_task": per_task,
    }

    summary_path = os.path.join(args.output_dir, "legal_hallucinations_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")
    print(f"Scored data: {output_path}")


if __name__ == "__main__":
    main()
