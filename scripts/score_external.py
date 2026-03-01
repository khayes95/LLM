#!/usr/bin/env python3
"""Score external question-answer pairs with the unified UQ model.

Takes a JSONL file with {"question": "...", "response": "..."} pairs and
outputs P(correct) scores using our Qwen3-VL-8B + LoRA calibrator.

Designed for cross-project use: other projects prepare the JSONL, we score it.

Usage:
    # Smoke test (first 5 rows)
    CUDA_VISIBLE_DEVICES=0 python scripts/score_external.py \
        --input data/external/energetics_qa.jsonl \
        --output data/external/energetics_scored.jsonl \
        --smoke_test

    # Full run
    CUDA_VISIBLE_DEVICES=0 python scripts/score_external.py \
        --input data/external/energetics_qa.jsonl \
        --output data/external/energetics_scored.jsonl

    # With custom checkpoint
    CUDA_VISIBLE_DEVICES=0 python scripts/score_external.py \
        --input data/external/energetics_qa.jsonl \
        --output data/external/energetics_scored.jsonl \
        --checkpoint uq_models/best_unified

Input JSONL format (required fields marked with *):
    {
        "question": "...",          * required
        "response": "...",          * required
        "id": "unique_id",           optional (auto-generated if missing)
        "model_name": "gpt-5",       optional (passed through)
        "correct_answer": "...",     optional (passed through)
        "is_correct": 0 or 1,        optional (enables AUROC computation)
        ...                           any extra fields are passed through
    }

Output JSONL format:
    {
        ...all input fields...,
        "uq_score": 0.732,          P(correct) from UQ model [0, 1]
    }
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_CHECKPOINT = "uq_models/best_unified"

PROMPT_TEMPLATE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""

# Text-only: use gray placeholder image (model was trained this way)
GRAY_IMAGE = Image.new('RGB', (224, 224), color='gray')


# ============================================================
# MODEL
# ============================================================

def load_model(checkpoint_path: str):
    """Load Qwen3-VL-8B + LoRA checkpoint."""
    print(f"Loading base model: {BASE_MODEL}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Find the actual checkpoint (use latest if subdirectories exist)
    lora_path = checkpoint_path
    subdirs = [
        d for d in sorted(os.listdir(checkpoint_path))
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_path, d))
    ]
    if subdirs and not os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        lora_path = os.path.join(checkpoint_path, subdirs[-1])
        print(f"Using latest checkpoint: {lora_path}")

    print(f"Loading LoRA weights: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    return model, processor, device


def get_p_correct(model, processor, question: str, response: str,
                  image: Image.Image, device) -> float:
    """Get P(correct) for a question-response pair."""
    prompt = PROMPT_TEMPLATE.format(
        question=question[:500],
        response=response[:300],
    )

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
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
    parser = argparse.ArgumentParser(description="Score external QA pairs with UQ model")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file with uq_score")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help=f"UQ model checkpoint (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Only score first 5 samples")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to score")
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load input data
    print(f"Reading input: {args.input}")
    samples = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            # Validate required fields
            if "question" not in row or "response" not in row:
                print(f"WARNING: Line {i+1} missing 'question' or 'response', skipping")
                continue

            # Normalize ID field (accept "id" or "question_id")
            if "id" not in row:
                row["id"] = row.get("question_id", f"sample_{i}")

            # Normalize model field (accept "model" or "model_name")
            if "model" in row and "model_name" not in row:
                row["model_name"] = row["model"]

            samples.append(row)

    if not samples:
        print("ERROR: No valid samples found in input file")
        sys.exit(1)

    # Apply limits
    if args.smoke_test:
        samples = samples[:5]
        print(f"SMOKE TEST: scoring first 5 samples only")
    elif args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Loaded {len(samples)} samples")

    # Check if we can compute AUROC
    has_labels = all("is_correct" in s for s in samples)
    if has_labels:
        n_correct = sum(s["is_correct"] for s in samples)
        print(f"Labels available: {n_correct}/{len(samples)} correct "
              f"({100*n_correct/len(samples):.1f}%)")
    else:
        print("No 'is_correct' field — will skip AUROC computation")

    # Load model
    model, processor, device = load_model(args.checkpoint)

    # Score samples
    print(f"\nScoring {len(samples)} samples...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    start_time = time.time()
    all_scores = []
    all_labels = []

    with open(args.output, "w") as f_out:
        for i, sample in enumerate(samples):
            try:
                p_correct = get_p_correct(
                    model, processor,
                    sample["question"], sample["response"],
                    GRAY_IMAGE, device
                )
            except Exception as e:
                if i < 5:
                    print(f"  Error on sample {i} ({sample['id']}): {e}")
                p_correct = 0.5

            all_scores.append(p_correct)
            if has_labels:
                all_labels.append(sample["is_correct"])

            # Write output: all original fields + uq_score
            out_row = dict(sample)
            out_row["uq_score"] = round(p_correct, 6)
            f_out.write(json.dumps(out_row) + "\n")

            if (i + 1) % 50 == 0 or (i + 1) == len(samples):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(samples) - i - 1) / rate if rate > 0 else 0

                status = f"[{i+1}/{len(samples)}] {rate:.1f} samples/s, ETA {eta/60:.1f}m"
                if has_labels and len(set(all_labels)) > 1:
                    from sklearn.metrics import roc_auc_score
                    auroc = roc_auc_score(all_labels, all_scores)
                    status += f", interim AUROC={auroc:.4f}"
                print(f"  {status}")
                f_out.flush()

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print(f"SCORING COMPLETE")
    print(f"{'='*60}")
    print(f"Samples scored: {len(samples)}")
    print(f"Time: {elapsed:.0f}s ({len(samples)/elapsed:.1f} samples/s)")
    print(f"Output: {args.output}")
    print(f"Mean P(correct): {np.mean(all_scores):.4f}")
    print(f"Median P(correct): {np.median(all_scores):.4f}")
    print(f"P(correct) range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")

    if has_labels and len(set(all_labels)) > 1:
        from sklearn.metrics import roc_auc_score, brier_score_loss
        auroc = roc_auc_score(all_labels, all_scores)
        brier = brier_score_loss(all_labels, all_scores)
        print(f"AUROC: {auroc:.4f}")
        print(f"Brier score: {brier:.4f}")

    # Per-model breakdown if model_name field exists
    models = set(s.get("model_name") for s in samples if s.get("model_name"))
    if models:
        print(f"\nPer-model breakdown:")
        for model_name in sorted(models):
            idx = [i for i, s in enumerate(samples) if s.get("model_name") == model_name]
            m_scores = [all_scores[i] for i in idx]
            print(f"  {model_name}: n={len(idx)}, mean P(correct)={np.mean(m_scores):.4f}")
            if has_labels:
                m_labels = [all_labels[i] for i in idx]
                if len(set(m_labels)) > 1:
                    from sklearn.metrics import roc_auc_score
                    print(f"    AUROC={roc_auc_score(m_labels, m_scores):.4f}")

    print(f"\nDone. Scored file: {args.output}")


if __name__ == "__main__":
    main()
