#!/usr/bin/env python3
"""
Zero-shot baseline: Run base Qwen3-VL-8B (WITHOUT LoRA fine-tuning)
on the same correctness prediction task.

This answers: "Does fine-tuning actually help, or can the base model
already distinguish correct from incorrect responses?"

Uses the same prompt template as the calibrator:
    Question: {question}
    Answer: {response}
    Is the answer correct? (i) No (ii) Yes

Extracts P(Yes) from the model's logits.

Usage:
    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/zero_shot_baseline.py \
        --scored_dir data/use_cases/scored_unified --smoke_test

    # Full run
    CUDA_VISIBLE_DEVICES=0,2 python scripts/zero_shot_baseline.py \
        --scored_dir data/use_cases/scored_unified
"""

import argparse
import json
import os
import gc
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image


def load_scored(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_scored(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def build_prompt(question_preview, response_preview):
    """Build the same prompt template used by the calibrator."""
    # Truncate very long text to fit in context
    q = question_preview[:2000] if question_preview else ""
    r = response_preview[:3000] if response_preview else ""
    return f"Question: {q}\n\nAnswer: {r}\n\nIs the answer correct? (i) No (ii) Yes"


def get_placeholder_image():
    """Gray placeholder for text benchmarks (same as training)."""
    return Image.new("RGB", (224, 224), (128, 128, 128))


def run_zero_shot(model, processor, data, batch_size=4, device="cuda"):
    """Run zero-shot inference and extract P(Yes) from logits."""
    results = []

    # Get token IDs for "Yes" and "No"
    yes_ids = processor.tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = processor.tokenizer.encode("No", add_special_tokens=False)
    # Use last token of each (handles subword tokenization)
    yes_token = yes_ids[-1]
    no_token = no_ids[-1]

    # Also check for common variants
    ii_ids = processor.tokenizer.encode("ii", add_special_tokens=False)
    i_ids = processor.tokenizer.encode("i", add_special_tokens=False)

    print(f"  Yes token ID: {yes_token} ({processor.tokenizer.decode([yes_token])})")
    print(f"  No token ID: {no_token} ({processor.tokenizer.decode([no_token])})")

    placeholder = get_placeholder_image()

    for i in tqdm(range(0, len(data), batch_size), desc="Zero-shot inference"):
        batch = data[i:i+batch_size]

        for item in batch:
            prompt_text = build_prompt(
                item.get("question_preview", ""),
                item.get("response_preview", "")
            )

            # Build messages in Qwen VL format
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": placeholder},
                    {"type": "text", "text": prompt_text},
                ]}
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=[placeholder],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                # Get logits of the last token (where the model would generate Yes/No)
                last_logits = outputs.logits[0, -1, :]
                # Softmax over all tokens
                probs = torch.softmax(last_logits, dim=0)
                p_yes = probs[yes_token].item()
                p_no = probs[no_token].item()
                # Normalize to binary choice
                p_correct_zeroshot = p_yes / (p_yes + p_no) if (p_yes + p_no) > 0 else 0.5

            item["p_zeroshot"] = float(p_correct_zeroshot)
            item["p_zeroshot_raw_yes"] = float(p_yes)
            item["p_zeroshot_raw_no"] = float(p_no)
            results.append(item)

            # Free memory
            del inputs, outputs
            if i % 100 == 0:
                torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_unified")
    parser.add_argument("--output_dir", default=None,
                        help="Output dir (default: same as scored_dir)")
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Base model (without LoRA)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run on 20 samples per target")
    parser.add_argument("--targets", default="gpt5mini,gpt52,qwen35",
                        help="Comma-separated target models to process")
    args = parser.parse_args()

    output_dir = args.output_dir or args.scored_dir
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading base model: {args.model_name}")

    # Load base model WITHOUT LoRA
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_name)
    print(f"Model loaded on {device}")

    targets = args.targets.split(",")

    for target in targets:
        path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if not path.exists():
            print(f"Skipping {target}: {path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {target}")
        print(f"{'='*60}")

        data = load_scored(path)
        if args.smoke_test:
            data = data[:20]
        print(f"  Loaded {len(data)} samples")

        data = run_zero_shot(model, processor, data,
                             batch_size=args.batch_size, device=device)

        # Report results
        valid = [d for d in data if d.get("p_zeroshot") is not None
                 and d.get("is_correct") is not None]
        if valid:
            y = np.array([d["is_correct"] for d in valid])
            p_zs = np.array([d["p_zeroshot"] for d in valid])
            p_cal = np.array([d["p_correct"] for d in valid])

            zs_auroc = roc_auc_score(y, p_zs)
            cal_auroc = roc_auc_score(y, p_cal)
            print(f"\n  Results for {target}:")
            print(f"    Zero-shot AUROC:    {zs_auroc:.4f}")
            print(f"    Calibrator AUROC:   {cal_auroc:.4f}")
            print(f"    Delta:              {cal_auroc - zs_auroc:+.4f}")

        out_path = Path(output_dir) / f"{target}_scored.jsonl"
        save_scored(data, out_path)
        print(f"  Saved to {out_path}")

    print("\nDone! Zero-shot field added: p_zeroshot")


if __name__ == "__main__":
    main()
