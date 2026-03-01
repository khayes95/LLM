#!/usr/bin/env python3
"""Run ablation experiments for the unified UQ model.

Supports three ablation dimensions:
  1. LoRA rank (r=4, 8, 16, 32)
  2. Source model filtering (gpt5mini-only, gpt52-only, qwen35-only)
  3. Modality filtering (text-only, vlm-only)

Each ablation trains on 1 GPU and evaluates on the held-out test set.
Uses the SAME train/test split as the best unified model for fair comparison.

Usage:
    # LoRA rank ablation
    CUDA_VISIBLE_DEVICES=0 python scripts/run_ablations.py \
        --ablation lora_rank --lora_r 4 \
        --output_dir data/ablations/lora_rank/r4

    # Source model ablation
    CUDA_VISIBLE_DEVICES=0 python scripts/run_ablations.py \
        --ablation source_model --source_models gpt5mini \
        --output_dir data/ablations/source_model/gpt5mini_only

    # Modality ablation
    CUDA_VISIBLE_DEVICES=0 python scripts/run_ablations.py \
        --ablation modality --modality text_only \
        --output_dir data/ablations/modality/text_only

    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/run_ablations.py \
        --ablation lora_rank --lora_r 4 --smoke_test \
        --output_dir data/ablations/smoke
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
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import data loading from train_best_uq
from scripts.train_best_uq import (
    MODEL_NAME, DATA_SOURCES, VLM_BENCHMARKS, EXCLUDED,
    PROMPT_TEMPLATE, IMAGE_CACHE_DIR,
    Sample, load_all_samples, prepare_all_images,
    UnifiedUQDataset, evaluate_model,
)


def filter_by_source_model(samples, source_models):
    """Keep only samples from specified source models."""
    allowed = set(source_models)
    return [s for s in samples if s.source_model in allowed]


def filter_by_modality(samples, modality):
    """Filter samples by modality: text_only, vlm_only, or all."""
    if modality == "text_only":
        return [s for s in samples if not s.has_image]
    elif modality == "vlm_only":
        return [s for s in samples if s.has_image]
    return samples


def get_canonical_split(all_samples, test_fraction=0.15):
    """Create the same train/test split used by the best unified model."""
    strat_key = [s.benchmark for s in all_samples]
    strat_counts = defaultdict(int)
    for k in strat_key:
        strat_counts[k] += 1
    strat_key_safe = [k if strat_counts[k] >= 3 else "other" for k in strat_key]

    indices = list(range(len(all_samples)))
    try:
        train_idx, test_idx = train_test_split(
            indices, test_size=test_fraction, random_state=42, stratify=strat_key_safe
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            indices, test_size=test_fraction, random_state=42
        )
    return train_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description="UQ ablation experiments")
    parser.add_argument("--ablation", type=str, required=True,
                        choices=["lora_rank", "source_model", "modality"],
                        help="Ablation dimension")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--smoke_test", action="store_true")

    # LoRA rank ablation
    parser.add_argument("--lora_r", type=int, default=16)

    # Source model ablation
    parser.add_argument("--source_models", type=str, default="gpt5mini,gpt52,qwen35",
                        help="Comma-separated list of source models to include")

    # Modality ablation
    parser.add_argument("--modality", type=str, default="all",
                        choices=["text_only", "vlm_only", "all"])

    # Training hyperparams (keep defaults matching best unified)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_per_bench = 5 if args.smoke_test else None

    print("=" * 70)
    print(f"ABLATION: {args.ablation}")
    print("=" * 70)

    # Save config
    config = vars(args)
    config["model"] = MODEL_NAME
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- Load ALL data first (for canonical split) ---
    print("\nLoading all data for canonical split...")
    all_samples = load_all_samples(max_per_benchmark=max_per_bench)
    print(f"Total samples: {len(all_samples)}")

    # --- Get canonical split (same as best unified) ---
    train_idx, test_idx = get_canonical_split(all_samples)
    all_train = [all_samples[i] for i in train_idx]
    all_test = [all_samples[i] for i in test_idx]
    print(f"Canonical split: {len(all_train)} train / {len(all_test)} test")

    # --- Apply ablation filters to TRAINING data only ---
    # Test set always uses ALL data for fair comparison
    train_samples = all_train
    test_samples = all_test

    if args.ablation == "source_model":
        source_list = args.source_models.split(",")
        train_samples = filter_by_source_model(train_samples, source_list)
        print(f"Source model filter ({args.source_models}): {len(train_samples)} train samples")

    elif args.ablation == "modality":
        train_samples = filter_by_modality(train_samples, args.modality)
        # For modality ablation, also filter test set to matching modality
        # (can't evaluate VLM if trained text-only without images, etc.)
        # But we ALSO report on full test set for cross-modality transfer measurement
        test_samples_filtered = filter_by_modality(test_samples, args.modality)
        print(f"Modality filter ({args.modality}): {len(train_samples)} train, "
              f"{len(test_samples_filtered)} matched test, {len(test_samples)} full test")

    elif args.ablation == "lora_rank":
        print(f"LoRA rank: {args.lora_r} (all data used)")

    # Check we have enough data
    if len(train_samples) < 10:
        print(f"ERROR: Only {len(train_samples)} training samples after filtering. Aborting.")
        sys.exit(1)

    # Stats
    n_vlm = sum(1 for s in train_samples if s.has_image)
    n_txt = len(train_samples) - n_vlm
    by_model = defaultdict(int)
    for s in train_samples:
        by_model[s.source_model] += 1
    print(f"\nTraining data: {len(train_samples)} samples ({n_vlm} VLM, {n_txt} text)")
    for m, c in sorted(by_model.items()):
        print(f"  {m}: {c}")
    n_correct = sum(1 for s in train_samples if s.is_correct)
    print(f"Correct: {n_correct} ({100*n_correct/len(train_samples):.1f}%)")

    # --- Download images ---
    print("\nPreparing images...")
    prepare_all_images(train_samples + test_samples)

    # --- Load model ---
    print(f"\nLoading {MODEL_NAME}...")
    from transformers import (
        Qwen3VLForConditionalGeneration, AutoProcessor,
        TrainingArguments, Trainer, TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s)")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_r = args.lora_r
    lora_alpha = lora_r * 2  # Keep alpha = 2*r as convention
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Create dataset ---
    train_dataset = UnifiedUQDataset(train_samples, processor)

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

    # --- Train ---
    # Adjust grad_accum for 1 GPU to keep effective batch size reasonable
    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(train_dataset) // eff_batch

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=max(10, steps_per_epoch // 10),
        save_strategy="epoch",
        save_total_limit=1,
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
    print(f"TRAINING ({args.ablation} ablation)")
    print(f"  Samples: {len(train_dataset)}")
    print(f"  Epochs: {args.epochs}, Effective batch: {eff_batch}")
    print(f"  Steps/epoch: ~{steps_per_epoch}")
    print(f"  LoRA r={lora_r}, alpha={lora_alpha}")
    print("=" * 50)

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time/60:.1f} minutes")

    # Save
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    # --- Evaluate on FULL test set ---
    print("\n" + "=" * 50)
    print("EVALUATING ON FULL TEST SET")
    print("=" * 50)

    device = next(model.parameters()).device
    results = evaluate_model(model, processor, test_samples, device)
    results["ablation"] = args.ablation
    results["config"] = config
    results["train_samples"] = len(train_samples)
    results["train_time_minutes"] = round(train_time / 60, 1)

    # For modality ablation, also evaluate on filtered test set
    if args.ablation == "modality":
        print(f"\nEVALUATING ON {args.modality.upper()} TEST SET")
        filtered_test = filter_by_modality(test_samples, args.modality)
        if len(filtered_test) > 0:
            filtered_results = evaluate_model(model, processor, filtered_test, device)
            results["filtered_test"] = filtered_results
            print(f"  Filtered AUROC: {filtered_results['auroc']:.4f} (n={len(filtered_test)})")

        # Also evaluate cross-modality
        cross_mod = "vlm_only" if args.modality == "text_only" else "text_only"
        cross_test = filter_by_modality(test_samples, cross_mod)
        if len(cross_test) > 0:
            cross_results = evaluate_model(model, processor, cross_test, device)
            results["cross_modality_test"] = cross_results
            print(f"  Cross-modality AUROC ({cross_mod}): {cross_results['auroc']:.4f} (n={len(cross_test)})")

    # Print results
    print(f"\n{'='*50}")
    print(f"RESULTS: {args.ablation} ablation")
    print(f"{'='*50}")
    print(f"Overall AUROC: {results['auroc']:.4f}")
    print(f"VLM AUROC:     {results.get('vlm_auroc', 'N/A')}")
    print(f"Text AUROC:    {results.get('text_auroc', 'N/A')}")
    print(f"ECE:           {results['ece']:.4f}")
    print(f"Brier:         {results['brier']:.4f}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Train time:    {train_time/60:.1f} min")

    print("\nPer-benchmark:")
    for bench, data in sorted(results["per_benchmark"].items(),
                               key=lambda x: x[1].get("auroc", 0), reverse=True):
        auroc = data.get("auroc", "N/A")
        vlm_tag = " [VLM]" if data.get("is_vlm") else ""
        if isinstance(auroc, float):
            print(f"  {bench:<20} AUROC={auroc:.3f} (n={data['n_samples']}){vlm_tag}")
        else:
            print(f"  {bench:<20} {auroc} (n={data['n_samples']}){vlm_tag}")

    # Save
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
