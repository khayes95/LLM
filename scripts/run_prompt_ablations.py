#!/usr/bin/env python3
"""Prompt/elicitation ablation experiments for the UQ model.

Tests whether changing what information the model sees or how we ask
the question affects AUROC. All use the same data and LoRA config.

Ablations:
  1. cot       - Chain-of-thought: "Analyze... then answer (i)/(ii)"
  2. longer    - Longer context: question 1500 chars, response 800 chars (vs 500/300)
  3. metadata  - Include benchmark name and source model in prompt
  4. combined  - CoT + longer context + metadata (kitchen sink)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_prompt_ablations.py \
        --prompt_variant cot \
        --output_dir data/ablations/prompt/cot

    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/run_prompt_ablations.py \
        --prompt_variant cot --smoke_test \
        --output_dir data/ablations/prompt/smoke
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

from scripts.train_best_uq import (
    MODEL_NAME, VLM_BENCHMARKS, IMAGE_CACHE_DIR,
    Sample, load_all_samples, prepare_all_images,
)


# ============================================================
# PROMPT TEMPLATES
# ============================================================

PROMPT_TEMPLATES = {
    # Baseline (same as train_best_uq.py)
    "baseline": """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes""",

    # Chain-of-thought: ask the model to reason before answering
    "cot": """Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes""",

    # Longer context (handled via truncation lengths, same template)
    "longer": """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes""",

    # Metadata-enriched: include benchmark name and source model
    "metadata": """Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes""",

    # Combined: CoT + metadata
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

# Truncation lengths per variant
TRUNCATION_LENGTHS = {
    "baseline": (500, 300),
    "cot": (500, 300),
    "longer": (1500, 800),
    "metadata": (500, 300),
    "combined": (1500, 800),
}


# ============================================================
# DATASET WITH CUSTOM PROMPTS
# ============================================================

class PromptAblationDataset(torch.utils.data.Dataset):
    """Dataset with configurable prompt templates."""

    def __init__(self, samples, processor, prompt_variant="baseline", max_length=2048):
        self.samples = samples
        self.processor = processor
        self.prompt_template = PROMPT_TEMPLATES[prompt_variant]
        self.q_len, self.r_len = TRUNCATION_LENGTHS[prompt_variant]
        self.max_length = max_length
        self.fallback_image = Image.new('RGB', (336, 336), color='gray')
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 512 * 28 * 28

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

        target = "ii" if sample.is_correct else "i"

        # Format prompt with metadata
        prompt = self.prompt_template.format(
            question=sample.question[:self.q_len],
            response=sample.response[:self.r_len],
            benchmark=sample.benchmark,
            source_model=sample.source_model,
        )

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

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        assistant_token = 77091
        assistant_positions = (input_ids == assistant_token).nonzero(as_tuple=True)[0]
        if len(assistant_positions) > 0:
            answer_pos = assistant_positions[-1].item() + 2
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
# EVALUATION WITH CUSTOM PROMPTS
# ============================================================

def evaluate_with_prompt(model, processor, test_samples, device, prompt_variant="baseline"):
    """Evaluate using a specific prompt template."""
    model.eval()
    fallback = Image.new('RGB', (224, 224), color='gray')
    prompt_template = PROMPT_TEMPLATES[prompt_variant]
    q_len, r_len = TRUNCATION_LENGTHS[prompt_variant]

    all_preds, all_labels = [], []
    per_benchmark = defaultdict(lambda: {"preds": [], "labels": []})

    for i, sample in enumerate(test_samples):
        if i % 100 == 0:
            print(f"  Eval {i}/{len(test_samples)}...")

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

        prompt = prompt_template.format(
            question=sample.question[:q_len],
            response=sample.response[:r_len],
            benchmark=sample.benchmark,
            source_model=sample.source_model,
        )
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

def get_canonical_split(all_samples, test_fraction=0.15):
    """Same split as best unified model."""
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
    parser = argparse.ArgumentParser(description="Prompt/elicitation ablation")
    parser.add_argument("--prompt_variant", type=str, required=True,
                        choices=["baseline", "cot", "longer", "metadata", "combined"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_per_bench = 5 if args.smoke_test else None

    print("=" * 70)
    print(f"PROMPT ABLATION: {args.prompt_variant}")
    print("=" * 70)
    print(f"Template:\n{PROMPT_TEMPLATES[args.prompt_variant][:200]}...")
    q_len, r_len = TRUNCATION_LENGTHS[args.prompt_variant]
    print(f"Truncation: question={q_len}, response={r_len}")

    config = vars(args)
    config["model"] = MODEL_NAME
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config["prompt_template"] = PROMPT_TEMPLATES[args.prompt_variant]
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load data
    print("\nLoading all data...")
    all_samples = load_all_samples(max_per_benchmark=max_per_bench)
    print(f"Total: {len(all_samples)} samples")

    # Canonical split
    train_idx, test_idx = get_canonical_split(all_samples)
    train_samples = [all_samples[i] for i in train_idx]
    test_samples = [all_samples[i] for i in test_idx]
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    # Prepare images
    prepare_all_images(train_samples + test_samples)

    # Load model
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

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create dataset with custom prompt
    train_dataset = PromptAblationDataset(
        train_samples, processor, prompt_variant=args.prompt_variant
    )

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

    # Train
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

    print(f"\n{'='*50}")
    print(f"TRAINING (prompt variant: {args.prompt_variant})")
    print(f"  Samples: {len(train_dataset)}, Steps/epoch: ~{steps_per_epoch}")
    print(f"  Epochs: {args.epochs}, Effective batch: {eff_batch}")
    print(f"{'='*50}")

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time/60:.1f} minutes")

    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    # Evaluate
    print(f"\n{'='*50}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*50}")

    device = next(model.parameters()).device
    results = evaluate_with_prompt(model, processor, test_samples, device,
                                   prompt_variant=args.prompt_variant)
    results["prompt_variant"] = args.prompt_variant
    results["config"] = config
    results["train_samples"] = len(train_samples)
    results["train_time_minutes"] = round(train_time / 60, 1)

    print(f"\n{'='*50}")
    print(f"RESULTS: prompt variant = {args.prompt_variant}")
    print(f"{'='*50}")
    print(f"Overall AUROC: {results['auroc']:.4f}")
    print(f"VLM AUROC:     {results.get('vlm_auroc', 'N/A')}")
    print(f"Text AUROC:    {results.get('text_auroc', 'N/A')}")
    print(f"ECE:           {results['ece']:.4f}")
    print(f"Brier:         {results['brier']:.4f}")
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

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
