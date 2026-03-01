#!/usr/bin/env python3
"""
Train Qwen3-VL as UQ judge on GPT-5-mini evaluation results.

Attempts Qwen3-VL-30B-A3B (MoE) first, falls back to 8B if OOM.

Usage:
    CUDA_VISIBLE_DEVICES=1,2,3,4 python scripts/train_qwen3_vlm_uq.py --data_dir data/gpt5_mini_eval

Model memory estimates (bf16):
- Qwen3-VL-30B-A3B: ~60GB model + ~40GB activations = ~100GB (fits on 2x A100-80GB)
- Qwen3-VL-8B: ~16GB model + ~20GB activations = ~36GB (fits on 1x A100-80GB)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
import traceback

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODELS = [
    {
        "name": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "size": "30B-MoE",
        "min_gpus": 2,
        "max_memory_per_gpu": "78GiB",
    },
    {
        "name": "Qwen/Qwen3-VL-8B-Instruct",
        "size": "8B",
        "min_gpus": 1,
        "max_memory_per_gpu": "75GiB",
    },
]

PROMPT_TEMPLATE = """Question: {question}
Answer: {response}
Is the answer correct? (i) No (ii) Yes"""


# ============================================================
# DATA LOADING
# ============================================================

@dataclass
class UQSample:
    """Training sample for UQ classifier."""
    id: str
    benchmark: str
    input: str  # Question
    ground_truth: str
    model_response: str
    correct: int  # 0 or 1
    image_path: Optional[str] = None


def load_training_data(data_dir: Path) -> tuple[list[UQSample], list[UQSample]]:
    """Load training data from GPT-5-mini evaluation runs."""
    samples = []

    # Look for predictions.jsonl files
    for pred_file in data_dir.rglob("predictions*.jsonl"):
        print(f"Loading {pred_file}...")
        with open(pred_file) as f:
            for line in f:
                data = json.loads(line)

                # Skip ungraded samples
                if data.get("correct", -1) == -1:
                    continue

                samples.append(UQSample(
                    id=data.get("id", ""),
                    benchmark=data.get("benchmark", ""),
                    input=data.get("input", ""),
                    ground_truth=data.get("ground_truth", ""),
                    model_response=data.get("model_response", ""),
                    correct=data.get("correct", 0),
                    image_path=data.get("image_path"),
                ))

    if not samples:
        # Fall back to runs directory format
        runs_dir = Path("runs")
        for run_dir in runs_dir.glob("*gpt-5-mini*"):
            pred_file = run_dir / "predictions.jsonl"
            if pred_file.exists():
                print(f"Loading {pred_file}...")
                with open(pred_file) as f:
                    for line in f:
                        data = json.loads(line)
                        example = data.get("example", {})
                        response = data.get("response", {})
                        score = data.get("score", {})

                        if score.get("correct") in [0, 1]:
                            samples.append(UQSample(
                                id=example.get("id", ""),
                                benchmark=run_dir.name.split("_")[2] if "_" in run_dir.name else "",
                                input=example.get("input", ""),
                                ground_truth=example.get("expected_output", ""),
                                model_response=response.get("text", ""),
                                correct=score.get("correct", 0),
                            ))

    print(f"Loaded {len(samples)} graded samples")

    # Show class balance
    n_correct = sum(1 for s in samples if s.correct == 1)
    print(f"Class balance: {n_correct} correct ({100*n_correct/len(samples):.1f}%), "
          f"{len(samples)-n_correct} incorrect ({100*(len(samples)-n_correct)/len(samples):.1f}%)")

    # Split 80/20
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    split = int(0.8 * len(samples))

    train = [samples[i] for i in indices[:split]]
    test = [samples[i] for i in indices[split:]]

    return train, test


# ============================================================
# DATASET
# ============================================================

class UQDataset(torch.utils.data.Dataset):
    """Dataset for UQ classifier training."""

    def __init__(self, samples: list[UQSample], processor, max_length=2048, use_images=False):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.use_images = use_images
        self.fallback_image = Image.new('RGB', (224, 224), color=(128, 128, 128))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get image (placeholder for text-only samples)
        if self.use_images and sample.image_path:
            try:
                image = Image.open(sample.image_path).convert('RGB')
            except:
                image = self.fallback_image
        else:
            image = self.fallback_image

        # Target: "i" for incorrect, "ii" for correct
        target = "ii" if sample.correct == 1 else "i"

        prompt = PROMPT_TEMPLATE.format(
            question=sample.input[:500],
            response=sample.model_response[:300]
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": target,
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            min_pixels=256 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        # Mask everything except the answer token
        assistant_token = self.processor.tokenizer.encode("assistant", add_special_tokens=False)[-1]
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
            "labels": labels,
        }

        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            result["pixel_values"] = pv[0] if isinstance(pv, list) else pv.squeeze(0) if pv.dim() > 3 else pv

        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]

        return result


# ============================================================
# MODEL LOADING WITH FALLBACK
# ============================================================

def try_load_model(model_config: dict, num_gpus: int):
    """Try to load a model, return None if OOM."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model

    model_name = model_config["name"]
    print(f"\nTrying to load {model_name} ({model_config['size']})...")

    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Set up memory allocation
        max_memory = {}
        for i in range(num_gpus):
            max_memory[i] = model_config["max_memory_per_gpu"]

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
        )

        # Apply LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        model = get_peft_model(model, lora_config)

        print("Trainable parameters:")
        model.print_trainable_parameters()

        # Test forward pass to check memory
        print("Testing forward pass...")
        test_input = processor.tokenizer("Test", return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**test_input)
        print(f"Successfully loaded {model_name}")

        return model, processor, model_name

    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM loading {model_name}: {e}")
        torch.cuda.empty_cache()
        return None, None, None
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        traceback.print_exc()
        return None, None, None


def load_best_model(num_gpus: int):
    """Try models in order of preference, fall back on OOM."""
    for model_config in MODELS:
        if num_gpus >= model_config["min_gpus"]:
            model, processor, name = try_load_model(model_config, num_gpus)
            if model is not None:
                return model, processor, name

    raise RuntimeError("Could not load any model - check GPU availability")


# ============================================================
# TRAINING
# ============================================================

def train(args):
    from transformers import TrainingArguments, Trainer
    from sklearn.metrics import roc_auc_score

    print("=" * 70)
    print("QWEN3-VL UQ CLASSIFIER TRAINING")
    print("=" * 70)

    # Detect GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\nDetected {num_gpus} GPUs")
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")

    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    train_samples, test_samples = load_training_data(Path(args.data_dir))

    if len(train_samples) == 0:
        print("ERROR: No training samples found!")
        print("Run GPT-5-mini evaluation first:")
        print("  ./scripts/run_all_gpt5_mini.sh")
        return

    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    # Load model with fallback
    model, processor, model_name = load_best_model(num_gpus)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = UQDataset(train_samples, processor, use_images=args.use_images)
    test_dataset = UQDataset(test_samples, processor, use_images=args.use_images)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
    )

    # Collator
    def collate_fn(batch):
        max_len = max(x["input_ids"].size(0) for x in batch)
        pad_token_id = processor.tokenizer.pad_token_id or 0

        input_ids = []
        attention_mask = []
        labels = []

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
            labels.append(torch.cat([
                torch.full((pad_len,), -100, dtype=x["labels"].dtype),
                x["labels"]
            ]))

        result = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

        if "pixel_values" in batch[0]:
            result["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in batch])

        return result

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    trainer.train()

    # Save final model
    print("\nSaving model...")
    trainer.save_model()

    # Save config
    config = {
        "base_model": model_name,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "epochs": args.epochs,
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nModel saved to {output_dir}")
    print("\nDone!")


def check_gpus_available(min_gpus: int = 4, min_free_memory_gb: float = 70.0) -> bool:
    """Check if enough GPUs are available with sufficient free memory."""
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )

        available_gpus = 0
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                free_mb, total_mb = map(float, line.split(","))
                free_gb = free_mb / 1024
                if free_gb >= min_free_memory_gb:
                    available_gpus += 1

        return available_gpus >= min_gpus
    except Exception as e:
        print(f"Error checking GPU availability: {e}")
        return False


def wait_for_gpus(min_gpus: int = 4, poll_interval_hours: float = 1.0):
    """Poll for GPU availability, waiting until enough are free."""
    import time
    from datetime import datetime

    poll_seconds = poll_interval_hours * 3600

    print(f"Waiting for {min_gpus} GPUs with >=70GB free memory...")
    print(f"Polling every {poll_interval_hours} hour(s)")

    while True:
        if check_gpus_available(min_gpus):
            print(f"\n[{datetime.now()}] GPUs available! Starting training...")
            return True

        print(f"[{datetime.now()}] GPUs busy. Checking again in {poll_interval_hours}h...")
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL UQ classifier")
    parser.add_argument("--data_dir", default="runs", help="Directory with GPT-5-mini predictions")
    parser.add_argument("--output_dir", default="data/qwen3_uq_judge", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--use_images", action="store_true", help="Load actual images (slower)")
    parser.add_argument("--wait_for_gpus", action="store_true", help="Poll hourly until GPUs are available")
    parser.add_argument("--min_gpus", type=int, default=4, help="Minimum GPUs required")
    parser.add_argument("--poll_interval", type=float, default=1.0, help="Hours between GPU checks")

    args = parser.parse_args()

    if args.wait_for_gpus:
        wait_for_gpus(min_gpus=args.min_gpus, poll_interval_hours=args.poll_interval)

    train(args)


if __name__ == "__main__":
    main()
