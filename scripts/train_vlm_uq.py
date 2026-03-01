#!/usr/bin/env python3
"""LoRA finetune LLaVA-1.5-7B to predict correctness of VLM responses.

Uses actual images from benchmarks (not dummy images).
Prompt format matches text UQ: "Is the answer correct? (i) No (ii) Yes"

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_vlm_uq.py
"""
import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class UQTrainingSample:
    """Training sample with image path info for lazy loading."""
    question_id: str
    benchmark: str
    prompt: str
    response: str
    is_correct: bool
    # Index in benchmark dataset for image retrieval
    dataset_index: int


def parse_question_id(question_id: str, benchmark: str) -> Optional[int]:
    """Extract dataset index from question_id.

    Examples:
        vsr_test_1730 -> 1730
        mmmu_0 -> 0
        hallusion_42 -> 42
    """
    # Try to extract trailing number
    match = re.search(r'_(\d+)$', question_id)
    if match:
        return int(match.group(1))
    return None


def load_benchmark_dataset(benchmark: str):
    """Load benchmark dataset for image retrieval."""
    if benchmark == "vsr":
        return load_dataset("cambridgeltl/vsr_random", split="test")
    elif benchmark == "mmmu":
        return load_dataset("MMMU/MMMU", "Art", split="validation")
    elif benchmark == "charxiv":
        return load_dataset("princeton-nlp/CharXiv", split="validation")
    elif benchmark == "hallusionbench":
        return load_dataset("lmms-lab/HallusionBench", split="image")
    elif benchmark == "erqa":
        return load_dataset("lmms-lab/ERQA", split="test")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def get_image_from_dataset(ds, idx: int, benchmark: str) -> Optional[Image.Image]:
    """Get image from dataset by index."""
    try:
        if idx >= len(ds):
            return None

        row = ds[idx]

        # Different benchmarks store images differently
        if benchmark == "erqa":
            # ERQA stores images as a list
            images = row.get("images", [])
            if images and len(images) > 0:
                img = images[0]
            else:
                img = row.get("image")
        else:
            img = row.get("image")

        if img is None:
            return None

        if isinstance(img, Image.Image):
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img

        return None
    except Exception as e:
        print(f"Error loading image for {benchmark}[{idx}]: {e}")
        return None


def load_training_samples(base_dir: Path) -> list[UQTrainingSample]:
    """Load all training samples from feature files."""
    samples = []

    for bench_dir in base_dir.iterdir():
        if not bench_dir.is_dir() or bench_dir.name == "smoke_test":
            continue

        benchmark = bench_dir.name
        pt_files = list(bench_dir.glob("*.pt"))
        print(f"  {benchmark}: {len(pt_files)} files")

        for pt_file in pt_files:
            try:
                data = torch.load(pt_file, weights_only=False)

                question_id = data.get("question_id", "")
                idx = parse_question_id(question_id, benchmark)

                if idx is None:
                    continue

                samples.append(UQTrainingSample(
                    question_id=question_id,
                    benchmark=benchmark,
                    prompt=data.get("prompt", ""),
                    response=data.get("response", ""),
                    is_correct=data.get("is_correct", False),
                    dataset_index=idx,
                ))
            except Exception as e:
                continue

    return samples


class VLMUQDataset(torch.utils.data.Dataset):
    """Dataset for VLM UQ training with actual images."""

    def __init__(self, samples: list[UQTrainingSample], processor, max_length=512):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length

        # Cache loaded datasets
        self._datasets = {}

        # Fallback gray image for failed loads
        self.fallback_image = Image.new('RGB', (336, 336), color='gray')

    def _get_dataset(self, benchmark: str):
        """Lazy load and cache benchmark dataset."""
        if benchmark not in self._datasets:
            try:
                self._datasets[benchmark] = load_benchmark_dataset(benchmark)
            except Exception as e:
                print(f"Failed to load {benchmark}: {e}")
                self._datasets[benchmark] = None
        return self._datasets[benchmark]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get actual image from benchmark dataset
        ds = self._get_dataset(sample.benchmark)
        if ds is not None:
            image = get_image_from_dataset(ds, sample.dataset_index, sample.benchmark)
        else:
            image = None

        if image is None:
            image = self.fallback_image

        # Format prompt like text UQ
        # Question: {question}
        # Answer: {response}
        # Is the answer correct? (i) No (ii) Yes
        prompt = f"""USER: <image>
Question: {sample.prompt[:500]}

Answer: {sample.response[:300]}

Is the answer correct? (i) No (ii) Yes
ASSISTANT: {"(ii) Yes" if sample.is_correct else "(i) No"}"""

        # Process with LLaVA processor
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # Create labels - mask everything except the answer tokens
        labels = inputs["input_ids"].clone()
        # Only train on last few tokens (the answer)
        labels[0, :-5] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels.squeeze(0),
        }


def main():
    print("=" * 60)
    print("LLaVA-1.5-7B LoRA FINETUNING FOR VLM UQ JUDGE")
    print("=" * 60)
    print("\nUsing ACTUAL IMAGES from benchmarks")
    print("Prompt format: Question/Answer/Is correct? (i) No (ii) Yes\n")

    # Config - Use HuggingFace official LLaVA
    model_name = "llava-hf/llava-1.5-7b-hf"
    output_dir = "data/llava_uq_lora"

    # Load samples
    print("Loading training data...")
    base_dir = Path("data/features")
    all_samples = load_training_samples(base_dir)
    print(f"\nTotal samples: {len(all_samples)}")

    if len(all_samples) == 0:
        print("ERROR: No samples found!")
        return

    n_correct = sum(1 for s in all_samples if s.is_correct)
    print(f"Correct: {n_correct} ({100*n_correct/len(all_samples):.1f}%)")

    # Split
    train_samples, val_samples = train_test_split(
        all_samples, test_size=0.1, random_state=42
    )
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Load model
    print(f"\nLoading {model_name}...")

    processor = AutoProcessor.from_pretrained(model_name)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # LoRA config - target attention layers
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    # Create datasets
    train_dataset = VLMUQDataset(train_samples, processor)
    val_dataset = VLMUQDataset(val_samples, processor)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues with dataset caching
    )

    # Collator
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
        }

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    # Train
    print("\nStarting LoRA training...")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

    trainer.train()

    # Save
    print(f"\nSaving LoRA adapter to {output_dir}...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    print("\n" + "=" * 60)
    print("LORA TRAINING COMPLETE")
    print("=" * 60)
    print(f"LoRA adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
