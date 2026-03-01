#!/usr/bin/env python3
"""LoRA finetune Qwen3-VL-30B as VLM UQ judge.

Uses ACTUAL images from benchmarks and same train/test split as probe.
Prompt format: "Question: ... Answer: ... Is the answer correct? (i) No (ii) Yes"

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_vlm_judge.py
"""
import sys
import json
import re
import io
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
import tensorflow as tf
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# DATA LOADING
# ============================================================

@dataclass
class UQTrainingSample:
    """Training sample for VLM judge."""
    question_id: str
    benchmark: str
    prompt: str
    response: str
    is_correct: bool
    dataset_index: int


def parse_question_id(question_id: str) -> Optional[int]:
    """Extract dataset index from question_id."""
    match = re.search(r'_(\d+)$', question_id)
    if match:
        return int(match.group(1))
    return None


def load_erqa_from_tfrecord(tfrecord_path: str) -> list:
    """Load ERQA dataset from TFRecord file."""
    feature_description = {
        'answer': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.VarLenFeature(tf.string),
        'question_type': tf.io.VarLenFeature(tf.string),
        'visual_indices': tf.io.VarLenFeature(tf.int64),
        'question': tf.io.FixedLenFeature([], tf.string)
    }

    samples = []
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    for example_proto in dataset:
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        images_encoded = tf.sparse.to_dense(parsed['image/encoded']).numpy()

        # Get first image
        if len(images_encoded) > 0:
            img_bytes = images_encoded[0]
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                img = None
        else:
            img = None

        samples.append({
            "image": img,
            "question": parsed['question'].numpy().decode('utf-8'),
            "answer": parsed['answer'].numpy().decode('utf-8'),
        })

    return samples


class ERQADataset:
    """Wrapper for ERQA TFRecord data to match HuggingFace dataset interface."""
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


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
        # Load from local TFRecord
        tfrecord_path = Path("data/erqa_repo/data/erqa.tfrecord")
        if tfrecord_path.exists():
            samples = load_erqa_from_tfrecord(str(tfrecord_path))
            return ERQADataset(samples)
        return None
    else:
        return None


def get_image_from_dataset(ds, idx: int, benchmark: str) -> Optional[Image.Image]:
    """Get image from dataset by index."""
    try:
        if idx >= len(ds):
            return None
        row = ds[idx]

        if benchmark == "erqa":
            # ERQADataset stores image directly
            img = row.get("image")
        else:
            img = row.get("image")

        if img is None:
            return None
        if isinstance(img, Image.Image):
            return img.convert("RGB") if img.mode != "RGB" else img
        return None
    except Exception:
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
                idx = parse_question_id(question_id)

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
            except Exception:
                continue

    return samples


def load_test_ids(path: Path) -> set:
    """Load test IDs from probe evaluation."""
    with open(path) as f:
        test_ids = json.load(f)
    return {item["question_id"] for item in test_ids}


# ============================================================
# PROMPT FORMAT
# ============================================================

PROMPT_TEMPLATE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""


# ============================================================
# DATASET
# ============================================================

class VLMJudgeDataset(torch.utils.data.Dataset):
    """Dataset for VLM judge training with actual images."""

    def __init__(self, samples: list[UQTrainingSample], processor, max_length=2048):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self._datasets = {}
        self.fallback_image = Image.new('RGB', (336, 336), color='gray')

        # Control image resolution to limit tokens
        self.min_pixels = 256 * 28 * 28  # Qwen3-VL uses 28x28 patches
        self.max_pixels = 512 * 28 * 28

        # Preload datasets
        benchmarks = set(s.benchmark for s in samples)
        for bench in benchmarks:
            try:
                self._datasets[bench] = load_benchmark_dataset(bench)
                print(f"    Loaded {bench} dataset")
            except Exception as e:
                print(f"    Failed to load {bench}: {e}")
                self._datasets[bench] = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get actual image
        ds = self._datasets.get(sample.benchmark)
        image = None
        if ds is not None:
            image = get_image_from_dataset(ds, sample.dataset_index, sample.benchmark)
        if image is None:
            image = self.fallback_image

        # Target: "i" for incorrect, "ii" for correct
        target = "ii" if sample.is_correct else "i"

        prompt = PROMPT_TEMPLATE.format(
            question=sample.prompt[:500],
            response=sample.response[:300]
        )

        # Build conversation format for Qwen2-VL
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

        # Apply chat template
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # Process with Qwen3-VL processor
        # Use padding=True, no truncation to avoid image token mismatch
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        # Labels - mask everything except the assistant's answer token
        # Find where assistant response starts (after "assistant\n")
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        # Find the assistant token and train on the token after it
        # The pattern is: <|im_start|> assistant \n <answer> <|im_end|>
        assistant_token = 77091  # 'assistant' token
        im_end_token = 151645  # '<|im_end|>' token

        # Find last occurrence of assistant token
        assistant_positions = (input_ids == assistant_token).nonzero(as_tuple=True)[0]
        if len(assistant_positions) > 0:
            # The answer is 2 tokens after "assistant" (skip the newline)
            answer_pos = assistant_positions[-1].item() + 2
            # Mask everything except the answer token
            labels[:] = -100
            if answer_pos < len(labels):
                labels[answer_pos] = input_ids[answer_pos]
        else:
            # Fallback: mask all but keep some signal
            labels[:-3] = -100

        result = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

        # Handle pixel_values - might be nested
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            if isinstance(pv, list):
                result["pixel_values"] = pv[0] if len(pv) > 0 else pv
            else:
                result["pixel_values"] = pv.squeeze(0) if pv.dim() > 3 else pv

        # Handle image_grid_thw if present
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]

        return result


# ============================================================
# EVALUATION
# ============================================================

def get_p_correct(model, processor, image, question, response, device):
    """Extract P(correct) from model logits."""
    prompt = PROMPT_TEMPLATE.format(question=question, response=response)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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

    # Get token IDs for "i" and "ii"
    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]

    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()  # P(correct) = P("ii")


def evaluate_vlm_judge(
    model,
    processor,
    test_samples: list[UQTrainingSample],
    datasets_cache: dict,
    device: torch.device,
    fallback_image: Image.Image,
) -> dict:
    """Evaluate VLM judge and compute metrics."""
    model.eval()

    all_preds = []
    all_labels = []
    per_benchmark = defaultdict(lambda: {"preds": [], "labels": []})

    # Get token IDs for "i" and "ii"
    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
    print(f"Token IDs: i={token_i}, ii={token_ii}")

    for i, sample in enumerate(test_samples):
        if i % 50 == 0:
            print(f"  Evaluating {i}/{len(test_samples)}...")

        # Get image
        ds = datasets_cache.get(sample.benchmark)
        image = None
        if ds is not None:
            image = get_image_from_dataset(ds, sample.dataset_index, sample.benchmark)
        if image is None:
            image = fallback_image

        try:
            p_correct = get_p_correct(
                model, processor, image,
                sample.prompt[:500], sample.response[:300], device
            )
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            p_correct = 0.5  # Default

        all_preds.append(p_correct)
        all_labels.append(float(sample.is_correct))
        per_benchmark[sample.benchmark]["preds"].append(p_correct)
        per_benchmark[sample.benchmark]["labels"].append(float(sample.is_correct))

    # Compute metrics
    results = {
        "auroc": roc_auc_score(all_labels, all_preds),
        "auprc": average_precision_score(all_labels, all_preds),
        "n_samples": len(all_labels),
        "n_correct": sum(all_labels),
        "base_rate": sum(all_labels) / len(all_labels),
    }

    # Per-benchmark
    results["per_benchmark"] = {}
    for bench, data in per_benchmark.items():
        if len(set(data["labels"])) < 2:
            continue
        results["per_benchmark"][bench] = {
            "auroc": roc_auc_score(data["labels"], data["preds"]),
            "n_samples": len(data["labels"]),
        }

    # ECE and Brier
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    results["brier"] = float(np.mean((preds - labels) ** 2))

    # ECE
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for j in range(n_bins):
        in_bin = (preds >= bin_boundaries[j]) & (preds < bin_boundaries[j + 1])
        if in_bin.sum() == 0:
            continue
        bin_conf = preds[in_bin].mean()
        bin_acc = labels[in_bin].mean()
        ece += (in_bin.sum() / len(preds)) * abs(bin_acc - bin_conf)
    results["ece"] = ece

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("VLM JUDGE TRAINING (Qwen3-VL-8B LoRA)")
    print("=" * 70)
    print("\nGoal: Beat probe baseline (0.704 AUROC)")
    print("Using ACTUAL images from benchmarks")
    print("Same train/test split as probe evaluation\n")

    # Config
    # Using Qwen3-VL-8B as smaller alternative - the 30B MoE model has OOM issues with current GPU allocation
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    output_dir = Path("data/vlm_judge_lora")
    output_dir.mkdir(exist_ok=True)

    # Load all samples
    print("Loading training data...")
    base_dir = Path("data/features")
    all_samples = load_training_samples(base_dir)
    print(f"\nTotal samples: {len(all_samples)}")

    # Load test IDs from probe evaluation
    test_ids_path = Path("data/probe_results/test_ids.json")
    if test_ids_path.exists():
        test_ids = load_test_ids(test_ids_path)
        print(f"Loaded {len(test_ids)} test IDs from probe evaluation")
    else:
        print("WARNING: No test_ids.json found, using random 20% split")
        test_ids = set()

    # Split using same IDs as probe
    if test_ids:
        train_samples = [s for s in all_samples if s.question_id not in test_ids]
        test_samples = [s for s in all_samples if s.question_id in test_ids]
    else:
        # Fallback to random split
        np.random.seed(42)
        indices = np.random.permutation(len(all_samples))
        split = int(0.8 * len(all_samples))
        train_samples = [all_samples[i] for i in indices[:split]]
        test_samples = [all_samples[i] for i in indices[split:]]

    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    n_correct_train = sum(1 for s in train_samples if s.is_correct)
    n_correct_test = sum(1 for s in test_samples if s.is_correct)
    print(f"Train correct: {n_correct_train} ({100*n_correct_train/len(train_samples):.1f}%)")
    print(f"Test correct: {n_correct_test} ({100*n_correct_test/len(test_samples):.1f}%)")

    # Load model
    print(f"\nLoading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Use all 4 GPUs to spread the model and leave headroom for fp32 conversions
    # The 8B model fits easily on 4 A100s with room for activation memory
    max_memory = {
        0: "75GiB",  # Leave headroom on each GPU for fp32 conversions during forward pass
        1: "75GiB",
        2: "75GiB",
        3: "78GiB",
    }

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)

    # Note: Gradient checkpointing causes OOM during backward pass recomputation
    # with this MoE model. Disabled for now - the original run worked without it.

    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = VLMJudgeDataset(train_samples, processor)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,  # Increased to reduce memory overhead
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch_fused",  # More memory efficient optimizer
        max_grad_norm=1.0,
    )

    # Collator - handle variable-length sequences and pixel_values
    def collate_fn(batch):
        # Pad input_ids, attention_mask, labels to max length in batch
        max_len = max(x["input_ids"].size(0) for x in batch)
        pad_token_id = processor.tokenizer.pad_token_id or 0

        input_ids = []
        attention_mask = []
        labels = []

        for x in batch:
            seq_len = x["input_ids"].size(0)
            pad_len = max_len - seq_len

            # Left-pad for causal LM
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

        # Handle pixel_values - cat along batch dim
        if "pixel_values" in batch[0]:
            result["pixel_values"] = torch.cat([x["pixel_values"] for x in batch], dim=0)

        # Handle image_grid_thw if present
        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in batch])

        return result

    # Memory cleanup callback
    class MemoryCleanupCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 100 == 0:
                torch.cuda.empty_cache()
            return control

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[MemoryCleanupCallback()],
    )

    # Train
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

    trainer.train()

    # Save
    print(f"\nSaving LoRA adapter to {output_dir}...")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    # Evaluate
    print("\n" + "=" * 50)
    print("EVALUATING ON TEST SET")
    print("=" * 50)

    # Get cached datasets from training
    datasets_cache = train_dataset._datasets
    fallback_image = train_dataset.fallback_image
    device = next(model.parameters()).device

    results = evaluate_vlm_judge(
        model, processor, test_samples, datasets_cache, device, fallback_image
    )

    # Print results
    print("\n" + "=" * 70)
    print("VLM JUDGE RESULTS")
    print("=" * 70)
    print(f"\nOverall AUROC: {results['auroc']:.4f}")
    print(f"Overall AUPRC: {results['auprc']:.4f}")
    print(f"ECE: {results['ece']:.4f}")
    print(f"Brier: {results['brier']:.4f}")

    print("\nPer-benchmark AUROC:")
    for bench, metrics in sorted(results["per_benchmark"].items()):
        print(f"  {bench}: {metrics['auroc']:.4f} (n={metrics['n_samples']})")

    # Compare to probe
    print("\n" + "-" * 50)
    print("COMPARISON TO PROBE BASELINE")
    print("-" * 50)
    probe_auroc = 0.704
    vlm_auroc = results["auroc"]
    diff = vlm_auroc - probe_auroc

    print(f"Probe baseline: {probe_auroc:.4f}")
    print(f"VLM Judge:      {vlm_auroc:.4f}")
    print(f"Difference:     {diff:+.4f}")

    if vlm_auroc > probe_auroc:
        print("\n✓ VLM Judge beats probe baseline!")
    else:
        print("\n✗ VLM Judge does not beat probe baseline")

    if vlm_auroc > 0.75:
        print("✓ VLM Judge achieves target AUROC > 0.75!")

    # Save results
    results_path = output_dir / "vlm_judge_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
