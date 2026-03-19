#!/usr/bin/env python3
"""Fine-tune the UQ model on FineGRAIN diffusion T2I compliance data.

Continues from the v2 checkpoint (uq_models/best_v2_r32_combined/) and trains
on FineGRAIN human-labeled and/or judge-labeled data for diffusion image
quality assessment.

Three experiment modes:
  human_cv   (Exp 1): 5-fold leave-one-model-out CV on human data
  judge_only (Exp 2): Train on judge data, test on all human data
  combined   (Exp 3): Train on human + judge + QA replay, test on held-out model

Usage:
    # Smoke test (verify pipeline)
    CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_finetune.py \
        --experiment human_cv --output_dir data/finegrain_uq/smoke --smoke_test

    # Experiment 1: 5-fold CV on human data
    CUDA_VISIBLE_DEVICES=0,1 python scripts/finegrain_finetune.py \
        --experiment human_cv --output_dir data/finegrain_uq/exp1_human_cv

    # Experiment 1: single fold
    CUDA_VISIBLE_DEVICES=0,1 python scripts/finegrain_finetune.py \
        --experiment human_cv --output_dir data/finegrain_uq/exp1_fold_flux \
        --held_out_model flux

    # Experiment 2: judge-only
    CUDA_VISIBLE_DEVICES=0,1 python scripts/finegrain_finetune.py \
        --experiment judge_only --output_dir data/finegrain_uq/exp2_judge

    # Experiment 2: exclude flux from training
    CUDA_VISIBLE_DEVICES=0,1 python scripts/finegrain_finetune.py \
        --experiment judge_only --output_dir data/finegrain_uq/exp2_judge_noflux \
        --exclude_flux

    # Experiment 3: combined (human + judge + QA replay)
    CUDA_VISIBLE_DEVICES=0,1 python scripts/finegrain_finetune.py \
        --experiment combined --output_dir data/finegrain_uq/exp3_combined \
        --held_out_model sd3_xl
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_CHECKPOINT = "uq_models/best_v2_r32_combined"

# Matches v2 checkpoint prompt template
PROMPT_TEMPLATE_COMBINED = """Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes"""

# Truncation lengths matching v2 config
Q_TRUNCATION = 1500
R_TRUNCATION = 800

# VLM benchmarks from the original training script (for QA replay image loading)
VLM_BENCHMARKS = {
    "charxiv", "mmmu", "mmstar", "hallusionbench", "mathverse",
    "mathvision", "mathvista", "realworldqa", "vizwiz", "hle_multimodal", "mmvet",
}

EXCLUDED_BENCHMARKS = {
    "triviaqa", "babilong", "vsr", "aokvqa", "erqa", "tutorbench",
    "healthbench", "arc", "oolong",
}

IMAGE_CACHE_DIR = Path("data/training_images")

# Human-labeled models for 5-fold CV
HUMAN_MODELS = ["flux", "sd3.5_large", "sd3.5_medium", "sd3_m", "sd3_xl"]

# QA data sources (same as train_best_uq.py)
QA_DATA_SOURCES = {
    "gpt5mini": {"dir": "runs/gpt5_mini_combined", "type": "combined"},
    "gpt52": {"dir": "runs", "prefix": "gpt52_high_", "type": "prefixed"},
    "qwen35": {"dir": "runs", "prefix": "qwen35_397b_", "type": "prefixed"},
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Sample:
    id: str
    benchmark: str
    source_model: str
    question: str
    response: str
    is_correct: bool
    has_image: bool
    image_path: Optional[str] = None  # Absolute path for FineGRAIN images


# ============================================================
# FINEGRAIN DATA LOADING
# ============================================================

def load_finegrain_jsonl(path: str) -> list[Sample]:
    """Load FineGRAIN samples from a JSONL file."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(Sample(
                id=d["id"],
                benchmark=d["benchmark"],
                source_model=d["source_model"],
                question=d["question"],
                response=d["response"],
                is_correct=bool(d["is_correct"]),
                has_image=bool(d["has_image"]),
                image_path=d.get("image_path"),
            ))
    return samples


def load_human_data(data_dir: str) -> dict[str, list[Sample]]:
    """Load human-labeled data, organized by model.

    Returns dict: model_name -> list[Sample]
    """
    per_model = {}
    for model in HUMAN_MODELS:
        path = Path(data_dir) / f"human_{model}.jsonl"
        if path.exists():
            per_model[model] = load_finegrain_jsonl(str(path))
            print(f"  human_{model}: {len(per_model[model])} samples")
        else:
            print(f"  WARNING: {path} not found, skipping model {model}")
    return per_model


def load_judge_data(data_dir: str, exclude_flux: bool = False) -> list[Sample]:
    """Load judge-labeled data.

    Args:
        data_dir: Directory containing judge_all.jsonl
        exclude_flux: If True, remove flux samples from judge data
    """
    path = Path(data_dir) / "judge_all.jsonl"
    samples = load_finegrain_jsonl(str(path))
    if exclude_flux:
        n_before = len(samples)
        samples = [s for s in samples if s.source_model != "flux"]
        print(f"  judge_all: {n_before} total, excluded {n_before - len(samples)} flux, "
              f"kept {len(samples)}")
    else:
        print(f"  judge_all: {len(samples)} samples")
    return samples


# ============================================================
# QA REPLAY DATA LOADING (from original training pipeline)
# ============================================================

def extract_question_text(input_data) -> str:
    """Extract question text from input field (copied from train_best_uq.py)."""
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, dict):
        for key in ["question", "query", "query_cot", "prompt", "text"]:
            if key in input_data and input_data[key]:
                val = input_data[key]
                if isinstance(val, str):
                    return val
        if "messages" in input_data:
            for msg in input_data["messages"]:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        texts = [p.get("text", "") for p in content
                                 if isinstance(p, dict) and "text" in p]
                        return " ".join(texts)
        clean = {k: v for k, v in input_data.items() if k != "images"}
        return json.dumps(clean)[:2000]
    return str(input_data)[:2000]


def load_qa_replay_samples(n_samples: int, seed: int = 42) -> list[Sample]:
    """Load a random subset of the original QA training data for replay.

    Loads from the same runs/ directories used by train_best_uq.py, then
    subsamples to n_samples. Only loads samples whose IDs are in the v2
    training split to avoid leaking test data.
    """
    # Load the v2 split info to get training IDs
    split_info_path = Path(DEFAULT_CHECKPOINT) / "split_info.json"
    train_ids = None
    if split_info_path.exists():
        with open(split_info_path) as f:
            split_info = json.load(f)
        train_ids = set(split_info.get("train_ids", []))
        print(f"  Loaded {len(train_ids)} training IDs from v2 split")

    all_qa = []

    for model_name, config in QA_DATA_SOURCES.items():
        if config["type"] == "combined":
            combined_path = Path(config["dir"])
            if not combined_path.exists():
                continue
            for bench_dir in sorted(combined_path.iterdir()):
                if not bench_dir.is_dir():
                    continue
                benchmark = bench_dir.name
                if benchmark in EXCLUDED_BENCHMARKS:
                    continue
                pred_file = bench_dir / "predictions.jsonl"
                if not pred_file.exists():
                    continue
                with open(pred_file) as f:
                    for line in f:
                        try:
                            pred = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        score = pred.get("score", -1)
                        if isinstance(score, dict):
                            correct = score.get("correct", -1)
                        else:
                            correct = score
                        if correct not in (0, 1):
                            continue
                        sid = str(pred.get("id", ""))
                        prefixed_id = f"{benchmark}_{sid}"
                        if train_ids and prefixed_id not in train_ids and sid not in train_ids:
                            continue
                        question = extract_question_text(pred.get("input", {}))
                        response = pred.get("response_text", "") or str(pred.get("prediction", ""))
                        if not question or not response:
                            continue
                        all_qa.append(Sample(
                            id=sid,
                            benchmark=benchmark,
                            source_model=model_name,
                            question=question[:2000],
                            response=response[:1000],
                            is_correct=bool(correct == 1),
                            has_image=benchmark in VLM_BENCHMARKS,
                            image_path=None,  # Uses IMAGE_CACHE_DIR
                        ))
        else:
            runs_path = Path(config["dir"])
            prefix = config["prefix"]
            if not runs_path.exists():
                continue
            for run_dir in sorted(runs_path.iterdir()):
                if not run_dir.name.startswith(prefix):
                    continue
                benchmark = run_dir.name[len(prefix):]
                if benchmark in EXCLUDED_BENCHMARKS:
                    continue
                pred_file = run_dir / "predictions.jsonl"
                if not pred_file.exists():
                    continue
                with open(pred_file) as f:
                    for line in f:
                        try:
                            pred = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        score = pred.get("score", {})
                        if isinstance(score, dict):
                            correct = score.get("correct", -1)
                        else:
                            correct = score
                        if correct not in (0, 1):
                            continue
                        sid = str(pred.get("id", ""))
                        prefixed_id = f"{benchmark}_{sid}"
                        if train_ids and prefixed_id not in train_ids and sid not in train_ids:
                            continue
                        question = extract_question_text(pred.get("input", {}))
                        response = pred.get("response_text", "")
                        if not response:
                            response = str(pred.get("prediction", {}).get("answer", ""))
                        if not question or not response:
                            continue
                        all_qa.append(Sample(
                            id=sid,
                            benchmark=benchmark,
                            source_model=model_name,
                            question=question[:2000],
                            response=response[:1000],
                            is_correct=bool(correct == 1),
                            has_image=benchmark in VLM_BENCHMARKS,
                            image_path=None,
                        ))

    print(f"  Loaded {len(all_qa)} QA replay candidates")

    if len(all_qa) == 0:
        print("  WARNING: No QA replay samples found!")
        return []

    # Subsample
    rng = np.random.RandomState(seed)
    n = min(n_samples, len(all_qa))
    idx = rng.choice(len(all_qa), n, replace=False)
    selected = [all_qa[i] for i in sorted(idx)]
    print(f"  Subsampled {n} QA replay samples (seed={seed})")

    return selected


# ============================================================
# DATASET
# ============================================================

class FineGRAINDataset(torch.utils.data.Dataset):
    """Dataset for FineGRAIN + QA replay training.

    Adapts the UnifiedUQDataset pattern from train_best_uq.py but with:
    - Direct image_path loading for FineGRAIN samples
    - IMAGE_CACHE_DIR fallback for QA replay samples
    - Combined prompt template (matching v2 checkpoint)
    """

    def __init__(self, samples: list[Sample], processor, max_length=2048):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.fallback_image = Image.new('RGB', (336, 336), color='gray')
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 512 * 28 * 28

        # Dynamic assistant token lookup
        assistant_ids = self.processor.tokenizer.encode("assistant", add_special_tokens=False)
        self.assistant_token = assistant_ids[-1] if assistant_ids else 77091

    def __len__(self):
        return len(self.samples)

    def _load_image(self, sample: Sample) -> tuple[Image.Image, int, int]:
        """Load image for a sample. Returns (image, min_pixels, max_pixels)."""
        if not sample.has_image:
            # Text-only: gray placeholder with minimal pixel budget
            return self.fallback_image, 256 * 28 * 28, 256 * 28 * 28

        image = None

        # FineGRAIN samples have image_path set to absolute paths
        if sample.image_path:
            try:
                image = Image.open(sample.image_path).convert("RGB")
            except Exception as e:
                pass

        # QA replay samples use IMAGE_CACHE_DIR
        if image is None and sample.benchmark != "finegrain":
            cache_path = IMAGE_CACHE_DIR / sample.benchmark / f"{sample.id}.jpg"
            if cache_path.exists():
                try:
                    image = Image.open(cache_path).convert("RGB")
                except Exception:
                    pass

        if image is None:
            image = self.fallback_image

        return image, self.min_pixels, self.max_pixels

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image, min_px, max_px = self._load_image(sample)

        # Target token
        target = "ii" if sample.is_correct else "i"

        # Format prompt using combined template
        prompt = PROMPT_TEMPLATE_COMBINED.format(
            benchmark=sample.benchmark,
            source_model=sample.source_model,
            question=sample.question[:Q_TRUNCATION],
            response=sample.response[:R_TRUNCATION],
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
            {"role": "assistant", "content": target},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            min_pixels=min_px,
            max_pixels=max_px,
        )

        # Labels: mask everything except the answer token
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        target_token_id = self.processor.tokenizer.encode(target, add_special_tokens=False)[-1]
        assistant_positions = (input_ids == self.assistant_token).nonzero(as_tuple=True)[0]

        if len(assistant_positions) > 0:
            search_start = assistant_positions[-1].item()
            answer_pos = None
            for pos in range(search_start, len(input_ids)):
                if input_ids[pos].item() == target_token_id:
                    answer_pos = pos
                    break

            if answer_pos is not None:
                labels[:] = -100
                labels[answer_pos] = input_ids[answer_pos]
            else:
                # Fallback: use offset
                answer_pos = search_start + 2
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
# COLLATION
# ============================================================

def make_collate_fn(processor):
    """Create a collation function with left-padding."""
    pad_token_id = processor.tokenizer.pad_token_id or 0

    def collate_fn(batch):
        max_len = max(x["input_ids"].size(0) for x in batch)

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
            result["pixel_values"] = torch.cat(
                [x["pixel_values"] for x in batch], dim=0
            )
        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.cat(
                [x["image_grid_thw"] for x in batch]
            )

        return result

    return collate_fn


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, processor, test_samples: list[Sample], device) -> dict:
    """Evaluate model on test samples. Returns metrics dict."""
    model.eval()
    fallback = Image.new('RGB', (224, 224), color='gray')

    all_preds, all_labels = [], []
    per_model_data = defaultdict(lambda: {"preds": [], "labels": []})

    for i, sample in enumerate(test_samples):
        if i % 100 == 0:
            print(f"  Eval {i}/{len(test_samples)}...")

        # Load image
        image = None
        if sample.has_image and sample.image_path:
            try:
                image = Image.open(sample.image_path).convert("RGB")
            except Exception:
                pass
        if image is None and sample.has_image and sample.benchmark != "finegrain":
            cache_path = IMAGE_CACHE_DIR / sample.benchmark / f"{sample.id}.jpg"
            if cache_path.exists():
                try:
                    image = Image.open(cache_path).convert("RGB")
                except Exception:
                    pass
        if image is None:
            image = fallback

        prompt = PROMPT_TEMPLATE_COMBINED.format(
            benchmark=sample.benchmark,
            source_model=sample.source_model,
            question=sample.question[:Q_TRUNCATION],
            response=sample.response[:R_TRUNCATION],
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
                print(f"  Error at sample {i}: {e}")
            p_correct = 0.5

        all_preds.append(p_correct)
        all_labels.append(float(sample.is_correct))
        per_model_data[sample.source_model]["preds"].append(p_correct)
        per_model_data[sample.source_model]["labels"].append(float(sample.is_correct))

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    results = {
        "auroc": float(roc_auc_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
        "auprc": float(average_precision_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
        "brier": float(brier_score_loss(labels, preds)),
        "n_samples": len(labels),
        "n_correct": int(sum(labels)),
        "accuracy": float(labels.mean()),
    }

    # ECE
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, 11)
    for j in range(10):
        in_bin = (preds >= bin_boundaries[j]) & (preds < bin_boundaries[j + 1])
        if in_bin.sum() == 0:
            continue
        ece += float(
            (in_bin.sum() / len(preds))
            * abs(labels[in_bin].mean() - preds[in_bin].mean())
        )
    results["ece"] = ece

    # Per-model breakdown
    results["per_model"] = {}
    for model_name, data in sorted(per_model_data.items()):
        bp = np.array(data["preds"])
        bl = np.array(data["labels"])
        entry = {
            "n_samples": len(bl),
            "accuracy": float(bl.mean()),
        }
        if len(set(bl)) > 1:
            entry["auroc"] = float(roc_auc_score(bl, bp))
            entry["auprc"] = float(average_precision_score(bl, bp))
        else:
            entry["auroc"] = 0.5
            entry["auprc"] = 0.5
        results["per_model"][model_name] = entry

    return results


# ============================================================
# TRAINING LOOP (single fold / single run)
# ============================================================

def train_single_run(
    train_samples: list[Sample],
    test_samples: list[Sample],
    checkpoint_path: str,
    output_dir: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    warmup_ratio: float,
    smoke_test: bool = False,
) -> dict:
    """Run a single training + evaluation cycle.

    Loads the v2 checkpoint (base model + LoRA adapter), continues training
    on the provided samples, then evaluates on test_samples.

    Returns results dict.
    """
    from transformers import (
        AutoModelForImageTextToText, AutoProcessor,
        TrainingArguments, Trainer, TrainerCallback,
    )
    from peft import PeftModel

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- Load model + existing adapter ----
    print(f"\n  Loading base model: {BASE_MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

    num_gpus = torch.cuda.device_count()
    max_memory = {i: "78GiB" for i in range(num_gpus)}
    print(f"  Using {num_gpus} GPUs")

    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )

    print(f"  Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        is_trainable=True,
    )

    # Verify all adapter parameters are trainable
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.2f}%)")

    # ---- Create dataset ----
    print(f"  Creating dataset: {len(train_samples)} train, {len(test_samples)} test")
    train_dataset = FineGRAINDataset(train_samples, processor)
    collate_fn = make_collate_fn(processor)

    # ---- Training ----
    eff_batch = batch_size * grad_accum * max(num_gpus, 1)
    steps_per_epoch = max(1, len(train_dataset) // eff_batch)
    logging_steps = max(1, steps_per_epoch // 10)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
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
            if state.global_step % 50 == 0:
                torch.cuda.empty_cache()
            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[MemoryCleanupCallback()],
    )

    print(f"\n  TRAINING: epochs={epochs}, eff_batch={eff_batch}, "
          f"steps/epoch~{steps_per_epoch}, lr={learning_rate}")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"  Training completed in {train_time / 60:.1f} minutes")

    # Save final adapter
    final_dir = output_path / "checkpoint-best"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"  Saved adapter to {final_dir}")

    # ---- Evaluate ----
    print(f"\n  EVALUATING on {len(test_samples)} test samples...")
    device = next(model.parameters()).device
    results = evaluate_model(model, processor, test_samples, device)
    results["train_time_min"] = round(train_time / 60, 1)
    results["n_train"] = len(train_samples)
    results["n_test"] = len(test_samples)

    # Save results
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n  Overall AUROC: {results['auroc']:.4f}")
    print(f"  AUPRC:         {results['auprc']:.4f}")
    print(f"  Brier:         {results['brier']:.4f}")
    print(f"  ECE:           {results['ece']:.4f}")
    print(f"  Per-model breakdown:")
    for m, d in sorted(results["per_model"].items()):
        print(f"    {m:<20} AUROC={d['auroc']:.3f} (n={d['n_samples']}, "
              f"acc={d['accuracy']:.2f})")

    # Cleanup to free GPU memory before next fold
    del model, base_model, trainer
    torch.cuda.empty_cache()

    return results


# ============================================================
# EXPERIMENT MODES
# ============================================================

def run_human_cv(args):
    """Experiment 1: 5-fold leave-one-model-out CV on human data."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Human-Only Leave-One-Model-Out CV")
    print("=" * 70)

    human_data = load_human_data(args.data_dir)

    if args.held_out_model and args.held_out_model != "all":
        # Single fold mode
        fold_models = [args.held_out_model]
    else:
        fold_models = list(human_data.keys())

    all_fold_results = {}

    for fold_idx, held_out in enumerate(fold_models):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{len(fold_models)}: Held-out model = {held_out}")
        print(f"{'='*60}")

        if held_out not in human_data:
            print(f"  WARNING: Model {held_out} not found in data, skipping")
            continue

        # Split: train on all other models, test on held-out
        train_samples = []
        for model_name, samples in human_data.items():
            if model_name != held_out:
                train_samples.extend(samples)

        test_samples = human_data[held_out]

        if args.smoke_test:
            train_samples = train_samples[:10]
            test_samples = test_samples[:10]

        n_correct_train = sum(1 for s in train_samples if s.is_correct)
        n_correct_test = sum(1 for s in test_samples if s.is_correct)
        print(f"  Train: {len(train_samples)} samples "
              f"({n_correct_train} correct, {100*n_correct_train/max(1,len(train_samples)):.1f}%)")
        print(f"  Test:  {len(test_samples)} samples "
              f"({n_correct_test} correct, {100*n_correct_test/max(1,len(test_samples)):.1f}%)")

        fold_dir = str(Path(args.output_dir) / f"fold_{held_out}")

        # Save split info
        Path(fold_dir).mkdir(parents=True, exist_ok=True)
        split_info = {
            "held_out_model": held_out,
            "train_models": [m for m in human_data.keys() if m != held_out],
            "n_train": len(train_samples),
            "n_test": len(test_samples),
            "train_ids": [f"{s.benchmark}_{s.id}" for s in train_samples],
            "test_ids": [f"{s.benchmark}_{s.id}" for s in test_samples],
        }
        with open(Path(fold_dir) / "split_info.json", "w") as f:
            json.dump(split_info, f, indent=2)

        results = train_single_run(
            train_samples=train_samples,
            test_samples=test_samples,
            checkpoint_path=args.checkpoint,
            output_dir=fold_dir,
            learning_rate=args.learning_rate,
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            warmup_ratio=args.warmup_ratio,
            smoke_test=args.smoke_test,
        )

        all_fold_results[held_out] = results

    # Aggregate across folds
    if len(all_fold_results) > 1:
        aurocs = [r["auroc"] for r in all_fold_results.values()]
        summary = {
            "experiment": "human_cv",
            "n_folds": len(all_fold_results),
            "per_fold_auroc": {k: r["auroc"] for k, r in all_fold_results.items()},
            "mean_auroc": float(np.mean(aurocs)),
            "std_auroc": float(np.std(aurocs)),
            "min_auroc": float(np.min(aurocs)),
            "max_auroc": float(np.max(aurocs)),
            "per_fold_results": all_fold_results,
        }

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "cv_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print("CV SUMMARY")
        print(f"{'='*60}")
        for model, auroc in sorted(summary["per_fold_auroc"].items()):
            print(f"  {model:<20} AUROC = {auroc:.4f}")
        print(f"\n  Mean AUROC: {summary['mean_auroc']:.4f} +/- {summary['std_auroc']:.4f}")
        print(f"  Range: [{summary['min_auroc']:.4f}, {summary['max_auroc']:.4f}]")
        print(f"\n  Saved to {output_path / 'cv_summary.json'}")
    else:
        print(f"\n  Single fold completed. Results in {args.output_dir}/fold_{fold_models[0]}/")


def run_judge_only(args):
    """Experiment 2: Train on judge data, test on all human data."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Judge-Only Training")
    print("=" * 70)

    # Load data
    print("\nLoading judge data...")
    judge_samples = load_judge_data(args.data_dir, exclude_flux=args.exclude_flux)

    print("\nLoading human data (test set)...")
    human_data = load_human_data(args.data_dir)
    test_samples = []
    for samples in human_data.values():
        test_samples.extend(samples)

    train_samples = judge_samples

    if args.smoke_test:
        train_samples = train_samples[:10]
        test_samples = test_samples[:10]

    n_correct_train = sum(1 for s in train_samples if s.is_correct)
    n_correct_test = sum(1 for s in test_samples if s.is_correct)
    print(f"\nTrain: {len(train_samples)} samples "
          f"({n_correct_train} correct, {100*n_correct_train/max(1,len(train_samples)):.1f}%)")
    print(f"Test:  {len(test_samples)} samples "
          f"({n_correct_test} correct, {100*n_correct_test/max(1,len(test_samples)):.1f}%)")

    # Save split info
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_models = sorted(set(s.source_model for s in train_samples))
    test_models = sorted(set(s.source_model for s in test_samples))
    split_info = {
        "experiment": "judge_only",
        "exclude_flux": args.exclude_flux,
        "train_models": train_models,
        "test_models": test_models,
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "train_ids": [f"{s.benchmark}_{s.id}" for s in train_samples],
        "test_ids": [f"{s.benchmark}_{s.id}" for s in test_samples],
    }
    with open(output_path / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    results = train_single_run(
        train_samples=train_samples,
        test_samples=test_samples,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        smoke_test=args.smoke_test,
    )

    results["experiment"] = "judge_only"
    results["exclude_flux"] = args.exclude_flux

    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path / 'results.json'}")


def run_combined(args):
    """Experiment 3: Train on human + judge + QA replay, test on held-out model."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Combined (Human + Judge + QA Replay)")
    print("=" * 70)

    held_out = args.held_out_model
    print(f"Held-out model for test: {held_out}")

    # Load human data
    print("\nLoading human data...")
    human_data = load_human_data(args.data_dir)

    if held_out not in human_data:
        print(f"ERROR: Held-out model '{held_out}' not in human data. "
              f"Available: {list(human_data.keys())}")
        sys.exit(1)

    # Human train: all models except held-out
    human_train = []
    for model_name, samples in human_data.items():
        if model_name != held_out:
            human_train.extend(samples)
    test_samples = human_data[held_out]

    print(f"  Human train: {len(human_train)} (from {len(human_data) - 1} models)")
    print(f"  Human test:  {len(test_samples)} (model: {held_out})")

    # Load judge data
    print("\nLoading judge data...")
    judge_samples = load_judge_data(args.data_dir, exclude_flux=args.exclude_flux)

    # Load QA replay data
    print("\nLoading QA replay data...")
    qa_replay = load_qa_replay_samples(args.qa_replay_n, seed=42)

    # Combine training data
    train_samples = human_train + judge_samples + qa_replay

    if args.smoke_test:
        # Keep a small mix from each source
        train_samples = human_train[:4] + judge_samples[:4] + qa_replay[:2]
        test_samples = test_samples[:10]

    # Report composition
    n_human = len(human_train) if not args.smoke_test else 4
    n_judge = len(judge_samples) if not args.smoke_test else 4
    n_qa = len(qa_replay) if not args.smoke_test else 2
    print(f"\nTraining composition:")
    print(f"  Human:     {n_human}")
    print(f"  Judge:     {n_judge}")
    print(f"  QA replay: {n_qa}")
    print(f"  Total:     {len(train_samples)}")
    print(f"  Test:      {len(test_samples)}")

    # Save split info
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    split_info = {
        "experiment": "combined",
        "held_out_model": held_out,
        "exclude_flux": args.exclude_flux,
        "n_human_train": n_human,
        "n_judge_train": n_judge,
        "n_qa_replay": n_qa,
        "n_train_total": len(train_samples),
        "n_test": len(test_samples),
        "train_ids": [f"{s.benchmark}_{s.id}" for s in train_samples],
        "test_ids": [f"{s.benchmark}_{s.id}" for s in test_samples],
    }
    with open(output_path / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    results = train_single_run(
        train_samples=train_samples,
        test_samples=test_samples,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        smoke_test=args.smoke_test,
    )

    results["experiment"] = "combined"
    results["held_out_model"] = held_out
    results["train_composition"] = {
        "human": n_human,
        "judge": n_judge,
        "qa_replay": n_qa,
    }

    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path / 'results.json'}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune UQ model on FineGRAIN diffusion T2I compliance data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test
  python scripts/finegrain_finetune.py --experiment human_cv --output_dir /tmp/fg_smoke --smoke_test

  # Experiment 1: full 5-fold CV
  python scripts/finegrain_finetune.py --experiment human_cv --output_dir data/finegrain_uq/exp1

  # Experiment 1: single fold (faster)
  python scripts/finegrain_finetune.py --experiment human_cv --output_dir data/finegrain_uq/exp1_flux --held_out_model flux

  # Experiment 2: judge-only, exclude flux
  python scripts/finegrain_finetune.py --experiment judge_only --output_dir data/finegrain_uq/exp2 --exclude_flux

  # Experiment 3: combined with QA replay
  python scripts/finegrain_finetune.py --experiment combined --output_dir data/finegrain_uq/exp3 --held_out_model sd3_xl
        """,
    )

    # Required
    parser.add_argument("--experiment", required=True,
                        choices=["human_cv", "judge_only", "combined"],
                        help="Experiment mode")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for checkpoints and results")

    # Data
    parser.add_argument("--data_dir", type=str,
                        default="data/finegrain_uq/finetune",
                        help="Directory containing FineGRAIN JSONL files")
    parser.add_argument("--checkpoint", type=str,
                        default=DEFAULT_CHECKPOINT,
                        help="Path to v2 LoRA checkpoint to continue from")

    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate (lower than original for continued training)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=16,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warmup ratio")

    # Experiment-specific
    parser.add_argument("--held_out_model", type=str, default="sd3_xl",
                        help="Model to hold out for test (Experiments 1 and 3). "
                             "Use 'all' for full 5-fold CV in Experiment 1.")
    parser.add_argument("--exclude_flux", action="store_true",
                        help="Exclude flux from judge training data (Experiment 2)")
    parser.add_argument("--qa_replay_n", type=int, default=2000,
                        help="Number of QA replay samples (Experiment 3)")

    # Flags
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick pipeline test: 10 samples, 1 epoch")

    args = parser.parse_args()

    # Print config
    print("=" * 70)
    print("FINEGRAIN UQ FINE-TUNING")
    print("=" * 70)
    print(f"Experiment:      {args.experiment}")
    print(f"Output:          {args.output_dir}")
    print(f"Data dir:        {args.data_dir}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Learning rate:   {args.learning_rate}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch size:      {args.batch_size} x {args.grad_accum} grad accum")
    print(f"Warmup ratio:    {args.warmup_ratio}")
    if args.experiment in ("human_cv", "combined"):
        print(f"Held-out model:  {args.held_out_model}")
    if args.experiment in ("judge_only", "combined"):
        print(f"Exclude flux:    {args.exclude_flux}")
    if args.experiment == "combined":
        print(f"QA replay N:     {args.qa_replay_n}")
    print(f"Smoke test:      {args.smoke_test}")
    print()

    # Verify data exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"ERROR: Data directory not found: {data_path}")
        sys.exit(1)
    if not (data_path / "human_all.jsonl").exists():
        print(f"ERROR: human_all.jsonl not found in {data_path}")
        sys.exit(1)

    # Verify checkpoint exists
    ckpt_path = Path(args.checkpoint)
    if not (ckpt_path / "adapter_config.json").exists():
        print(f"ERROR: adapter_config.json not found in {ckpt_path}")
        sys.exit(1)

    # Dispatch to experiment mode
    if args.experiment == "human_cv":
        run_human_cv(args)
    elif args.experiment == "judge_only":
        run_judge_only(args)
    elif args.experiment == "combined":
        run_combined(args)
    else:
        print(f"Unknown experiment: {args.experiment}")
        sys.exit(1)


if __name__ == "__main__":
    main()
