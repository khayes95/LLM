#!/usr/bin/env python3
"""UC-C Stage 2: Train Student Models on Calibrator-Filtered Data.

Uses calibrator P(correct) scores to filter teacher model (GPT-5-mini) outputs,
then trains Qwen2.5-1.5B-Instruct as a student on filtered vs unfiltered data.
Compares 5 training variants to prove that calibrator filtering improves distillation.

Training Variants:
    1. all_data          — Train on all GPT-5-mini responses (unfiltered baseline)
    2. calibrator_p07    — Only responses where calibrator P(correct) > 0.7
    3. random_subsample  — Same size as variant 2, randomly selected
    4. verbalized_p07    — Only responses where verbalized_confidence > 0.7
    5. oracle_filtered   — Only responses where is_correct == 1 (ground truth ceiling)

Usage:
    # Smoke test (~5 min, 50 samples, 1 epoch, 1 variant)
    CUDA_VISIBLE_DEVICES=0 python scripts/uc_c_train_students.py --smoke_test

    # Full run (~2-3 hours on 1 GPU, all 5 variants)
    CUDA_VISIBLE_DEVICES=0 python scripts/uc_c_train_students.py

    # Specific variants only
    CUDA_VISIBLE_DEVICES=0 python scripts/uc_c_train_students.py --variants all_data,calibrator_p07
"""
import argparse
import gc
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


# ============================================================
# CONFIG
# ============================================================

STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = "runs/gpt5_mini_combined"
SCORED_FILE = "gpt5mini_scored.jsonl"

EXCLUDED_BENCHMARKS = {
    "triviaqa", "babilong", "vsr", "aokvqa", "erqa",
    "tutorbench", "healthbench", "arc", "oolong",
}

SYSTEM_PROMPT = "Answer the following question."

VARIANT_DEFS = {
    "all_data": {
        "description": "All GPT-5-mini responses (unfiltered baseline)",
        "filter_fn": lambda s: True,
    },
    "calibrator_p07": {
        "description": "Calibrator P(correct) > 0.7",
        "filter_fn": lambda s: s["p_correct"] > 0.7,
    },
    "random_subsample": {
        "description": "Random subsample (same size as calibrator_p07)",
        "filter_fn": None,  # handled specially
    },
    "verbalized_p07": {
        "description": "Verbalized confidence > 0.7",
        "filter_fn": lambda s: (
            s.get("verbalized_confidence") is not None
            and s["verbalized_confidence"] > 0.7
        ),
    },
    "oracle_filtered": {
        "description": "Oracle: only correct responses (is_correct == 1)",
        "filter_fn": lambda s: s["is_correct"] == 1,
    },
}

VARIANT_ORDER = ["all_data", "calibrator_p07", "random_subsample",
                 "verbalized_p07", "oracle_filtered"]


# ============================================================
# DATA LOADING
# ============================================================

def extract_question_text(input_data) -> str:
    """Extract question text from prediction input field."""
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


def extract_answer_text(response_text: str, prediction: dict) -> str:
    """Extract clean answer text from raw model response.

    The GPT-5-mini responses are JSON objects with 'answer', 'reasoning', etc.
    We want the full response text for SFT (student learns to produce it).
    """
    return response_text


def load_raw_predictions(data_dir: str) -> dict:
    """Load raw predictions from combined directory, keyed by (benchmark, id).

    Returns dict mapping (benchmark, id) -> {question, response, target, is_correct, ...}
    """
    samples = {}
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"WARNING: {data_dir} does not exist")
        return samples

    for bench_dir in sorted(data_path.iterdir()):
        if not bench_dir.is_dir():
            continue
        benchmark = bench_dir.name
        if benchmark in EXCLUDED_BENCHMARKS:
            continue

        pred_file = bench_dir / "predictions.jsonl"
        if not pred_file.exists() or pred_file.stat().st_size == 0:
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

                question = extract_question_text(pred.get("input", {}))
                response = pred.get("response_text", "")
                if not response:
                    response = str(
                        (pred.get("prediction") or {}).get("answer", "")
                    )
                if not question or not response:
                    continue

                target = pred.get("target", "")
                sample_id = str(pred.get("id", ""))
                key = (benchmark, sample_id)

                samples[key] = {
                    "id": sample_id,
                    "benchmark": benchmark,
                    "question": question[:2000],
                    "response": response[:2000],
                    "target": str(target)[:500],
                    "is_correct": int(correct == 1),
                }

    return samples


def load_scored_data(scored_path: str) -> dict:
    """Load scored JSONL, keyed by (benchmark, id).

    Returns dict mapping (benchmark, id) -> {p_correct, is_correct, verbalized_confidence}
    """
    scored = {}
    with open(scored_path) as f:
        for line in f:
            row = json.loads(line)
            key = (row["benchmark"], row["id"])
            vc = row.get("verbalized_confidence")
            if vc is None or str(vc).strip().lower() == "none":
                vc = None
            else:
                vc = float(vc)
            scored[key] = {
                "p_correct": float(row["p_correct"]),
                "is_correct": int(row["is_correct"]),
                "verbalized_confidence": vc,
            }
    return scored


def merge_data(raw_preds: dict, scored: dict) -> list:
    """Merge raw predictions with calibrator scores. Only keep samples present
    in both datasets."""
    merged = []
    for key in raw_preds:
        if key not in scored:
            continue
        sample = dict(raw_preds[key])
        sample.update(scored[key])
        merged.append(sample)
    return merged


def build_variant_datasets(all_samples: list, calibrator_n: int,
                           seed: int = 42) -> dict:
    """Apply filters for each variant and return {variant_name: [samples]}."""
    variants = {}
    for vname in VARIANT_ORDER:
        vdef = VARIANT_DEFS[vname]
        if vname == "random_subsample":
            # Match calibrator_p07 size via random subsampling
            rng = np.random.RandomState(seed)
            n_target = calibrator_n
            if n_target >= len(all_samples):
                variants[vname] = list(all_samples)
            else:
                indices = rng.choice(
                    len(all_samples), size=n_target, replace=False
                )
                variants[vname] = [all_samples[i] for i in sorted(indices)]
        else:
            variants[vname] = [s for s in all_samples if vdef["filter_fn"](s)]
    return variants


# ============================================================
# TRAIN/TEST SPLIT
# ============================================================

def train_test_split(all_samples: list, test_frac: float = 0.15,
                     seed: int = 42):
    """Split samples into train and test sets. The split is deterministic and
    shared across all variants (we split by question key, then each variant
    filters its own subset of the train/test pools)."""
    rng = np.random.RandomState(seed)
    n = len(all_samples)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = max(1, int(n * test_frac))
    test_idx = set(indices[:n_test].tolist())
    train_idx = set(indices[n_test:].tolist())

    train_samples = [all_samples[i] for i in sorted(train_idx)]
    test_samples = [all_samples[i] for i in sorted(test_idx)]
    return train_samples, test_samples


# ============================================================
# TOKENIZED DATASET
# ============================================================

class SFTDataset(Dataset):
    """SFT dataset: given a question, produce the teacher's response.

    Format:
        System: "Answer the following question."
        User: <question>
        Assistant: <response>

    We tokenize the full conversation and mask the labels so that only the
    assistant turn is predicted.
    """

    def __init__(self, samples: list, tokenizer, max_seq_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []

        for s in samples:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s["question"][:800]},
                {"role": "assistant", "content": s["response"][:800]},
            ]
            # Tokenize the full conversation
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            full_ids = tokenizer.encode(
                full_text, add_special_tokens=False,
                truncation=True, max_length=max_seq_length
            )

            # Tokenize up to the assistant turn to find where labels start
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s["question"][:800]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer.encode(
                prompt_text, add_special_tokens=False,
                truncation=True, max_length=max_seq_length
            )

            # Build labels: -100 for prompt tokens, actual ids for response
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            # Ensure same length
            if len(labels) < len(full_ids):
                labels = labels + full_ids[len(labels):]
            labels = labels[:len(full_ids)]

            if len(full_ids) < 5:
                continue

            self.examples.append({
                "input_ids": full_ids,
                "labels": labels,
                "attention_mask": [1] * len(full_ids),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(
                ex["attention_mask"], dtype=torch.long
            ),
        }


# ============================================================
# GRADING
# ============================================================

def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    text = text.strip().lower()
    # Remove common wrapper tokens
    for prefix in ["answer:", "the answer is", "answer is"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # Remove trailing punctuation
    text = text.rstrip(".,;:!?")
    return text.strip()


def extract_answer_from_response(response: str, target: str) -> str:
    """Extract the final answer from a student's free-form response.

    Strategies:
    1. Try to parse as JSON (GPT-style {"answer": "..."})
    2. For MCQ (target is A/B/C/D): find last letter match
    3. For numeric: extract numbers
    4. Fallback: use the whole response, normalized
    """
    response = response.strip()

    # Try JSON parsing first
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            ans = parsed.get("answer", "")
            if ans:
                return str(ans).strip()
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON-like answer field in text
    json_match = re.search(r'"answer"\s*:\s*"([^"]*)"', response)
    if json_match:
        return json_match.group(1).strip()

    target_norm = normalize_answer(target)

    # MCQ: target is a single letter A-D
    if re.fullmatch(r'[a-d]', target_norm):
        # Find all letter matches, prefer the last one (final answer)
        letters = re.findall(r'\b([A-Da-d])\b', response)
        if letters:
            return letters[-1].upper()
        # Also look for (A), (B), etc.
        paren_letters = re.findall(r'\(([A-Da-d])\)', response)
        if paren_letters:
            return paren_letters[-1].upper()

    # Numeric: target is a number
    if re.fullmatch(r'[-+]?\d*\.?\d+', target_norm):
        numbers = re.findall(r'[-+]?\d*\.?\d+', response)
        if numbers:
            return numbers[-1]

    # Fallback: return full response normalized
    return response


def grade_response(student_response: str, target: str) -> int:
    """Grade a student response against the ground truth target.

    Returns 1 if correct, 0 if incorrect.
    """
    if not student_response or not target:
        return 0

    extracted = extract_answer_from_response(student_response, target)
    extracted_norm = normalize_answer(extracted)
    target_norm = normalize_answer(target)

    # Exact match after normalization
    if extracted_norm == target_norm:
        return 1

    # MCQ letter comparison
    if (
        re.fullmatch(r'[a-d]', target_norm)
        and re.fullmatch(r'[a-d]', extracted_norm)
    ):
        return int(extracted_norm == target_norm)

    # Numeric comparison with tolerance
    try:
        ex_num = float(extracted_norm)
        tgt_num = float(target_norm)
        if tgt_num == 0:
            return int(abs(ex_num) < 1e-6)
        return int(abs(ex_num - tgt_num) / max(abs(tgt_num), 1e-9) < 0.01)
    except (ValueError, ZeroDivisionError):
        pass

    # Substring containment (target in extracted or vice versa)
    if target_norm in extracted_norm or extracted_norm in target_norm:
        return 1

    return 0


# ============================================================
# TRAINING
# ============================================================

def train_variant(variant_name: str, train_samples: list,
                  eval_samples: list, tokenizer, model_name: str,
                  output_dir: str, epochs: int, batch_size: int,
                  learning_rate: float, max_seq_length: int,
                  smoke_test: bool = False):
    """Train one student model variant and return the path to the best checkpoint."""
    print(f"\n{'='*70}")
    print(f"TRAINING VARIANT: {variant_name}")
    print(f"  {VARIANT_DEFS[variant_name]['description']}")
    print(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}")

    if len(train_samples) < 5:
        print(f"  WARNING: Too few training samples ({len(train_samples)}), skipping.")
        return None

    # Build datasets
    print("  Building tokenized datasets...")
    train_dataset = SFTDataset(train_samples, tokenizer, max_seq_length)
    eval_dataset = SFTDataset(eval_samples, tokenizer, max_seq_length)
    print(f"  Train examples: {len(train_dataset)}, Eval examples: {len(eval_dataset)}")

    if len(train_dataset) < 2:
        print(f"  WARNING: Too few tokenized examples, skipping.")
        return None

    # Load fresh model
    print(f"  Loading base model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    gradient_accumulation = 4
    effective_batch = batch_size * gradient_accumulation
    logging_steps = max(1, len(train_dataset) // (effective_batch * 10))

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=logging_steps if logging_steps > 0 else 1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=max_seq_length,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("  Starting training...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    print(f"  Training complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save best model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Cleanup
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


# ============================================================
# EVALUATION
# ============================================================

def evaluate_student(model_path: str, test_samples: list,
                     tokenizer, max_new_tokens: int = 512,
                     batch_size: int = 8) -> list:
    """Generate student responses on test samples and grade them.

    Returns list of dicts with student_response, target, is_correct, benchmark.
    """
    print(f"  Loading student model from {model_path}...")
    from peft import PeftModel as PM

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PM.from_pretrained(model, model_path)
    model.eval()

    device = next(model.parameters()).device
    results = []

    print(f"  Generating responses for {len(test_samples)} test samples...")
    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i:i + batch_size]
        for sample in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample["question"][:800]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                prompt_text, return_tensors="pt",
                truncation=True, max_length=1024
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            student_response = tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            correct = grade_response(student_response, sample["target"])
            results.append({
                "id": sample["id"],
                "benchmark": sample["benchmark"],
                "question": sample["question"][:200],
                "target": sample["target"],
                "teacher_response": sample["response"][:200],
                "student_response": student_response[:500],
                "is_correct": correct,
                "teacher_was_correct": sample.get("is_correct", -1),
            })

        if (i + batch_size) % 50 < batch_size or i + batch_size >= len(test_samples):
            done = min(i + batch_size, len(test_samples))
            n_correct = sum(r["is_correct"] for r in results)
            print(f"    [{done}/{len(test_samples)}] "
                  f"Accuracy so far: {n_correct/max(len(results),1):.3f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ============================================================
# FIGURES
# ============================================================

def plot_student_comparison(all_variant_results: dict, fig_path: str):
    """Bar chart comparing student accuracy across variants."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    names = []
    accs = []
    train_sizes = []
    colors = []
    color_map = {
        "all_data": "C3",
        "calibrator_p07": "C0",
        "random_subsample": "gray",
        "verbalized_p07": "C1",
        "oracle_filtered": "C2",
    }
    label_map = {
        "all_data": "All Data\n(Unfiltered)",
        "calibrator_p07": "Calibrator\n(p>0.7)",
        "random_subsample": "Random\nSubsample",
        "verbalized_p07": "Verbalized\n(conf>0.7)",
        "oracle_filtered": "Oracle\n(Correct Only)",
    }

    for vname in VARIANT_ORDER:
        if vname not in all_variant_results:
            continue
        vr = all_variant_results[vname]
        names.append(label_map.get(vname, vname))
        accs.append(vr["student_accuracy"])
        train_sizes.append(vr["train_size"])
        colors.append(color_map.get(vname, "C4"))

    x = np.arange(len(names))
    bars = ax.bar(x, accs, color=colors, alpha=0.85, edgecolor="black",
                  linewidth=0.5)

    # Annotate with accuracy and training size
    for i, (bar, acc, n) in enumerate(zip(bars, accs, train_sizes)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}\n(n={n})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Student Accuracy on Held-Out Test", fontsize=12)
    ax.set_title("UC-C Stage 2: Student Accuracy by Training Data Filter",
                 fontsize=14)
    ax.set_ylim(0, min(1.0, max(accs) * 1.3) if accs else 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    # Add horizontal line for unfiltered baseline
    if "all_data" in all_variant_results:
        baseline = all_variant_results["all_data"]["student_accuracy"]
        ax.axhline(y=baseline, color="C3", linestyle=":", alpha=0.5,
                    label=f"Unfiltered baseline ({baseline:.3f})")
        ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {fig_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="UC-C Stage 2: Train Student Models on Calibrator-Filtered Data"
    )
    parser.add_argument("--scored_dir", default="data/use_cases/scored_unified",
                        help="Directory with scored JSONL files")
    parser.add_argument("--output_dir", default="data/use_cases/results_unified",
                        help="Directory for results JSON")
    parser.add_argument("--fig_dir", default="figures/use_cases_unified",
                        help="Directory for output figures")
    parser.add_argument("--model_dir", default="uq_models/uc_c_students",
                        help="Directory to save trained student models")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--smoke_test", action="store_true",
                        help="Only 50 samples, 1 epoch, 1 variant")
    parser.add_argument("--variants", type=str, default="all",
                        help="Comma-separated list of variants to train, or 'all'")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, only evaluate existing models")
    args = parser.parse_args()

    # Parse variants
    if args.variants == "all":
        variants_to_run = list(VARIANT_ORDER)
    else:
        variants_to_run = [v.strip() for v in args.variants.split(",")]
        for v in variants_to_run:
            if v not in VARIANT_DEFS:
                print(f"ERROR: Unknown variant '{v}'. "
                      f"Valid: {', '.join(VARIANT_ORDER)}")
                sys.exit(1)

    if args.smoke_test:
        variants_to_run = [variants_to_run[0]]
        args.epochs = 1
        print("SMOKE TEST MODE: 1 variant, 1 epoch, 50 samples")

    # Create directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # 1. Load and merge data
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print("STAGE 1: Loading Data")
    print(f"{'='*70}")

    scored_path = Path(args.scored_dir) / SCORED_FILE
    if not scored_path.exists():
        print(f"ERROR: Scored file not found: {scored_path}")
        sys.exit(1)

    print(f"Loading raw predictions from {DATA_DIR}...")
    raw_preds = load_raw_predictions(DATA_DIR)
    print(f"  Loaded {len(raw_preds)} raw predictions")

    print(f"Loading calibrator scores from {scored_path}...")
    scored = load_scored_data(str(scored_path))
    print(f"  Loaded {len(scored)} scored samples")

    print("Merging data...")
    all_samples = merge_data(raw_preds, scored)
    print(f"  Merged: {len(all_samples)} samples")

    if args.smoke_test:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_samples), size=min(50, len(all_samples)),
                             replace=False)
        all_samples = [all_samples[i] for i in sorted(indices)]
        print(f"  Smoke test: using {len(all_samples)} samples")

    # Print stats
    n_correct = sum(s["is_correct"] for s in all_samples)
    print(f"\n  Total: {len(all_samples)} samples, "
          f"{n_correct} correct ({n_correct/len(all_samples):.1%})")
    bench_counts = defaultdict(int)
    for s in all_samples:
        bench_counts[s["benchmark"]] += 1
    print(f"  Benchmarks ({len(bench_counts)}): "
          + ", ".join(f"{b}({c})" for b, c in sorted(bench_counts.items())))

    # -------------------------------------------------------
    # 2. Train/test split (shared across all variants)
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print("STAGE 2: Train/Test Split")
    print(f"{'='*70}")

    train_pool, test_pool = train_test_split(all_samples, test_frac=0.15,
                                             seed=42)
    print(f"  Train pool: {len(train_pool)} samples")
    print(f"  Test pool:  {len(test_pool)} samples")

    # -------------------------------------------------------
    # 3. Build variant datasets
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print("STAGE 3: Building Variant Datasets")
    print(f"{'='*70}")

    # Calculate calibrator_p07 size first (needed for random_subsample)
    cal_train = [s for s in train_pool
                 if VARIANT_DEFS["calibrator_p07"]["filter_fn"](s)]
    calibrator_n = len(cal_train)
    print(f"  Calibrator (p>0.7) train size: {calibrator_n}")

    variant_trains = build_variant_datasets(train_pool, calibrator_n, seed=42)

    # The test set is always the FULL test pool (unfiltered) for fair comparison
    # We evaluate all variants on the same test questions
    test_set = test_pool

    for vname in variants_to_run:
        vtrain = variant_trains.get(vname, [])
        n_correct_v = sum(s["is_correct"] for s in vtrain)
        teacher_acc = n_correct_v / max(len(vtrain), 1)
        print(f"  {vname:20s}: {len(vtrain):5d} train samples, "
              f"teacher acc = {teacher_acc:.3f}")

    # -------------------------------------------------------
    # 4. Load tokenizer
    # -------------------------------------------------------
    print(f"\nLoading tokenizer: {STUDENT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(
        STUDENT_MODEL, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # -------------------------------------------------------
    # 5. Train and evaluate each variant
    # -------------------------------------------------------
    all_variant_results = {}

    for vname in variants_to_run:
        vtrain = variant_trains.get(vname, [])

        if len(vtrain) < 5:
            print(f"\n  Skipping {vname}: too few samples ({len(vtrain)})")
            continue

        variant_model_dir = os.path.join(args.model_dir, vname)
        Path(variant_model_dir).mkdir(parents=True, exist_ok=True)

        # --- Train ---
        if not args.skip_training:
            # Build eval subset for training (small, from the variant's own data)
            rng = np.random.RandomState(42)
            n_eval = max(1, min(100, len(vtrain) // 5))
            eval_idx = rng.choice(len(vtrain), size=n_eval, replace=False)
            eval_set_train = [vtrain[i] for i in eval_idx]
            train_set = [vtrain[i] for i in range(len(vtrain))
                         if i not in set(eval_idx)]

            model_path = train_variant(
                variant_name=vname,
                train_samples=train_set,
                eval_samples=eval_set_train,
                tokenizer=tokenizer,
                model_name=STUDENT_MODEL,
                output_dir=variant_model_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_seq_length=args.max_seq_length,
                smoke_test=args.smoke_test,
            )
        else:
            model_path = variant_model_dir
            if not Path(model_path).exists():
                print(f"  Skipping {vname}: model dir not found at {model_path}")
                continue

        if model_path is None:
            continue

        # --- Evaluate ---
        print(f"\n  Evaluating {vname} on held-out test set...")
        eval_results = evaluate_student(
            model_path=model_path,
            test_samples=test_set,
            tokenizer=tokenizer,
            max_new_tokens=512,
        )

        # Compute metrics
        n_total_eval = len(eval_results)
        n_correct_eval = sum(r["is_correct"] for r in eval_results)
        student_acc = n_correct_eval / max(n_total_eval, 1)

        # Per-benchmark breakdown
        bench_results = defaultdict(lambda: {"total": 0, "correct": 0})
        for r in eval_results:
            bench_results[r["benchmark"]]["total"] += 1
            bench_results[r["benchmark"]]["correct"] += r["is_correct"]
        per_bench = {}
        for b in sorted(bench_results.keys()):
            br = bench_results[b]
            per_bench[b] = {
                "n_total": br["total"],
                "n_correct": br["correct"],
                "accuracy": br["correct"] / max(br["total"], 1),
            }

        # Training data stats
        n_correct_train = sum(s["is_correct"] for s in vtrain)
        teacher_acc_train = n_correct_train / max(len(vtrain), 1)

        variant_result = {
            "variant": vname,
            "description": VARIANT_DEFS[vname]["description"],
            "train_size": len(vtrain),
            "teacher_accuracy": float(teacher_acc_train),
            "test_size": n_total_eval,
            "student_correct": n_correct_eval,
            "student_accuracy": float(student_acc),
            "per_benchmark": per_bench,
            "model_path": variant_model_dir,
        }
        all_variant_results[vname] = variant_result

        print(f"\n  RESULT: {vname}")
        print(f"    Train size:       {len(vtrain)}")
        print(f"    Teacher accuracy: {teacher_acc_train:.3f}")
        print(f"    Student accuracy: {student_acc:.3f} "
              f"({n_correct_eval}/{n_total_eval})")

        print(f"\n    Per-benchmark breakdown:")
        print(f"    {'Benchmark':<20} {'N':>5} {'Correct':>8} {'Acc':>7}")
        print(f"    {'-'*44}")
        for b, br in sorted(per_bench.items()):
            print(f"    {b:<20} {br['n_total']:>5} "
                  f"{br['n_correct']:>8} {br['accuracy']:>7.3f}")

    # -------------------------------------------------------
    # 6. Summary and comparison
    # -------------------------------------------------------
    if not all_variant_results:
        print("\nERROR: No variants completed successfully.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("FINAL COMPARISON: Student Accuracy by Variant")
    print(f"{'='*70}")
    print(f"  {'Variant':<22} {'Train':>7} {'Teacher':>8} {'Student':>8} {'Delta':>7}")
    print(f"  {'-'*56}")

    baseline_acc = None
    if "all_data" in all_variant_results:
        baseline_acc = all_variant_results["all_data"]["student_accuracy"]

    for vname in VARIANT_ORDER:
        if vname not in all_variant_results:
            continue
        vr = all_variant_results[vname]
        delta = ""
        if baseline_acc is not None and vname != "all_data":
            d = vr["student_accuracy"] - baseline_acc
            delta = f"{d:+.3f}"
        print(f"  {vname:<22} {vr['train_size']:>7} "
              f"{vr['teacher_accuracy']:>8.3f} "
              f"{vr['student_accuracy']:>8.3f} {delta:>7}")

    # -------------------------------------------------------
    # 7. Save results
    # -------------------------------------------------------
    results_path = Path(args.output_dir) / "uc_c_s2_results.json"
    output = {
        "description": "UC-C Stage 2: Student model accuracy by training data filter",
        "student_model": STUDENT_MODEL,
        "test_size": len(test_set),
        "total_samples": len(all_samples),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "variants": all_variant_results,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # -------------------------------------------------------
    # 8. Generate figure
    # -------------------------------------------------------
    if len(all_variant_results) >= 2:
        fig_path = Path(args.fig_dir) / "uc_c_student_comparison.pdf"
        plot_student_comparison(all_variant_results, str(fig_path))

    print(f"\n{'='*70}")
    print("UC-C Stage 2 Complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
