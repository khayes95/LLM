"""
UQ Fine-tuning: Train a model to predict correctness of LLM answers.

Based on "Large Language Models Must Be Taught to Know What They Don't Know"
(Kapoor et al., NeurIPS 2024).

The "LoRA + Prompt" method:
- Prompt: "Question: ... Answer: ... Is the answer correct? (i) No (ii) Yes"
- Train to predict single token: "i" (incorrect) or "ii" (correct)
- Inference: P(correct) = P("ii") / (P("i") + P("ii"))

This module provides:
- Data preparation from evaluation predictions
- Fine-tuning loop for correctness prediction
- Support for LoRA for efficient training
- Inference function for calibrated confidence scores
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import roc_auc_score


@dataclass
class UQTrainingConfig:
    """Configuration for UQ fine-tuning.

    Hyperparameters from Kapoor et al. (2024):
    - LoRA rank r=8, alpha=32, dropout=0.1
    - Learning rate 1e-4 with cosine decay
    - Effective batch size 32
    """

    # Model
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    # LoRA config (from paper)
    use_lora: bool = True
    lora_r: int = 8  # Paper uses r=8
    lora_alpha: int = 32
    lora_dropout: float = 0.1  # Paper uses 0.1
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # Training (from paper)
    output_dir: str = "uq_models"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Effective batch size = 4 * 8 = 32
    learning_rate: float = 1e-4  # Paper uses 1e-4
    warmup_ratio: float = 0.1  # Paper uses 10% warmup
    weight_decay: float = 0.01
    max_seq_length: int = 2048

    # Data
    train_split: float = 0.9
    balance_classes: bool = True  # Balance correct/incorrect examples


class UQDataset(Dataset):
    """Dataset for UQ fine-tuning from evaluation predictions."""

    def __init__(
        self,
        predictions_paths: list[Path],
        tokenizer,
        max_length: int = 2048,
        mode: str = "classifier",  # "classifier" or "verbalized"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.examples = []

        # Load predictions from all benchmark runs
        for path in predictions_paths:
            self._load_predictions(path)

        print(f"Loaded {len(self.examples)} examples")
        correct = sum(1 for e in self.examples if e["correct"])
        print(f"  Correct: {correct}, Incorrect: {len(self.examples) - correct}")

    def _load_predictions(self, path: Path):
        """Load predictions from JSONL file."""
        if not path.exists():
            print(f"Warning: {path} not found")
            return

        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)

                # Extract relevant fields
                question = row.get("input", "")
                answer = row.get("prediction", {}).get("answer", "")
                raw_text = row.get("response_text", "")
                correct = row.get("score", {}).get("correct", 0)
                gold = row.get("target", "")

                if not question or not answer:
                    continue

                self.examples.append({
                    "question": str(question),
                    "answer": str(answer),
                    "raw_response": str(raw_text),
                    "correct": int(correct) == 1,
                    "gold": str(gold),
                    "benchmark": row.get("id", "").split("_")[0] if row.get("id") else "unknown",
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Paper format: predict single token "i" (incorrect) or "ii" (correct)
        prompt = self._format_classifier_prompt(ex)
        label = "ii" if ex["correct"] else "i"

        # Tokenize prompt + label
        full_text = prompt + label

        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        # Create labels (mask prompt tokens with -100, only train on label token)
        prompt_encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        prompt_len = len(prompt_encodings["input_ids"])

        labels = [-100] * prompt_len + encodings["input_ids"][prompt_len:]

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }

    def _format_classifier_prompt(self, ex: dict) -> str:
        """Format example per Kapoor et al. (2024) paper."""
        return (
            f"Question: {ex['question']}\n\n"
            f"Answer: {ex['answer']}\n\n"
            f"Is the answer correct? (i) No (ii) Yes\n\n"
        )


def balance_dataset(examples: list[dict]) -> list[dict]:
    """Balance correct/incorrect examples."""
    correct = [e for e in examples if e["correct"]]
    incorrect = [e for e in examples if not e["correct"]]

    min_count = min(len(correct), len(incorrect))

    if min_count == 0:
        return examples

    random.shuffle(correct)
    random.shuffle(incorrect)

    balanced = correct[:min_count] + incorrect[:min_count]
    random.shuffle(balanced)

    return balanced


def prepare_uq_data(
    runs_dir: Path,
    tokenizer,
    config: UQTrainingConfig,
) -> tuple[UQDataset, UQDataset]:
    """Prepare train/val datasets from evaluation runs."""

    # Find all predictions.jsonl files
    predictions_paths = list(runs_dir.glob("*/predictions.jsonl"))
    print(f"Found {len(predictions_paths)} prediction files")

    # Load all data
    all_examples = []
    for path in predictions_paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)

                question = row.get("input", "")
                answer = row.get("prediction", {}).get("answer", "")
                raw_text = row.get("response_text", "")
                correct = row.get("score", {}).get("correct", 0)
                gold = row.get("target", "")

                if not question or not answer:
                    continue

                # Skip API failures
                if "API_FAILED" in str(raw_text):
                    continue

                all_examples.append({
                    "question": str(question),
                    "answer": str(answer),
                    "raw_response": str(raw_text),
                    "correct": int(correct) == 1,
                    "gold": str(gold),
                    "benchmark": row.get("id", "").split("_")[0] if row.get("id") else "unknown",
                })

    print(f"Total examples: {len(all_examples)}")

    # Balance if requested
    if config.balance_classes:
        all_examples = balance_dataset(all_examples)
        print(f"After balancing: {len(all_examples)}")

    # Split
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * config.train_split)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    # Create datasets
    train_dataset = UQDataset.__new__(UQDataset)
    train_dataset.tokenizer = tokenizer
    train_dataset.max_length = config.max_seq_length
    train_dataset.mode = "classifier"
    train_dataset.examples = train_examples

    val_dataset = UQDataset.__new__(UQDataset)
    val_dataset.tokenizer = tokenizer
    val_dataset.max_length = config.max_seq_length
    val_dataset.mode = "classifier"
    val_dataset.examples = val_examples

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    return train_dataset, val_dataset


def train_uq_model(
    config: UQTrainingConfig,
    runs_dir: Path,
):
    """Train UQ model on evaluation predictions."""

    print(f"Loading base model: {config.base_model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA if configured
    if config.use_lora:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Prepare data
    train_dataset, val_dataset = prepare_uq_data(runs_dir, tokenizer, config)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )

    # Training arguments (following paper: cosine schedule, warmup)
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",  # Paper uses cosine decay
        warmup_ratio=config.warmup_ratio,  # Paper uses 10% warmup
        weight_decay=config.weight_decay,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        report_to="none",
        dataloader_num_workers=4,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    return model, tokenizer


def load_jsonl_data(train_path: Path, val_path: Path, tokenizer, config: UQTrainingConfig):
    """Load pre-balanced train/val data from JSONL files."""

    def load_examples(path: Path) -> list[dict]:
        examples = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                examples.append({
                    "question": str(row.get("input", "")),
                    "answer": str(row.get("prediction", {}).get("answer", "")),
                    "raw_response": str(row.get("model_response", "")),
                    "correct": row.get("correct", 0) == 1,
                    "gold": str(row.get("target", "")),
                    "benchmark": row.get("benchmark", "unknown"),
                })
        return examples

    train_examples = load_examples(train_path)
    val_examples = load_examples(val_path)

    print(f"Loaded {len(train_examples)} train, {len(val_examples)} val examples")

    # Create datasets
    train_dataset = UQDataset.__new__(UQDataset)
    train_dataset.tokenizer = tokenizer
    train_dataset.max_length = config.max_seq_length
    train_dataset.mode = "classifier"
    train_dataset.examples = train_examples

    val_dataset = UQDataset.__new__(UQDataset)
    val_dataset.tokenizer = tokenizer
    val_dataset.max_length = config.max_seq_length
    val_dataset.mode = "classifier"
    val_dataset.examples = val_examples

    return train_dataset, val_dataset


def train_uq_model_from_jsonl(
    config: UQTrainingConfig,
    train_path: Path,
    val_path: Path,
):
    """Train UQ model on pre-prepared JSONL data."""

    print(f"Loading base model: {config.base_model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA if configured
    if config.use_lora:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load data
    train_dataset, val_dataset = load_jsonl_data(train_path, val_path, tokenizer, config)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )

    # Training arguments (following paper: cosine schedule, warmup)
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",  # Paper uses cosine decay
        warmup_ratio=config.warmup_ratio,  # Paper uses 10% warmup
        weight_decay=config.weight_decay,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        report_to="none",
        dataloader_num_workers=4,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    return model, tokenizer


# =============================================================================
# Inference Functions
# =============================================================================

def format_uq_prompt(question: str, answer: str) -> str:
    """Format a question/answer pair for UQ inference."""
    return (
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Is the answer correct? (i) No (ii) Yes\n\n"
    )


def _verify_tokens(tokenizer) -> tuple[int, int]:
    """Verify that 'i' and 'ii' encode to different single tokens."""
    tokens_i = tokenizer.encode("i", add_special_tokens=False)
    tokens_ii = tokenizer.encode("ii", add_special_tokens=False)

    if len(tokens_i) != 1 or len(tokens_ii) != 1:
        print(f"WARNING: Token encoding issue!")
        print(f"  'i' encodes to: {tokens_i}")
        print(f"  'ii' encodes to: {tokens_ii}")

    if tokens_i[0] == tokens_ii[0]:
        raise ValueError("'i' and 'ii' encode to the same first token! Need different target tokens.")

    return tokens_i[0], tokens_ii[0]


def get_confidence(
    model,
    tokenizer,
    question: str,
    answer: str,
    temperature: float = 1.0,
) -> float:
    """Get calibrated confidence score for an answer.

    Args:
        model: The fine-tuned UQ model
        tokenizer: The tokenizer
        question: The question that was asked
        answer: The model's answer to evaluate
        temperature: Temperature scaling parameter (from calibration)

    Returns:
        Confidence score P(correct) in [0, 1]
    """
    # Verify token encoding
    token_i, token_ii = _verify_tokens(tokenizer)

    # Format prompt
    prompt = format_uq_prompt(question, answer)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        # Get logits at the last position
        logits = outputs.logits[0, -1, :]

        # Extract logits for our two tokens
        logit_i = logits[token_i].item()
        logit_ii = logits[token_ii].item()

        # Apply temperature scaling and softmax
        logits_pair = torch.tensor([logit_i, logit_ii]) / temperature
        probs = torch.softmax(logits_pair, dim=0)

        # P(correct) = P("ii")
        confidence = probs[1].item()

    return confidence


def get_confidence_batch(
    model,
    tokenizer,
    examples: list[dict],
    temperature: float = 1.0,
    batch_size: int = 8,
) -> list[float]:
    """Get confidence scores for a batch of examples.

    Args:
        model: The fine-tuned UQ model
        tokenizer: The tokenizer
        examples: List of dicts with 'question' and 'answer' keys
        temperature: Temperature scaling parameter
        batch_size: Batch size for inference

    Returns:
        List of confidence scores
    """
    confidences = []

    # Verify and get token IDs
    token_i, token_ii = _verify_tokens(tokenizer)

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        prompts = [format_uq_prompt(ex["question"], ex["answer"]) for ex in batch]

        # Tokenize with padding
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

            # For each example in batch, get logits at last non-pad position
            for j in range(len(batch)):
                # Find last non-pad position
                attention_mask = inputs["attention_mask"][j]
                last_pos = attention_mask.sum().item() - 1

                logits = outputs.logits[j, last_pos, :]
                logit_i = logits[token_i].item()
                logit_ii = logits[token_ii].item()

                # Apply temperature and softmax
                logits_pair = torch.tensor([logit_i, logit_ii]) / temperature
                probs = torch.softmax(logits_pair, dim=0)
                confidences.append(probs[1].item())

    return confidences


# =============================================================================
# Temperature Calibration
# =============================================================================

def calibrate_temperature(
    model,
    tokenizer,
    val_examples: list[dict],
    batch_size: int = 8,
) -> float:
    """Learn optimal temperature for calibration using validation set.

    Finds T that minimizes negative log-likelihood on validation data.

    Args:
        model: The fine-tuned UQ model
        tokenizer: The tokenizer
        val_examples: List of dicts with 'question', 'answer', 'correct' keys
        batch_size: Batch size for inference

    Returns:
        Optimal temperature T
    """
    from scipy.optimize import minimize_scalar

    # Verify and get token IDs
    token_i, token_ii = _verify_tokens(tokenizer)

    all_logits = []  # List of (logit_i, logit_ii, correct)

    print(f"Getting logits for {len(val_examples)} validation examples...")

    for i in range(0, len(val_examples), batch_size):
        batch = val_examples[i:i + batch_size]
        prompts = [format_uq_prompt(ex["question"], ex["answer"]) for ex in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

            for j, ex in enumerate(batch):
                attention_mask = inputs["attention_mask"][j]
                last_pos = attention_mask.sum().item() - 1

                logits = outputs.logits[j, last_pos, :]
                logit_i = logits[token_i].item()
                logit_ii = logits[token_ii].item()

                all_logits.append((logit_i, logit_ii, ex["correct"]))

    # Define NLL loss as function of temperature
    def nll_loss(T):
        total_loss = 0.0
        for logit_i, logit_ii, correct in all_logits:
            # Apply temperature
            logits_scaled = torch.tensor([logit_i, logit_ii]) / T
            log_probs = torch.log_softmax(logits_scaled, dim=0)

            # Target is index 1 if correct, index 0 if incorrect
            target = 1 if correct else 0
            total_loss -= log_probs[target].item()

        return total_loss / len(all_logits)

    # Optimize temperature
    print("Optimizing temperature...")
    result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
    optimal_T = result.x

    print(f"Optimal temperature: {optimal_T:.4f}")
    print(f"NLL before (T=1): {nll_loss(1.0):.4f}")
    print(f"NLL after (T={optimal_T:.2f}): {nll_loss(optimal_T):.4f}")

    return optimal_T


def evaluate_calibration(
    model,
    tokenizer,
    val_examples: list[dict],
    temperature: float = 1.0,
) -> dict:
    """Evaluate calibration metrics on validation set.

    Returns:
        Dict with accuracy, ECE, Brier score, etc.
    """
    confidences = get_confidence_batch(model, tokenizer, val_examples, temperature)

    correct = [ex["correct"] for ex in val_examples]
    predictions = [c > 0.5 for c in confidences]

    # Accuracy
    accuracy = sum(p == c for p, c in zip(predictions, correct)) / len(correct)

    # Brier score
    brier = sum((c - int(corr)) ** 2 for c, corr in zip(confidences, correct)) / len(correct)

    # ECE (Expected Calibration Error) - 10 bins
    n_bins = 10
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    bin_correct = [0] * n_bins
    bin_conf = [0] * n_bins
    bin_count = [0] * n_bins

    for conf, corr in zip(confidences, correct):
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bin_correct[bin_idx] += int(corr)
        bin_conf[bin_idx] += conf
        bin_count[bin_idx] += 1

    ece = 0.0
    for i in range(n_bins):
        if bin_count[i] > 0:
            avg_conf = bin_conf[i] / bin_count[i]
            avg_acc = bin_correct[i] / bin_count[i]
            ece += bin_count[i] * abs(avg_conf - avg_acc)
    ece /= len(correct)

    # AUROC (key metric from paper)
    try:
        auroc = roc_auc_score(correct, confidences)
    except ValueError:
        auroc = 0.5  # If all labels are same class

    return {
        "accuracy": accuracy,
        "brier_score": brier,
        "ece": ece,
        "auroc": auroc,
        "mean_confidence": sum(confidences) / len(confidences),
        "temperature": temperature,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train UQ model following Kapoor et al. (2024)")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--train_file", default="data/finetune/train.jsonl")
    parser.add_argument("--val_file", default="data/finetune/val.jsonl")
    parser.add_argument("--runs_dir", default=None, help="Directory with evaluation runs (alternative to train/val files)")
    parser.add_argument("--output_dir", default="uq_models/uq_classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (paper uses 1e-4)")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (paper uses 8)")
    parser.add_argument("--calibrate", action="store_true", help="Run temperature calibration after training")
    args = parser.parse_args()

    config = UQTrainingConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
    )

    if args.runs_dir:
        model, tokenizer = train_uq_model(config, Path(args.runs_dir))
    else:
        model, tokenizer = train_uq_model_from_jsonl(config, Path(args.train_file), Path(args.val_file))

    # Optional: calibrate temperature on validation set
    if args.calibrate:
        print("\n" + "=" * 50)
        print("TEMPERATURE CALIBRATION")
        print("=" * 50)

        # Load validation examples
        val_examples = []
        with open(args.val_file) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    val_examples.append({
                        "question": row.get("input", ""),
                        "answer": row.get("prediction", {}).get("answer", ""),
                        "correct": row.get("correct", 0) == 1,
                    })

        # Calibrate
        optimal_T = calibrate_temperature(model, tokenizer, val_examples)

        # Evaluate before and after
        print("\nCalibration Results:")
        metrics_before = evaluate_calibration(model, tokenizer, val_examples, temperature=1.0)
        metrics_after = evaluate_calibration(model, tokenizer, val_examples, temperature=optimal_T)

        print(f"Before (T=1.0): Acc={metrics_before['accuracy']:.3f}, ECE={metrics_before['ece']:.3f}, AUROC={metrics_before['auroc']:.3f}, Brier={metrics_before['brier_score']:.3f}")
        print(f"After  (T={optimal_T:.2f}): Acc={metrics_after['accuracy']:.3f}, ECE={metrics_after['ece']:.3f}, AUROC={metrics_after['auroc']:.3f}, Brier={metrics_after['brier_score']:.3f}")

        # Save temperature
        import json as json_module
        with open(Path(args.output_dir) / "calibration.json", "w") as f:
            json_module.dump({
                "temperature": optimal_T,
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
            }, f, indent=2)
        print(f"\nSaved calibration to {args.output_dir}/calibration.json")
