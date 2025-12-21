#!/usr/bin/env bash
#SBATCH -J uq_ft_v2
#SBATCH -p GPU
#SBATCH -w gpunode00
#SBATCH -t 4:00:00
#SBATCH --cpus-per-task=8
#SBATCH -o logs/uq_finetune_v2_%j.out
#SBATCH -e logs/uq_finetune_v2_%j.err

# LoRA finetuning for UQ classifier - Version 2 with proper train/val/test split
# Following Kapoor et al. (2024) "LLMs Must Be Taught to Know What They Don't Know"
# Usage: sbatch scripts/run_finetune_v2.sh

set -e

cd /scratch/$USER/LLM

source /scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

# Use GPUs 4,5,6,7 (free)
export CUDA_VISIBLE_DEVICES=4,5,6,7

mkdir -p logs uq_models

echo "=========================================="
echo "UQ MODEL FINETUNING V2 (LoRA + Prompt)"
echo "With proper train/val/test split"
echo "Following Kapoor et al. (2024)"
echo "Started at: $(date)"
echo "=========================================="

# Show data stats
echo "Training data (v2 - balanced splits):"
wc -l data/finetune/train_v2.jsonl data/finetune/val_v2.jsonl data/finetune/test_v2.jsonl

echo ""
echo "Hyperparameters (from paper):"
echo "  - LoRA rank: 8"
echo "  - Learning rate: 1e-4"
echo "  - Batch size: 4 x 8 = 32 effective"
echo "  - Scheduler: cosine"
echo ""

echo "Starting finetuning..."

python -m uq_eval.uq_finetune \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --train_file data/finetune/train_v2.jsonl \
    --val_file data/finetune/val_v2.jsonl \
    --output_dir uq_models/llama-8b-uq-lora-v2 \
    --epochs 3 \
    --batch_size 4 \
    --lr 1e-4 \
    --lora_r 8 \
    --calibrate

echo ""
echo "=========================================="
echo "TRAINING COMPLETE - Now evaluating on test set"
echo "=========================================="

# Evaluate on held-out test set
python3 << 'PYEOF'
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading model for test evaluation...")

# Load base model and LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "uq_models/llama-8b-uq-lora-v2")
tokenizer = AutoTokenizer.from_pretrained("uq_models/llama-8b-uq-lora-v2")

# Load calibration temperature
with open("uq_models/llama-8b-uq-lora-v2/calibration.json") as f:
    calib = json.load(f)
temperature = calib["temperature"]
print(f"Using temperature: {temperature:.4f}")

# Load test examples
test_examples = []
with open("data/finetune/test_v2.jsonl") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            test_examples.append({
                "question": row.get("input", ""),
                "answer": row.get("model_response", ""),
                "correct": row.get("correct", 0) == 1,
            })

print(f"Loaded {len(test_examples)} test examples")

# Import evaluation function
from uq_eval.uq_finetune import evaluate_calibration

# Evaluate
metrics = evaluate_calibration(model, tokenizer, test_examples, temperature=temperature)

print("\n" + "=" * 50)
print("TEST SET RESULTS (held-out)")
print("=" * 50)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"AUROC:    {metrics['auroc']:.3f}")
print(f"ECE:      {metrics['ece']:.3f}")
print(f"Brier:    {metrics['brier_score']:.3f}")
print(f"Mean Confidence: {metrics['mean_confidence']:.3f}")

# Save test metrics
with open("uq_models/llama-8b-uq-lora-v2/test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved test metrics to uq_models/llama-8b-uq-lora-v2/test_metrics.json")
PYEOF

echo ""
echo "=========================================="
echo "FINETUNING V2 COMPLETE"
echo "Finished at: $(date)"
echo "Model saved to: uq_models/llama-8b-uq-lora-v2"
echo "=========================================="
