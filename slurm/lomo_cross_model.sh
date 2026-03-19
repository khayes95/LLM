#!/bin/bash
#SBATCH --job-name=lomo_cross_model
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/%x_%j.out

# Leave-One-Model-Out (LOMO) cross-model evaluation
# Trains 3 models sequentially, each holding out one source model:
#   1. Hold out gpt5mini  -> train on gpt52 + qwen35
#   2. Hold out gpt52     -> train on gpt5mini + qwen35
#   3. Hold out qwen35    -> train on gpt5mini + gpt52
#
# Uses v3 config: r=32, alpha=64, epochs=3, lr=1e-4, combined prompt
#
# Smoke test (uncomment to run quick validation):
# CUDA_VISIBLE_DEVICES=0 python scripts/train_best_uq.py \
#     --output_dir uq_models/lomo_smoke --held_out_model gpt5mini \
#     --lora_r 32 --lora_alpha 64 --epochs 1 --learning_rate 1e-4 \
#     --prompt_variant combined --smoke_test

set -euo pipefail

cd /scratch/khayes/LLM
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

COMMON_ARGS="--lora_r 32 --lora_alpha 64 --epochs 3 --learning_rate 1e-4 --prompt_variant combined --seed 42"

echo "============================================================"
echo "LOMO Cross-Model Evaluation"
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

# --- Run 1: Hold out GPT-5-mini ---
echo ""
echo "============================================================"
echo "RUN 1/3: Hold out gpt5mini (train on gpt52 + qwen35)"
echo "Start: $(date)"
echo "============================================================"

python scripts/train_best_uq.py \
    --output_dir uq_models/lomo_gpt5mini \
    --held_out_model gpt5mini \
    $COMMON_ARGS

echo "RUN 1/3 complete: $(date)"
echo ""

# --- Run 2: Hold out GPT-5.2 ---
echo "============================================================"
echo "RUN 2/3: Hold out gpt52 (train on gpt5mini + qwen35)"
echo "Start: $(date)"
echo "============================================================"

python scripts/train_best_uq.py \
    --output_dir uq_models/lomo_gpt52 \
    --held_out_model gpt52 \
    $COMMON_ARGS

echo "RUN 2/3 complete: $(date)"
echo ""

# --- Run 3: Hold out Qwen3.5 ---
echo "============================================================"
echo "RUN 3/3: Hold out qwen35 (train on gpt5mini + gpt52)"
echo "Start: $(date)"
echo "============================================================"

python scripts/train_best_uq.py \
    --output_dir uq_models/lomo_qwen35 \
    --held_out_model qwen35 \
    $COMMON_ARGS

echo "RUN 3/3 complete: $(date)"
echo ""

# --- Summary ---
echo "============================================================"
echo "LOMO EXPERIMENT COMPLETE"
echo "End time: $(date)"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  uq_models/lomo_gpt5mini/results.json"
echo "  uq_models/lomo_gpt52/results.json"
echo "  uq_models/lomo_qwen35/results.json"
echo ""

# Print summary of held-out model AUROCs
echo "--- LOMO Summary ---"
for model in gpt5mini gpt52 qwen35; do
    results_file="uq_models/lomo_${model}/results.json"
    if [ -f "$results_file" ]; then
        echo "Held-out $model:"
        python -c "
import json
with open('$results_file') as f:
    r = json.load(f)
print(f'  Overall AUROC: {r[\"auroc\"]:.4f}')
if 'per_source_model' in r and '$model' in r['per_source_model']:
    m = r['per_source_model']['$model']
    print(f'  Held-out model AUROC: {m.get(\"auroc\", \"N/A\")}')
    print(f'  Held-out model samples: {m[\"n_samples\"]}')
"
    else
        echo "  $results_file not found!"
    fi
done
