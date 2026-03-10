#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/ablation_batch_b_%j.out
#SBATCH --job-name=abl_B

# Batch B: Source model + modality ablations — sequential on 1 GPU
# Source: train on 1 model, evaluate cross-model transfer
# Modality: train on text-only or vlm-only, test cross-modality transfer
# Estimated: ~5 hours total

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "============================================"
echo "Batch B: Source Model + Modality Ablations"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "============================================"

# --- Source: GPT-5-mini only ---
echo ""
echo ">>> Source: gpt5mini-only — $(date)"
CUDA_VISIBLE_DEVICES=7 python scripts/run_ablations.py \
    --ablation source_model --source_models gpt5mini \
    --output_dir data/ablations/source_model/gpt5mini_only \
    --epochs 3 --learning_rate 1e-4
echo "<<< gpt5mini-only done — $(date)"

# --- Source: GPT-5.2 only ---
echo ""
echo ">>> Source: gpt52-only — $(date)"
CUDA_VISIBLE_DEVICES=7 python scripts/run_ablations.py \
    --ablation source_model --source_models gpt52 \
    --output_dir data/ablations/source_model/gpt52_only \
    --epochs 3 --learning_rate 1e-4
echo "<<< gpt52-only done — $(date)"

# --- Source: Qwen3.5 only ---
echo ""
echo ">>> Source: qwen35-only — $(date)"
CUDA_VISIBLE_DEVICES=7 python scripts/run_ablations.py \
    --ablation source_model --source_models qwen35 \
    --output_dir data/ablations/source_model/qwen35_only \
    --epochs 3 --learning_rate 1e-4
echo "<<< qwen35-only done — $(date)"

# --- Modality: text-only ---
echo ""
echo ">>> Modality: text-only — $(date)"
CUDA_VISIBLE_DEVICES=7 python scripts/run_ablations.py \
    --ablation modality --modality text_only \
    --output_dir data/ablations/modality/text_only \
    --epochs 3 --learning_rate 1e-4
echo "<<< text-only done — $(date)"

# --- Modality: vlm-only ---
echo ""
echo ">>> Modality: vlm-only — $(date)"
CUDA_VISIBLE_DEVICES=7 python scripts/run_ablations.py \
    --ablation modality --modality vlm_only \
    --output_dir data/ablations/modality/vlm_only \
    --epochs 3 --learning_rate 1e-4
echo "<<< vlm-only done — $(date)"

echo ""
echo "============================================"
echo "Batch B complete: $(date)"
echo "============================================"
