#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/ablation_batch_a_%j.out
#SBATCH --job-name=abl_A

# Batch A: LoRA rank ablations (r=4, r=8, r=32) — sequential on 1 GPU
# Baseline r=16 is the best_unified model (AUROC=0.831)
# Estimated: ~6 hours total

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "============================================"
echo "Batch A: LoRA Rank Ablations (r=4, r=8, r=32)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "============================================"

# --- r=4 ---
echo ""
echo ">>> LoRA r=4 — $(date)"
CUDA_VISIBLE_DEVICES=4 python scripts/run_ablations.py \
    --ablation lora_rank --lora_r 4 \
    --output_dir data/ablations/lora_rank/r4 \
    --epochs 3 --learning_rate 1e-4
echo "<<< LoRA r=4 done — $(date)"

# --- r=8 ---
echo ""
echo ">>> LoRA r=8 — $(date)"
CUDA_VISIBLE_DEVICES=4 python scripts/run_ablations.py \
    --ablation lora_rank --lora_r 8 \
    --output_dir data/ablations/lora_rank/r8 \
    --epochs 3 --learning_rate 1e-4
echo "<<< LoRA r=8 done — $(date)"

# --- r=32 ---
echo ""
echo ">>> LoRA r=32 — $(date)"
CUDA_VISIBLE_DEVICES=4 python scripts/run_ablations.py \
    --ablation lora_rank --lora_r 32 \
    --output_dir data/ablations/lora_rank/r32 \
    --epochs 3 --learning_rate 1e-4
echo "<<< LoRA r=32 done — $(date)"

echo ""
echo "============================================"
echo "Batch A complete: $(date)"
echo "============================================"
