#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/ablation_prompt_b_%j.out
#SBATCH --job-name=abl_pB

# Prompt ablation batch B: metadata + combined (CoT+metadata+longer)
# Sequential on 1 GPU, ~2h each = ~4h total

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "============================================"
echo "Prompt Ablation Batch B: Metadata + Combined"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "============================================"

# --- Metadata-enriched ---
echo ""
echo ">>> Metadata prompt — $(date)"
CUDA_VISIBLE_DEVICES=7 python scripts/run_prompt_ablations.py \
    --prompt_variant metadata \
    --output_dir data/ablations/prompt/metadata \
    --epochs 3 --learning_rate 1e-4 --lora_r 16
echo "<<< Metadata done — $(date)"

# --- Combined (CoT + metadata + longer) ---
echo ""
echo ">>> Combined (kitchen sink) — $(date)"
CUDA_VISIBLE_DEVICES=7 python scripts/run_prompt_ablations.py \
    --prompt_variant combined \
    --output_dir data/ablations/prompt/combined \
    --epochs 3 --learning_rate 1e-4 --lora_r 16
echo "<<< Combined done — $(date)"

echo ""
echo "============================================"
echo "Batch B complete: $(date)"
echo "============================================"
