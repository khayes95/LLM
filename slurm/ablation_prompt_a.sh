#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/ablation_prompt_a_%j.out
#SBATCH --job-name=abl_pA

# Prompt ablation batch A: cot + longer context
# Sequential on 1 GPU, ~2h each = ~4h total

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "============================================"
echo "Prompt Ablation Batch A: CoT + Longer"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "============================================"

# --- Chain-of-thought ---
echo ""
echo ">>> CoT prompt — $(date)"
CUDA_VISIBLE_DEVICES=4 python scripts/run_prompt_ablations.py \
    --prompt_variant cot \
    --output_dir data/ablations/prompt/cot \
    --epochs 3 --learning_rate 1e-4 --lora_r 16
echo "<<< CoT done — $(date)"

# --- Longer context ---
echo ""
echo ">>> Longer context (1500/800) — $(date)"
CUDA_VISIBLE_DEVICES=4 python scripts/run_prompt_ablations.py \
    --prompt_variant longer \
    --output_dir data/ablations/prompt/longer \
    --epochs 3 --learning_rate 1e-4 --lora_r 16
echo "<<< Longer done — $(date)"

echo ""
echo "============================================"
echo "Batch A complete: $(date)"
echo "============================================"
