#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/retrain_combined_prompt_%j.out
#SBATCH --job-name=retrain_cp

# Retrain best_unified with combined prompt template (longer + CoT + metadata).
# Prompt ablation showed 0.831 → 0.867 AUROC (+3.6 pts).
# This produces best_unified_v2 with the combined template.
#
# Queued to run after elicitation ablation jobs finish (dependency).
# Uses GPUs 0-3.

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "============================================"
echo "Retrain with Combined Prompt (best_unified_v2)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "============================================"

# Smoke test first
echo "--- SMOKE TEST ---"
CUDA_VISIBLE_DEVICES=0 python scripts/run_prompt_ablations.py \
    --prompt_variant combined \
    --output_dir uq_models/best_unified_v2_smoke \
    --smoke_test

echo "Smoke test passed. Starting full training..."

# Full training
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_prompt_ablations.py \
    --prompt_variant combined \
    --output_dir uq_models/best_unified_v2 \
    --epochs 3 --batch_size 1 --grad_accum 16 --learning_rate 1e-4

echo "Done: $(date)"
