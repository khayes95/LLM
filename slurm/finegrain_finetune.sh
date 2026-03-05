#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --job-name=fg_ft
#SBATCH --output=logs/fg_finetune_%j.out

# Smoke test (uncomment one):
# CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_finetune.py --experiment human_cv --output_dir data/finegrain_uq/smoke_ft --smoke_test
# CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_finetune.py --experiment judge_only --output_dir data/finegrain_uq/smoke_ft_judge --smoke_test

set -euo pipefail
cd /scratch/khayes/LLM
export PATH=/scratch/khayes/anaconda3/bin:$PATH
mkdir -p logs data/finegrain_uq

echo "=========================================="
echo "FineGRAIN UQ: Fine-Tuning Experiments"
echo "Started: $(date)"
echo "GPUs: $(nvidia-smi --list-gpus | wc -l)"
echo "=========================================="

# Experiment 1: 5-fold leave-one-model-out CV on human data
echo ""
echo ">>> Experiment 1: Human-Only CV ($(date))"
CUDA_VISIBLE_DEVICES=0,1 python scripts/finegrain_finetune.py \
    --experiment human_cv \
    --output_dir data/finegrain_uq/exp1_human_cv \
    --held_out_model all
echo ">>> Experiment 1 done ($(date))"

# Experiment 2: Judge-only training (with flux excluded)
echo ""
echo ">>> Experiment 2: Judge-Only ($(date))"
CUDA_VISIBLE_DEVICES=0,1 python scripts/finegrain_finetune.py \
    --experiment judge_only \
    --output_dir data/finegrain_uq/exp2_judge \
    --exclude_flux
echo ">>> Experiment 2 done ($(date))"

# Experiment 3: Combined (human + judge + QA replay)
echo ""
echo ">>> Experiment 3: Combined ($(date))"
CUDA_VISIBLE_DEVICES=0,1 python scripts/finegrain_finetune.py \
    --experiment combined \
    --output_dir data/finegrain_uq/exp3_combined \
    --held_out_model sd3_xl \
    --exclude_flux
echo ">>> Experiment 3 done ($(date))"

echo ""
echo "=========================================="
echo "All experiments complete: $(date)"
echo "=========================================="
