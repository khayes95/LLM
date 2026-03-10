#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=ts_smoke
#SBATCH --output=logs/ablation_trunc_scramble_smoke_%j.out

cd /scratch/khayes/LLM
source activate uq_eval

echo "Starting truncation + scramble SMOKE TEST: $(date)"

CUDA_VISIBLE_DEVICES=2 python scripts/ablation_truncation_scramble.py \
    --smoke_test \
    --checkpoint uq_models/best_v2_r32_combined \
    --output_dir data/ablations/truncation_scramble_smoke

echo "Finished: $(date)"
