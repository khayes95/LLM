#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/ablation_smoke_%j.out
#SBATCH --job-name=abl_smoke

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "Smoke test: LoRA r=4 ablation"
CUDA_VISIBLE_DEVICES=0 python scripts/run_ablations.py \
    --ablation lora_rank \
    --lora_r 4 \
    --smoke_test \
    --output_dir data/ablations/smoke_test

echo "Exit code: $?"
