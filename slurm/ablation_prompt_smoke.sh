#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/ablation_prompt_smoke_%j.out
#SBATCH --job-name=abl_ps

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "Smoke test: CoT prompt ablation"
CUDA_VISIBLE_DEVICES=4 python scripts/run_prompt_ablations.py \
    --prompt_variant cot \
    --smoke_test \
    --output_dir data/ablations/prompt/smoke

echo "Exit code: $?"
