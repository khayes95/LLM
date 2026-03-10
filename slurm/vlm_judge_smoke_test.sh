#!/bin/bash
#SBATCH --job-name=vlm_smoke
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/khayes/LLM/logs/vlm_smoke_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/vlm_smoke_%j.log

# Smoke test: VLM judge with real images on GPT-5.2 (5 samples per benchmark)

set -e

cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "=========================================="
echo "VLM Judge Smoke Test (real images)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

CUDA_VISIBLE_DEVICES=0 python scripts/vlm_judge_cross_model_eval.py \
    --target gpt52 \
    --smoke_test \
    --output data/cross_model/vlm_judge_images_smoke.json

echo ""
echo "Done: $(date)"
echo "=========================================="
