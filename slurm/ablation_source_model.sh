#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/ablation_source_%j.out
#SBATCH --job-name=abl_src

# Source model ablation: train on single source model, eval on all

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

SOURCE=${1:-gpt5mini}
echo "============================================"
echo "Source Model Ablation: ${SOURCE}"
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python scripts/run_ablations.py \
    --ablation source_model \
    --source_models ${SOURCE} \
    --output_dir data/ablations/source_model/${SOURCE}_only \
    --epochs 3 --learning_rate 1e-4

echo "Done: $(date)"
