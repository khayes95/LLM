#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/ablation_modality_%j.out
#SBATCH --job-name=abl_mod

# Modality ablation: train on text-only or vlm-only data

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

MODALITY=${1:-text_only}
echo "============================================"
echo "Modality Ablation: ${MODALITY}"
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python scripts/run_ablations.py \
    --ablation modality \
    --modality ${MODALITY} \
    --output_dir data/ablations/modality/${MODALITY} \
    --epochs 3 --learning_rate 1e-4

echo "Done: $(date)"
