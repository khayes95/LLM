#!/bin/bash
#SBATCH --job-name=multi_seed2
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/multi_seed_remaining_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/multi_seed_remaining_%j.log

# Complete multi-seed training: seeds 123 and 456 (seed 42 already done)
# Each seed takes ~9 hours (526 min) for 3 epochs
# Total: ~18 hours

set -e
cd /scratch/khayes/LLM

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "============================================"
echo "MULTI-SEED REMAINING (seeds 123, 456)"
echo "Job: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "============================================"

python scripts/multi_seed_training.py \
    --output_dir data/ablations/multi_seed \
    --seeds 123 456 \
    --epochs 3 \
    --lora_r 32

echo ""
echo "============================================"
echo "MULTI-SEED COMPLETE — $(date)"
echo "============================================"
