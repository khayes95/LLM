#!/bin/bash
#SBATCH --job-name=uq_best_v2
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/retrain_best_v2_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/retrain_best_v2_%j.log

# Retrain with best config: r=32 + combined prompt (1500/800 truncation)
# Expected: ~3-4 hours on 1 GPU (GPU 7 free, others occupied by /cad jobs)

set -e
cd /scratch/khayes/LLM

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Retrain best UQ model v2 (r=32 + combined prompt)"
echo "Job: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

python scripts/run_prompt_ablations.py \
    --prompt_variant combined \
    --output_dir uq_models/best_v2 \
    --lora_r 32 \
    --epochs 3

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
