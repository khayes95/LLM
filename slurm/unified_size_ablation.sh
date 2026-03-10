#!/bin/bash
#SBATCH --job-name=uq_sizeabl
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/unified_size_ablation_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/unified_size_ablation_%j.log

# Unified VLM size ablation: Qwen3-VL-2B vs 4B vs 8B
# Each model trains sequentially (~30min for 2B, ~60min for 4B, ~120min for 8B)
# Uses 4 GPUs, images already cached

# Smoke test (uncomment to test):
# CUDA_VISIBLE_DEVICES=0 python scripts/unified_size_ablation.py --smoke_test --models 2b

set -e
cd /scratch/khayes/LLM

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=0,1,3,4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Unified VLM Size Ablation"
echo "Job: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader

python scripts/unified_size_ablation.py \
    --models 2b,4b,8b \
    --epochs 3

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
