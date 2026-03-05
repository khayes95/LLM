#!/bin/bash
#SBATCH --job-name=se_base
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/semantic_entropy_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/semantic_entropy_%j.log

# Proxy Semantic Entropy + Cross-Model Self-Consistency baselines
# Uses GPU 0 (~80GB free)
#
# Smoke test:
#   CUDA_VISIBLE_DEVICES=0 python scripts/semantic_entropy_baseline.py \
#       --stage all --gpu 0 --smoke_test

set -e
cd /scratch/khayes/LLM

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "============================================"
echo "SEMANTIC ENTROPY BASELINE"
echo "Job: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "============================================"

# Step 1: Cross-model self-consistency (CPU, fast)
echo ""
echo ">>> Step 1: Self-Consistency ($(date))"
python scripts/semantic_entropy_baseline.py \
    --stage self_consistency \
    2>&1

echo "--- Self-consistency done ($(date)) ---"

# Step 2: Proxy semantic entropy (GPU, ~2h)
echo ""
echo ">>> Step 2: Proxy Semantic Entropy ($(date))"
python scripts/semantic_entropy_baseline.py \
    --stage proxy_se \
    --gpu 0 \
    --n_samples 5 \
    2>&1

echo ""
echo "============================================"
echo "SEMANTIC ENTROPY BASELINE COMPLETE — $(date)"
echo "============================================"
echo ""
echo "Outputs: data/ablations/semantic_entropy/"
