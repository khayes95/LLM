#!/bin/bash
#SBATCH --job-name=q3vl_a
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/overnight_qwen3vl_a_%j.log
#SBATCH --error=logs/overnight_qwen3vl_a_%j.log

# Qwen3-VL-30B missing benchmarks batch A: omnimath, bbeh, charxiv
# GPU 0 - model loads once, runs 3 benchmarks sequentially

echo "=========================================="
echo "Qwen3-VL-30B Missing Benchmarks (Batch A)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/scratch/khayes/anaconda3/envs/finegrain_vlm/bin/python

$PYTHON scripts/run_qwen3_vl_30b_full.py \
    --benchmarks omnimath bbeh charxiv \
    --max_samples 1000 \
    --out_dir runs/qwen3vl_30b_overnight

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
