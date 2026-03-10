#!/bin/bash
#SBATCH --job-name=elicit_v2
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/elicitation_v2_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/elicitation_v2_%j.log

# Elicitation ablations v2: temperature scaling + token entropy + hidden state probing
# Uses v2 checkpoint on test-only scored data. GPU 2 (~60 GB free).
#
# Smoke test:
#   CUDA_VISIBLE_DEVICES=2 python scripts/elicitation_ablations_v2.py \
#       --strategy gpu_all --smoke_test

set -e
cd /scratch/khayes/LLM

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "============================================"
echo "ELICITATION ABLATIONS V2"
echo "Job: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "============================================"

# Step 1: Temperature scaling (CPU only, fast)
echo ""
echo ">>> Step 1: Temperature Scaling ($(date))"
python scripts/elicitation_ablations_v2.py \
    --strategy temperature_scaling \
    2>&1

echo "--- Temperature scaling done ($(date)) ---"

# Step 2: Token entropy + hidden state probing (GPU, single model load)
echo ""
echo ">>> Step 2: GPU strategies — token entropy + hidden state probing ($(date))"
python scripts/elicitation_ablations_v2.py \
    --strategy gpu_all \
    2>&1

echo ""
echo "============================================"
echo "ELICITATION ABLATIONS V2 COMPLETE — $(date)"
echo "============================================"
echo ""
echo "Outputs: data/ablations/elicitation_v2/"
