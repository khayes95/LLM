#!/bin/bash
#SBATCH --job-name=molmo_smoke
#SBATCH --output=logs/smoke_test_%j.log
#SBATCH --error=logs/smoke_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=320G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

# Phase 1 Smoke Test for Molmo-72B

echo "================================================"
echo "MOLMO-72B SMOKE TEST"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "================================================"

# Setup
cd /scratch/khayes/LLM
mkdir -p logs

# Activate environment (adjust as needed)
source ~/.bashrc
conda activate llm  # or your environment name

# Check GPU availability
nvidia-smi

# Run smoke test
echo ""
echo "Running smoke test..."
python scripts/smoke_test.py --model_size 72B --num_samples 10

echo ""
echo "================================================"
echo "End: $(date)"
echo "================================================"
