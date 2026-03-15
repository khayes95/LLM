#!/bin/bash
#SBATCH --job-name=brick_midkrnl
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/khayes/LLM/logs/brick_midkernel_%j.out
#SBATCH --error=/scratch/khayes/LLM/logs/brick_midkernel_%j.err

# BRICK ATTEMPT: SIGKILL self mid-CUDA-kernel
# The script loads vLLM, runs large inference batches, then SIGKILLs itself
# after 30 seconds while a CUDA kernel is actively running.
# WARNING: This is intentionally trying to fault the GPU.

conda activate uq_eval
cd /scratch/khayes/LLM
source scripts/gpu_test_helpers.sh

echo "=== BRICK REPRO: SIGKILL mid-CUDA-kernel ==="
echo "Start: $(date)"
echo "SLURM assigned CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo ""

gpu_health_check || exit 1

python scripts/test_gpu_brick_repro.py

echo "Finished: $(date)"
post_test_report
