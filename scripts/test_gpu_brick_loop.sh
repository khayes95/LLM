#!/bin/bash
#SBATCH --job-name=brick_loop
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/brick_loop_%j.out
#SBATCH --error=/scratch/khayes/LLM/logs/brick_loop_%j.err

# STRESS TEST: Repeatedly load model, infer, SIGKILL mid-kernel, check GPU health.
# Runs 50 iterations by default. Each iteration takes about 1 minute.
# Stops immediately if the GPU faults.
# WARNING: This is intentionally trying to break a GPU. Run during off-hours.

conda activate uq_eval
cd /scratch/khayes/LLM
source scripts/gpu_test_helpers.sh

echo "=== BRICK STRESS TEST: 50 iterations of load/infer/SIGKILL ==="
echo "Start: $(date)"
echo "SLURM assigned CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo ""

gpu_health_check || exit 1

python scripts/test_gpu_brick_loop.py --gpu "$CUDA_VISIBLE_DEVICES" --iterations 50 --kill_delay 30

echo "Finished: $(date)"
post_test_report
