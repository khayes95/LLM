#!/bin/bash
#SBATCH --job-name=brick_tdwn
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/khayes/LLM/logs/brick_teardown_%j.out
#SBATCH --error=/scratch/khayes/LLM/logs/brick_teardown_%j.err

# BRICK ATTEMPT: SIGKILL during CUDA context teardown
# The script runs inference, then on SIGTERM starts tearing down vLLM
# and immediately SIGKILLs itself mid-cleanup.
# Send "kill -TERM <pid>" manually once inference is running.
# WARNING: This is intentionally trying to fault the GPU.

conda activate uq_eval
cd /scratch/khayes/LLM
source scripts/gpu_test_helpers.sh

echo "=== BRICK REPRO: SIGKILL mid-CUDA-teardown ==="
echo "Start: $(date)"
echo "SLURM assigned CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo ""
echo "Once inference is running, send: kill -TERM <python_pid>"
echo ""

gpu_health_check || exit 1

python scripts/test_gpu_brick_teardown.py

echo "Finished: $(date)"
post_test_report
