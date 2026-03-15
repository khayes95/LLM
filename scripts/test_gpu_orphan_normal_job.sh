#!/bin/bash
#SBATCH --job-name=normal_job
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --signal=B:USR1@30
#SBATCH --output=/scratch/khayes/LLM/logs/normal_job_%j.out
#SBATCH --error=/scratch/khayes/LLM/logs/normal_job_%j.err

# NORMAL JOB: Verify that the fixes don't break a job that completes normally.
# Loads vLLM, runs 5 batches, exits cleanly. Should finish in ~1-2 minutes.

conda activate uq_eval
cd /scratch/khayes/LLM
source scripts/gpu_test_helpers.sh

echo "=== NORMAL JOB: should complete cleanly ==="
echo "Start: $(date)"
echo "SLURM assigned CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo ""

gpu_health_check || exit 1

cleanup() {
    echo "Shell trap: caught signal, forwarding to process group..."
    kill -TERM -$$ 2>/dev/null || true
    wait
    echo "Shell cleanup done: $(date)"
    post_test_report
}
trap cleanup USR1 TERM EXIT

srun python scripts/test_gpu_orphan_normal_job.py &
wait $!

echo "Finished cleanly: $(date)"
post_test_report
