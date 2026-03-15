#!/bin/bash
#SBATCH --job-name=orphan_fixed
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:02:00
#SBATCH --signal=B:USR1@30
#SBATCH --output=/scratch/khayes/LLM/logs/orphan_fixed_%j.out
#SBATCH --error=/scratch/khayes/LLM/logs/orphan_fixed_%j.err

# FIXED: srun + signal trap + Python signal handlers (all three combined)
# Expect: clean shutdown, no orphaned processes, GPU memory freed

conda activate uq_eval
cd /scratch/khayes/LLM
source scripts/gpu_test_helpers.sh

echo "=== FIXED: srun + signal handling, 2-min time limit ==="
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

srun python scripts/test_gpu_orphan_fixed.py &
wait $!

echo "Finished: $(date)"
post_test_report
