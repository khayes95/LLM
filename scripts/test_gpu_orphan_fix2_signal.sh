#!/bin/bash
#SBATCH --job-name=fix2_signal
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:02:00
#SBATCH --signal=B:USR1@30
#SBATCH --output=/scratch/khayes/LLM/logs/fix2_signal_%j.out
#SBATCH --error=/scratch/khayes/LLM/logs/fix2_signal_%j.err

# FIX 2: SLURM --signal + shell trap (still bare python, no srun)
# SLURM sends USR1 30s before timeout, shell trap kills the process group

conda activate uq_eval
cd /scratch/khayes/LLM
source scripts/gpu_test_helpers.sh

echo "=== FIX 2: shell signal trap only (bare python, no srun) ==="
echo "Start: $(date)"
echo "SLURM assigned CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo ""

gpu_health_check || exit 1

cleanup() {
    echo "Shell trap: caught signal, killing process group..."
    kill -TERM -$$ 2>/dev/null || true
    wait
    echo "Shell cleanup done: $(date)"
    post_test_report
}
trap cleanup USR1 TERM EXIT

python scripts/test_gpu_orphan_repro.py &
wait $!

echo "Finished: $(date)"
post_test_report
