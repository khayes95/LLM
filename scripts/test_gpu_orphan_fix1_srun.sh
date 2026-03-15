#!/bin/bash
#SBATCH --job-name=fix1_srun
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:02:00
#SBATCH --output=/scratch/khayes/LLM/logs/fix1_srun_%j.out
#SBATCH --error=/scratch/khayes/LLM/logs/fix1_srun_%j.err

# FIX 1: Only change is srun instead of bare python
# srun makes SLURM track the entire process tree and kill all children on timeout

conda activate uq_eval
cd /scratch/khayes/LLM
source scripts/gpu_test_helpers.sh

echo "=== FIX 1: srun only (no signal handling) ==="
echo "Start: $(date)"
echo "SLURM assigned CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo ""

gpu_health_check || exit 1

srun python scripts/test_gpu_orphan_repro.py

echo "Finished: $(date)"
post_test_report
