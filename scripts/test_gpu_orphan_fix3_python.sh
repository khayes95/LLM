#!/bin/bash
#SBATCH --job-name=fix3_python
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:02:00
#SBATCH --output=/scratch/khayes/LLM/logs/fix3_python_%j.out
#SBATCH --error=/scratch/khayes/LLM/logs/fix3_python_%j.err

# FIX 3: Python-level signal handlers only (bare python, no srun, no shell trap)
# The Python script itself catches SIGTERM and kills vLLM + its process group

conda activate uq_eval
cd /scratch/khayes/LLM
source scripts/gpu_test_helpers.sh

echo "=== FIX 3: Python signal handlers only (bare python, no srun, no shell trap) ==="
echo "Start: $(date)"
echo "SLURM assigned CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo ""

gpu_health_check || exit 1

python scripts/test_gpu_orphan_fixed.py

echo "Finished: $(date)"
post_test_report
