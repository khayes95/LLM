#!/bin/bash
#SBATCH --job-name=q35_ray
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_qwen3_5_ray_%j.log
#SBATCH --error=logs/test_qwen3_5_ray_%j.log

echo "=========================================="
echo "Qwen3.5-397B-A17B-FP8 Ray Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

source ~/.bashrc
conda activate uq_eval

python scripts/test_qwen3_5_ray.py 2>&1

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
