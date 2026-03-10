#!/bin/bash
#SBATCH --job-name=q35_smoke
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --nodelist=gpunode00
#SBATCH --output=logs/smoke_test_qwen3_5_%j.log
#SBATCH --error=logs/smoke_test_qwen3_5_%j.log

echo "=========================================="
echo "Qwen3.5-397B Smoke Test (API client)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

source ~/.bashrc
conda activate uq_eval

python scripts/smoke_test_qwen3_5.py 2>&1

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
