#!/bin/bash
#SBATCH --job-name=vlm_ablat
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/vlm_ablation_fixed_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/vlm_ablation_fixed_%j.log

# VLM training size ablation with FIXED VSR data
# Uses GPUs 1-4 (GPU 0 busy with SFT job)
# 6 sizes × ~30 min each = ~3 hours
#
# Smoke test: sbatch --export=QUICK=1 slurm/vlm_ablation_fixed.sh

set -e

echo "=========================================="
echo "VLM Training Size Ablation (Fixed Data)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Quick mode: ${QUICK:-0}"
echo "=========================================="

cd /scratch/khayes/LLM

eval "$(conda shell.bash hook)"
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=0,1,2,3

nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader

QUICK_FLAG=""
if [ "${QUICK:-0}" = "1" ]; then
    QUICK_FLAG="--quick"
fi

echo ""
echo "Running ablation (subprocess mode)..."
python scripts/vlm_training_size_ablation.py --run-all-subprocess --skip-transfer $QUICK_FLAG

echo ""
echo "=========================================="
echo "Results:"
cat data/ablations/vlm_training_size_vsr_fixed/ablation_results.json 2>/dev/null || echo "No results file found"
echo ""
echo "Done: $(date)"
