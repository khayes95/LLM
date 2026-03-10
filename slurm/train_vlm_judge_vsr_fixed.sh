#!/bin/bash
#SBATCH --job-name=vlm_vsr_fix
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/vlm_judge_vsr_fixed_%j.log
#SBATCH --error=logs/vlm_judge_vsr_fixed_%j.log

# Smoke test (uncomment to verify):
# CUDA_VISIBLE_DEVICES=0,1 python scripts/train_vlm_judge_vsr_fixed.py

echo "=========================================="
echo "VLM Judge Retraining - VSR Image Fix"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

# Use 5 free GPUs (GPUs 2,4,5 in use by others)
export CUDA_VISIBLE_DEVICES=0,1,3,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate environment (base has all required packages)
source /scratch/khayes/anaconda3/etc/profile.d/conda.sh
conda activate base

python scripts/train_vlm_judge_vsr_fixed.py

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
