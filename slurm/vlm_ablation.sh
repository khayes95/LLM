#!/bin/bash
#SBATCH --job-name=vlm_ablation
#SBATCH --output=logs/vlm_ablation_%j.out
#SBATCH --error=logs/vlm_ablation_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# VLM Training Data Size Ablation Experiment
# Tests minimum number of samples needed for good performance

echo "=========================================="
echo "VLM Training Size Ablation Experiment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /scratch/khayes/anaconda3/etc/profile.d/conda.sh
conda activate base

# Change to project directory
cd /scratch/khayes/LLM

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p data/ablations/vlm_training_size

# Set CUDA devices - use all 4 GPUs on the node
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
echo ""

# Run the ablation experiment
# Using --skip-transfer since cross-model needs Qwen2.5-VL-72B which is very large
# We can run cross-model separately after if needed
python scripts/vlm_training_size_ablation.py --skip-transfer

echo ""
echo "=========================================="
echo "Experiment completed at: $(date)"
echo "=========================================="
