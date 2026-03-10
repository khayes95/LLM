#!/bin/bash
#SBATCH --job-name=cross_model_v3
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/cross_model_eval_v3_%j.log
#SBATCH --error=logs/cross_model_eval_v3_%j.log

# Smoke test (uncomment to verify):
# CUDA_VISIBLE_DEVICES=0 python scripts/cross_model_eval_v3.py --smoke_test

echo "=========================================="
echo "Cross-Model Eval: text_calibrator_v3 on Qwen3-VL-30B"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

# Use single GPU
export CUDA_VISIBLE_DEVICES=0

# Activate environment
source /scratch/khayes/anaconda3/etc/profile.d/conda.sh
conda activate base

python scripts/cross_model_eval_v3.py \
    --calibrator uq_models/text_calibrator_v3 \
    --output data/cross_model/text_v3_on_qwen3vl.json

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
