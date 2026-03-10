#!/bin/bash
#SBATCH --job-name=vlm_gpt5m
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/overnight_vlm_judge_gpt5mini_%j.log
#SBATCH --error=logs/overnight_vlm_judge_gpt5mini_%j.log

# Cross-model eval: VLM judge (VSR-fixed) on GPT-5-mini responses
# Tests closed-source transfer (train on InternVL3-78B → eval on GPT-5-mini)
# GPU 2

echo "=========================================="
echo "VLM Judge Cross-Model: GPT-5-mini"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /scratch/khayes/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

# Smoke test first
echo "--- SMOKE TEST ---"
python scripts/vlm_judge_cross_model_eval.py \
    --target gpt5mini \
    --smoke_test \
    --output data/cross_model/vlm_judge_vsr_fixed_on_gpt5mini_smoke.json

if [ $? -ne 0 ]; then
    echo "SMOKE TEST FAILED - aborting full run"
    exit 1
fi
echo "--- SMOKE TEST PASSED ---"
echo ""

# Full run
python scripts/vlm_judge_cross_model_eval.py \
    --target gpt5mini \
    --output data/cross_model/vlm_judge_vsr_fixed_on_gpt5mini.json

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
