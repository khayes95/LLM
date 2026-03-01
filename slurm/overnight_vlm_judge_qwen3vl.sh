#!/bin/bash
#SBATCH --job-name=vlm_q3vl
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/overnight_vlm_judge_qwen3vl_%j.log
#SBATCH --error=logs/overnight_vlm_judge_qwen3vl_%j.log

# Cross-model eval: VLM judge (VSR-fixed) on Qwen3-VL-30B responses
# Tests open-source cross-model (train on InternVL3-78B → eval on Qwen3-VL-30B)
# GPU 7

echo "=========================================="
echo "VLM Judge Cross-Model: Qwen3-VL-30B"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /scratch/khayes/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

# Smoke test first
echo "--- SMOKE TEST ---"
python scripts/vlm_judge_cross_model_eval.py \
    --target qwen3vl \
    --smoke_test \
    --output data/cross_model/vlm_judge_vsr_fixed_on_qwen3vl_smoke.json

if [ $? -ne 0 ]; then
    echo "SMOKE TEST FAILED - aborting full run"
    exit 1
fi
echo "--- SMOKE TEST PASSED ---"
echo ""

# Full run
python scripts/vlm_judge_cross_model_eval.py \
    --target qwen3vl \
    --output data/cross_model/vlm_judge_vsr_fixed_on_qwen3vl.json

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
