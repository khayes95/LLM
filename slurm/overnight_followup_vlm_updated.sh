#!/bin/bash
#SBATCH --job-name=vlm_q3up
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/overnight_vlm_judge_qwen3vl_updated_%j.log
#SBATCH --error=logs/overnight_vlm_judge_qwen3vl_updated_%j.log

# Follow-up: Re-run VLM judge cross-model on Qwen3-VL-30B
# with newly completed benchmarks (omnimath, bbeh, livebench, simpleqa, charxiv)
# GPU 1 (freed after Qwen3-VL batch B finishes)

echo "=========================================="
echo "VLM Judge Cross-Model: Qwen3-VL-30B (Updated)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Base env has transformers/peft needed for VLM judge
python scripts/vlm_judge_cross_model_eval.py \
    --target qwen3vl \
    --output data/cross_model/vlm_judge_vsr_fixed_on_qwen3vl_updated.json

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
