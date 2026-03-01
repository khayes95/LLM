#!/bin/bash
#SBATCH --job-name=txtcal_xm
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/overnight_text_cal_xmodel_%j.log
#SBATCH --error=logs/overnight_text_cal_xmodel_%j.log

# Follow-up: Re-run text calibrator v3 cross-model eval on Qwen3-VL-30B
# with updated data (includes newly completed benchmarks from overnight)
# GPU 0 (freed after Qwen3-VL batch A finishes)

echo "=========================================="
echo "Text Calibrator v3 Cross-Model (Updated)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Base env has transformers/peft needed for text calibrator
python scripts/cross_model_eval_v3.py \
    --output data/cross_model/text_v3_on_qwen3vl_updated.json

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
