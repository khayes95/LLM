#!/bin/bash
#SBATCH --job-name=internvl3_vlm
#SBATCH --output=logs/internvl3_vlm_%j.out
#SBATCH --error=logs/internvl3_vlm_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# InternVL3-78B Evaluation on VLM Benchmarks
# Uses the SAME sampled IDs as GPT-5-mini runs for cross-model comparison
#
# Usage:
#   sbatch slurm/run_internvl3_vlm.sh

echo "=========================================="
echo "InternVL3-78B VLM Benchmark Evaluation"
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

# Create logs directory
mkdir -p logs

# Set CUDA devices - use all 4 GPUs for 78B model
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
nvidia-smi
echo ""

# VLM benchmarks with 15-85% accuracy range
# These match the GPT-5-mini runs
VLM_BENCHMARKS="mathvista mathverse mathvision mmmu mmstar charxiv realworldqa vizwiz erqa"

# Find sampled_ids from latest GPT-5-mini runs
# These ensure we evaluate the EXACT same questions for cross-model comparison
echo "Finding sampled IDs from GPT-5-mini runs..."

for bench in $VLM_BENCHMARKS; do
    # Find the latest GPT-5-mini run with sampled_ids
    latest_run=$(ls -td runs/*_${bench}_gpt-5-mini 2>/dev/null | head -1)

    if [[ -n "$latest_run" && -f "$latest_run/sampled_ids.json" ]]; then
        echo ""
        echo "=========================================="
        echo "Running InternVL3-78B on: $bench"
        echo "Using sampled IDs from: $latest_run"
        echo "=========================================="

        # Run evaluation with the same sampled IDs
        python -m uq_eval.cli \
            --bench "$bench" \
            --model_backend internvl \
            --model_name "OpenGVLab/InternVL3-78B-Instruct" \
            --timeout_s 600 \
            --max_output_tokens 16384 \
            --include_ids "$latest_run/sampled_ids.json"

        # Check result
        if [[ $? -eq 0 ]]; then
            echo "SUCCESS: $bench completed"
        else
            echo "ERROR: $bench failed"
        fi
    else
        echo "SKIP: $bench - no GPT-5-mini sampled_ids found"
    fi
done

echo ""
echo "=========================================="
echo "All benchmarks completed at: $(date)"
echo "=========================================="

# Print summary
echo ""
echo "Results Summary:"
for bench in $VLM_BENCHMARKS; do
    latest=$(ls -td runs/*_${bench}_InternVL3* 2>/dev/null | head -1)
    if [[ -n "$latest" && -f "$latest/metrics.json" ]]; then
        acc=$(python -c "import json; m=json.load(open('$latest/metrics.json')); print(f'{m.get(\"accuracy\", -1)*100:.1f}%')" 2>/dev/null || echo "N/A")
        echo "  $bench: $acc"
    else
        echo "  $bench: No results yet"
    fi
done
