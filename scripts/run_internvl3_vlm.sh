#!/bin/bash
# Run InternVL3-78B on VLM benchmarks (without SLURM)
# Uses the SAME sampled IDs as GPT-5-mini runs for cross-model comparison
#
# Usage:
#   ./scripts/run_internvl3_vlm.sh [benchmark]
#
# If no benchmark specified, runs all VLM benchmarks sequentially.
# InternVL3-78B requires ~160GB VRAM (4x A100 80GB)
#
# Estimated time per benchmark:
#   - 100 samples: ~1-2 hours (depending on image complexity)
#   - Total for all 9 benchmarks: ~12-18 hours

set -e

cd /scratch/khayes/LLM

# Use all available GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=========================================="
echo "InternVL3-78B VLM Benchmark Evaluation"
echo "=========================================="
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# VLM benchmarks with 15-85% accuracy range
VLM_BENCHMARKS="mathvista mathverse mathvision mmmu mmstar charxiv realworldqa vizwiz erqa"

# If a specific benchmark is provided, only run that one
if [[ -n "$1" ]]; then
    VLM_BENCHMARKS="$1"
fi

echo "Benchmarks to run: $VLM_BENCHMARKS"
echo ""

for bench in $VLM_BENCHMARKS; do
    # Find the latest GPT-5-mini run with sampled_ids
    latest_run=$(ls -td runs/*_${bench}_gpt-5-mini 2>/dev/null | head -1)

    if [[ -n "$latest_run" && -f "$latest_run/sampled_ids.json" ]]; then
        num_samples=$(python3 -c "import json; print(len(json.load(open('$latest_run/sampled_ids.json'))))")

        echo "=========================================="
        echo "Running InternVL3-78B on: $bench"
        echo "Using sampled IDs from: $latest_run"
        echo "Number of samples: $num_samples"
        echo "Start: $(date)"
        echo "=========================================="

        # Run evaluation with the same sampled IDs
        python -m uq_eval.cli \
            --bench "$bench" \
            --model_backend internvl \
            --model_name "OpenGVLab/InternVL3-78B-Instruct" \
            --timeout_s 600 \
            --max_output_tokens 16384 \
            --include_ids "$latest_run/sampled_ids.json"

        echo "Completed: $bench at $(date)"
        echo ""
    else
        echo "SKIP: $bench - no GPT-5-mini sampled_ids found at runs/*_${bench}_gpt-5-mini"
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
    latest=$(ls -td runs/*_${bench}_*InternVL* 2>/dev/null | head -1)
    if [[ -n "$latest" && -f "$latest/metrics.json" ]]; then
        acc=$(python3 -c "import json; m=json.load(open('$latest/metrics.json')); print(f'{m.get(\"accuracy\", -1)*100:.1f}%')" 2>/dev/null || echo "N/A")
        n=$(python3 -c "import json; m=json.load(open('$latest/metrics.json')); print(m.get('n', 'N/A'))" 2>/dev/null || echo "N/A")
        echo "  $bench: $acc ($n samples)"
    else
        echo "  $bench: No results yet"
    fi
done
