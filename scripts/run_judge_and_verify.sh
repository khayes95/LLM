#!/bin/bash
# Run LLM judge grading and verify accuracy after Qwen3-VL inference completes
#
# This script:
# 1. Waits for all inference jobs to complete (checks every 2 min)
# 2. Runs Qwen3-VL judge on prbench and mmvet predictions
# 3. Checks accuracy and updates research log

set -e

LOG_DIR="/scratch/khayes/LLM/logs"
RUNS_DIR="/scratch/khayes/LLM/runs"
RESEARCH_LOG="/scratch/khayes/LLM/RESEARCH_LOG.md"

echo "$(date): Starting judge and verify script"
echo "Waiting for inference jobs to complete..."

# Wait for inference jobs to complete
while true; do
    # Check if any run_qwen3 or run_charxiv processes are running
    running=$(ps aux | grep -E "run_qwen3_vl_30b_full|run_charxiv_fixed" | grep -v grep | wc -l)

    if [ "$running" -eq 0 ]; then
        echo "$(date): All inference jobs completed!"
        break
    fi

    echo "$(date): $running jobs still running, checking again in 2 minutes..."
    sleep 120
done

# Find the latest run directory (created today)
LATEST_RUN=$(ls -dt ${RUNS_DIR}/*Qwen3_VL_30B* 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "ERROR: No Qwen3-VL run directory found"
    exit 1
fi

echo "$(date): Using run directory: $LATEST_RUN"

# Run grading on prbench and mmvet
echo ""
echo "$(date): Running Qwen3-VL judge on LLM-judged benchmarks..."

# Find prbench predictions
PRBENCH_PRED="${LATEST_RUN}/prbench/predictions.jsonl"
if [ -f "$PRBENCH_PRED" ]; then
    echo "Grading prbench..."
    CUDA_VISIBLE_DEVICES=0 /scratch/khayes/anaconda3/envs/finegrain_vlm/bin/python \
        /scratch/khayes/LLM/scripts/grade_with_qwen3_vl.py \
        --predictions "$PRBENCH_PRED" \
        2>&1 | tee ${LOG_DIR}/grade_prbench.log
else
    echo "Warning: prbench predictions not found at $PRBENCH_PRED"
fi

# Find mmvet predictions
MMVET_PRED="${LATEST_RUN}/mmvet/predictions.jsonl"
if [ -f "$MMVET_PRED" ]; then
    echo "Grading mmvet..."
    CUDA_VISIBLE_DEVICES=0 /scratch/khayes/anaconda3/envs/finegrain_vlm/bin/python \
        /scratch/khayes/LLM/scripts/grade_with_qwen3_vl.py \
        --predictions "$MMVET_PRED" \
        2>&1 | tee ${LOG_DIR}/grade_mmvet.log
else
    echo "Warning: mmvet predictions not found at $MMVET_PRED"
fi

# Collect all metrics and verify accuracy
echo ""
echo "$(date): Collecting and verifying metrics..."

# Create summary
SUMMARY=""
ISSUES=""

for bench_dir in ${LATEST_RUN}/*/; do
    bench=$(basename "$bench_dir")
    metrics_file="${bench_dir}metrics.json"

    if [ -f "$metrics_file" ]; then
        n=$(jq -r '.n // .n_scored // 0' "$metrics_file" 2>/dev/null || echo "0")
        acc=$(jq -r '.accuracy // "N/A"' "$metrics_file" 2>/dev/null || echo "N/A")

        SUMMARY="${SUMMARY}| ${bench} | ${n} | ${acc} |\n"

        # Check for issues
        if [ "$n" -eq 0 ]; then
            ISSUES="${ISSUES}- ${bench}: 0 samples processed\n"
        fi
    else
        ISSUES="${ISSUES}- ${bench}: No metrics.json found\n"
    fi
done

# Update research log
echo ""
echo "$(date): Updating research log..."

TIMESTAMP=$(date +"%Y-%m-%d %H:%M")
cat >> "$RESEARCH_LOG" << EOF

## Qwen3-VL-30B Evaluation Results - $TIMESTAMP

### Run Directory
\`$LATEST_RUN\`

### Results Summary
| Benchmark | Samples | Accuracy |
|-----------|---------|----------|
$(echo -e "$SUMMARY")

EOF

if [ -n "$ISSUES" ]; then
    cat >> "$RESEARCH_LOG" << EOF
### Issues Detected
$(echo -e "$ISSUES")
EOF
    echo "WARNING: Issues detected in evaluation:"
    echo -e "$ISSUES"
else
    cat >> "$RESEARCH_LOG" << EOF
### Status
All benchmarks completed successfully with no discrepancies detected.
EOF
    echo "All benchmarks completed successfully - no discrepancies."
fi

echo ""
echo "$(date): Script completed. Results written to $RESEARCH_LOG"
