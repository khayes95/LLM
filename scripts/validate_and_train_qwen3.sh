#!/bin/bash
# Check RESEARCH_LOG.md for Qwen3-VL-30B results and trigger training if successful

cd /scratch/khayes/LLM

RESEARCH_LOG="RESEARCH_LOG.md"

echo "=== Qwen3-VL-30B Validation Script ==="
echo "Started at: $(date)"
echo ""

# Check if the Qwen3-VL results section exists in RESEARCH_LOG.md
echo "Checking RESEARCH_LOG.md for Qwen3-VL-30B evaluation results..."
echo ""

# Look for the results section
if ! grep -q "## Qwen3-VL-30B Evaluation Results" "$RESEARCH_LOG" 2>/dev/null; then
    echo "ERROR: Qwen3-VL-30B Evaluation Results section not found in RESEARCH_LOG.md"
    echo "The run_judge_and_verify.sh script may not have completed yet."
    echo ""
    echo "Waiting for results to appear..."

    # Wait up to 30 minutes for results
    for i in {1..30}; do
        sleep 60
        if grep -q "## Qwen3-VL-30B Evaluation Results" "$RESEARCH_LOG" 2>/dev/null; then
            echo "Results found after $i minutes!"
            break
        fi
        echo "  Waiting... ($i/30 minutes)"
    done

    if ! grep -q "## Qwen3-VL-30B Evaluation Results" "$RESEARCH_LOG" 2>/dev/null; then
        echo "TIMEOUT: Results not found after 30 minutes. Exiting."
        exit 1
    fi
fi

echo "Found Qwen3-VL-30B Evaluation Results section."
echo ""

# Extract and display the results section
echo "=== Results from RESEARCH_LOG.md ==="
# Get everything from "## Qwen3-VL-30B Evaluation Results" until the next "---" or "##"
sed -n '/## Qwen3-VL-30B Evaluation Results/,/^---$\|^## /p' "$RESEARCH_LOG" | head -50
echo ""

# Check for issues
echo "=== Checking for issues ==="

# Check if "Issues Detected" section exists
if grep -A 20 "## Qwen3-VL-30B Evaluation Results" "$RESEARCH_LOG" | grep -q "### Issues Detected"; then
    echo "WARNING: Issues were detected in the Qwen3-VL evaluation!"
    echo ""
    # Extract issues
    grep -A 20 "## Qwen3-VL-30B Evaluation Results" "$RESEARCH_LOG" | grep -A 10 "### Issues Detected"
    echo ""
    echo "Please resolve issues before proceeding with training."
    exit 1
fi

# Check for "All benchmarks completed successfully"
if grep -A 20 "## Qwen3-VL-30B Evaluation Results" "$RESEARCH_LOG" | grep -q "All benchmarks completed successfully"; then
    echo "SUCCESS: All benchmarks completed with no discrepancies!"
else
    echo "WARNING: Could not confirm successful completion."
    echo "Please check RESEARCH_LOG.md manually."

    # Still check if we should proceed
    read -t 10 -p "Proceed anyway? (y/N, 10s timeout): " response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Exiting."
        exit 1
    fi
fi

echo ""
echo "=== Validation Passed ==="
echo ""

# Check if UQ training should be triggered
echo "Checking UQ training status..."

if [ -d "uq_models/gpt5_mini_uq" ] && [ -f "uq_models/gpt5_mini_uq/adapter_config.json" ]; then
    echo "GPT-5-mini UQ model already exists at uq_models/gpt5_mini_uq/"
    echo "Training complete!"
elif pgrep -f "train_uq_unified.py" > /dev/null; then
    echo "UQ training is currently running."
elif pgrep -f "wait_and_train.sh" > /dev/null; then
    echo "wait_and_train.sh is running (waiting for prerequisites)."
else
    echo "UQ training not started and not queued."
    echo ""
    echo "To start training manually:"
    echo "  bash scripts/wait_and_train.sh > logs/wait_and_train.log 2>&1 &"
fi

echo ""
echo "Finished at: $(date)"
