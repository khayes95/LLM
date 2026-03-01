#!/bin/bash
# Grade all ungraded predictions across GPT-5.2 and Qwen3.5
# Uses GPT-5-mini as judge via Micah's API key
# ~933 samples total, estimated cost ~$0.50-1.00
#
# Usage: bash scripts/grade_all_ungraded.sh

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate uq_eval 2>/dev/null

mkdir -p logs

echo "=========================================="
echo "Grading All Ungraded Predictions"
echo "Judge: GPT-5-mini via OpenAI API"
echo "Start: $(date)"
echo "=========================================="

# List of all files that need grading (have correct=-1 samples)
FILES=(
    # GPT-5.2
    "runs/gpt52_high_healthbench/predictions.jsonl"
    "runs/gpt52_high_prbench/predictions.jsonl"
    "runs/gpt52_high_mmvet/predictions.jsonl"
    # Qwen3.5
    "runs/qwen35_397b_healthbench/predictions.jsonl"
    "runs/qwen35_397b_tutorbench/predictions.jsonl"
    "runs/qwen35_397b_prbench/predictions.jsonl"
    "runs/qwen35_397b_mmvet/predictions.jsonl"
)

# Grade each file sequentially (to avoid rate limit issues)
for f in "${FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "SKIP: $f (not found)"
        continue
    fi

    # Count ungraded
    ungraded=$(python3 -c "
import json
count = 0
for line in open('$f'):
    try:
        p = json.loads(line)
        s = p.get('score', {})
        if isinstance(s, dict) and (s.get('correct', -1) == -1 or s.get('needs_grading')):
            count += 1
    except: pass
print(count)
")

    if [ "$ungraded" -eq 0 ]; then
        echo "SKIP: $f (all graded)"
        continue
    fi

    echo ""
    echo "--- Grading: $f ($ungraded ungraded) ---"

    # Grade and write back to same file (grader preserves already-graded)
    python -m uq_eval.grader \
        --predictions "$f" \
        --output "$f" \
        --judge_backend openai \
        --judge_model gpt-5-mini \
        2>&1 | tee -a logs/grade_all_ungraded.log

    echo "--- Done: $f ---"
done

echo ""
echo "=========================================="
echo "All grading complete: $(date)"
echo "=========================================="

# Summary
echo ""
echo "=== Final counts ==="
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        total=$(wc -l < "$f")
        graded=$(python3 -c "
import json
count = 0
for line in open('$f'):
    try:
        p = json.loads(line)
        s = p.get('score', {})
        if isinstance(s, dict) and s.get('correct', -1) in (0, 1):
            count += 1
    except: pass
print(count)
")
        echo "  $f: $graded/$total graded"
    fi
done
