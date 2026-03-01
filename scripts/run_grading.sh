#!/bin/bash
# Run all GPT-5-mini grading jobs in parallel
# Usage: bash scripts/run_grading.sh

cd /scratch/khayes/LLM

echo "Starting GPT-5-mini grading jobs in parallel..."
echo "Each job will log to logs/grade_<benchmark>.log"

mkdir -p logs

# Run all 4 grading jobs in parallel (using openai backend for proper API key handling)
python -m uq_eval.grader --predictions runs/gpt5_mini_combined/healthbench/predictions.jsonl --judge_backend openai --judge_model gpt-5-mini > logs/grade_healthbench.log 2>&1 &
PID1=$!
echo "HealthBench: PID $PID1"

python -m uq_eval.grader --predictions runs/gpt5_mini_combined/tutorbench/predictions.jsonl --judge_backend openai --judge_model gpt-5-mini > logs/grade_tutorbench.log 2>&1 &
PID2=$!
echo "TutorBench: PID $PID2"

python -m uq_eval.grader --predictions runs/gpt5_mini_combined/prbench/predictions.jsonl --judge_backend openai --judge_model gpt-5-mini > logs/grade_prbench.log 2>&1 &
PID3=$!
echo "PRBench: PID $PID3"

python -m uq_eval.grader --predictions runs/gpt5_mini_combined/mmvet/predictions.jsonl --judge_backend openai --judge_model gpt-5-mini > logs/grade_mmvet.log 2>&1 &
PID4=$!
echo "MMVet: PID $PID4"

echo ""
echo "All jobs started. Monitor with:"
echo "  tail -f logs/grade_*.log"
echo ""
echo "Wait for all to complete..."
wait $PID1 $PID2 $PID3 $PID4

echo ""
echo "=== All grading complete ==="
echo ""
for bench in healthbench tutorbench prbench mmvet; do
    echo "=== $bench ==="
    tail -5 logs/grade_$bench.log
    echo ""
done
