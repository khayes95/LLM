#!/bin/bash
#SBATCH --job-name=missing_api
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/missing_api_%j.log
#SBATCH --error=logs/missing_api_%j.log

# Complete missing/incomplete API benchmarks:
# GPT-5-mini: arc_agi (250), hallusionbench (250), mmvet (250)
# GPT-5.2: hle_multimodal (250)
#
# Smoke test: SMOKE=1 sbatch slurm/complete_missing_benchmarks_api.sh

echo "=========================================="
echo "Complete Missing API Benchmarks"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

source /scratch/khayes/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval
source /home/khayes/.bashrc

echo "OPENAI_API_KEY set: ${OPENAI_API_KEY:0:8}..."
echo "Python: $(which python)"

mkdir -p logs

MAX_SAMPLES=250
TIMEOUT_S=600
SEED=42

if [[ -n "$SMOKE" ]]; then
    MAX_SAMPLES=3
    echo "*** SMOKE TEST MODE (3 samples) ***"
fi

run_bench() {
    local model=$1
    local backend=$2
    local bench=$3
    local out_dir=$4
    local extra=$5
    local logfile="logs/missing_${model}_${bench}.log"

    cmd="python -m uq_eval.cli --bench $bench --model_backend $backend --model_name $model"
    cmd="$cmd --timeout_s $TIMEOUT_S --max_output_tokens 512"
    cmd="$cmd --out_dir $out_dir --max_examples $MAX_SAMPLES --seed $SEED"

    if [[ -n "$extra" ]]; then
        cmd="$cmd $extra"
    fi

    echo ""
    echo "=== $bench ($model) ==="
    echo "  CMD: $cmd"
    echo "  LOG: $logfile"
    $cmd > "$logfile" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "  DONE: $bench succeeded"
        tail -3 "$logfile"
    else
        echo "  FAILED: $bench exit code $rc"
        tail -10 "$logfile"
    fi
    return $rc
}

echo ""
echo "============================================================"
echo "GPT-5-mini: Missing Benchmarks"
echo "============================================================"

# arc_agi - completely missing for gpt5mini
run_bench "gpt-5-mini" "openai" "arc_agi" "runs/gpt5_mini_combined/arc_agi" ""

# hallusionbench - have 100, need 250 (will overwrite with 250)
run_bench "gpt-5-mini" "openai" "hallusionbench" "runs/gpt5_mini_combined/hallusionbench" ""

# mmvet - have 100, need 250 (will overwrite with 250)
run_bench "gpt-5-mini" "openai" "mmvet" "runs/gpt5_mini_combined/mmvet" ""

echo ""
echo "============================================================"
echo "GPT-5.2: Missing Benchmarks"
echo "============================================================"

# hle_multimodal - missing for gpt52
run_bench "gpt-5.2" "openai" "hle" "runs/gpt52_high_hle_multimodal" "--hle_with_images --reasoning_effort high"

echo ""
echo "=========================================="
echo "All API benchmarks finished: $(date)"
echo "=========================================="
