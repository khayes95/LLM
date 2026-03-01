#!/bin/bash
# Scale GPT-5.2 (high reasoning) to 250 samples per benchmark.
# Resumes from existing 100-sample runs — only runs the ~150 new samples.
# No GPU needed — pure API calls. Run on debug partition.
#
# Dropped benchmarks (>=90% accuracy, no UQ signal):
#   babilong (100%), aokvqa (90%)
#
# Smoke test: ./scripts/run_gpt52_250.sh --dry-run
# Full run:   ./scripts/run_gpt52_250.sh
#
# Estimated cost: ~$100-130 for 150 new samples/benchmark x 23 benchmarks

set -e

# Source bashrc for API key
source /home/khayes/.bashrc

# Ensure API key is available
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "ERROR: OPENAI_API_KEY not set. Add it to ~/.bashrc"
    exit 1
fi

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate uq_eval

MODEL="gpt-5.2"
MODEL_BACKEND="openai"
REASONING_EFFORT="high"
MAX_SAMPLES=250
SEED=42
DRY_RUN=false
OUTPUT_BASE="runs"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --parallel) PARALLEL_JOBS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Reasoning models need longer timeouts
TIMEOUT_S="${TIMEOUT_S:-900}"

# Use 65536 max_output_tokens to avoid reasoning token exhaustion
# (16384 was too low — caused 77 empty responses in the 100-sample run)
MAX_OUTPUT_TOKENS=65536

# 16 parallel benchmark jobs — each is a sleeping API call, not CPU-bound
PARALLEL_JOBS="${PARALLEL_JOBS:-16}"

echo "============================================================"
echo "GPT-5.2 SCALE-UP TO 250 SAMPLES"
echo "============================================================"
echo "Model: $MODEL"
echo "Reasoning effort: $REASONING_EFFORT"
echo "Max samples: $MAX_SAMPLES (resume from existing 100)"
echo "Seed: $SEED"
echo "Timeout: ${TIMEOUT_S}s"
echo "Max output tokens: $MAX_OUTPUT_TOKENS"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Dry run: $DRY_RUN"
echo ""

# Ensure logs directory exists
mkdir -p logs

# Text benchmarks (dropped: babilong — 100% accuracy)
TEXT_BENCHMARKS="bbeh simpleqa prbench tutorbench healthbench hle arc_agi chembench gpqa livebench omnimath oolong"

# VLM benchmarks (dropped: aokvqa — 90% accuracy)
VLM_BENCHMARKS="mathvision mathverse hallusionbench mathvista mmmu mmstar charxiv realworldqa vizwiz mmvet vsr"

ALL_BENCHMARKS="$TEXT_BENCHMARKS $VLM_BENCHMARKS"

run_benchmark_bg() {
    local bench=$1
    local extra_args=$2
    local logfile="logs/gpt52_250_${bench}.log"
    local out_dir="${OUTPUT_BASE}/gpt52_high_${bench}"

    cmd="python -m uq_eval.cli --bench $bench --model_backend $MODEL_BACKEND --model_name $MODEL"
    cmd="$cmd --timeout_s $TIMEOUT_S --max_output_tokens $MAX_OUTPUT_TOKENS"
    cmd="$cmd --reasoning_effort $REASONING_EFFORT"
    cmd="$cmd --out_dir $out_dir"
    cmd="$cmd --max_examples $MAX_SAMPLES"
    cmd="$cmd --seed $SEED"

    if [[ -n "$extra_args" ]]; then
        cmd="$cmd $extra_args"
    fi

    # Count existing samples
    local existing=0
    if [[ -f "${out_dir}/predictions.jsonl" ]]; then
        existing=$(wc -l < "${out_dir}/predictions.jsonl")
    fi
    local remaining=$((MAX_SAMPLES - existing))
    if [[ $remaining -le 0 ]]; then
        echo "  [SKIP] $bench: already has $existing samples (>= $MAX_SAMPLES)"
        return
    fi

    echo "  [BG] $bench: $existing existing, ~$remaining new -> $logfile"

    if [[ "$DRY_RUN" == "false" ]]; then
        $cmd > "$logfile" 2>&1 &
    fi
}

wait_for_jobs() {
    local max_jobs=$1
    while [[ $(jobs -r | wc -l) -ge $max_jobs ]]; do
        sleep 10
    done
}

echo "============================================================"
echo "LAUNCHING BENCHMARKS (parallel: $PARALLEL_JOBS jobs)"
echo "============================================================"

for bench in $ALL_BENCHMARKS; do
    wait_for_jobs $PARALLEL_JOBS
    case $bench in
        gpqa)
            run_benchmark_bg "$bench" "--subset gpqa_diamond"
            ;;
        hle)
            run_benchmark_bg "$bench" "--hle_answer_type exact_match"
            ;;
        *)
            run_benchmark_bg "$bench"
            ;;
    esac
done

echo ""
echo "Waiting for all benchmarks to complete..."
wait
echo "All benchmarks completed!"

echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo ""

# Print per-benchmark results
for bench in $ALL_BENCHMARKS; do
    local_dir="${OUTPUT_BASE}/gpt52_high_${bench}"
    if [[ -f "${local_dir}/metrics.json" ]]; then
        n=$(python3 -c "import json; print(json.load(open('${local_dir}/metrics.json')).get('n', '?'))" 2>/dev/null)
        acc=$(python3 -c "import json; print(f\"{json.load(open('${local_dir}/metrics.json')).get('accuracy', 0):.1%}\")" 2>/dev/null)
        echo "  $bench: $acc ($n samples)"
    elif [[ -f "${local_dir}/predictions.jsonl" ]]; then
        n=$(wc -l < "${local_dir}/predictions.jsonl")
        echo "  $bench: $n samples (no metrics yet)"
    else
        echo "  $bench: NOT STARTED"
    fi
done

echo ""
echo "Logs: logs/gpt52_250_*.log"
echo "Check progress: for f in logs/gpt52_250_*.log; do echo \"=== \$f ===\"; tail -3 \$f; done"
