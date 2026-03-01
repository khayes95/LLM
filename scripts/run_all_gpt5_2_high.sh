#!/bin/bash
# Run GPT-5.2 with HIGH reasoning effort on all benchmarks.
# No GPU needed — pure API calls. Run on debug partition.
#
# Smoke test: ./scripts/run_all_gpt5_2_high.sh --max-samples 3
# Full run:   ./scripts/run_all_gpt5_2_high.sh --max-samples 100 --seed 42
#
# Uses fixed output dirs (runs/gpt52_high_<bench>/) so --resume works across runs.
# Smoke test samples are kept and the full run picks up where it left off.
#
# Cost estimate: ~$20-50 for 100 samples/benchmark (reasoning tokens are expensive)

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
MAX_SAMPLES="${MAX_SAMPLES:-100}"
SEED="${SEED:-42}"
DRY_RUN=false
OUTPUT_BASE="runs"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --reasoning-effort) REASONING_EFFORT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Reasoning models need longer timeouts (can take 60-180s per request with high effort)
TIMEOUT_S="${TIMEOUT_S:-600}"

# Higher max_output_tokens for reasoning models (reasoning tokens count against limit)
# The OpenAI client already enforces min 16384 for reasoning models, but CLI default is 256.
# We set 512 here; the client will bump it to 16384 internally.
MAX_OUTPUT_TOKENS=512

echo "============================================================"
echo "GPT-5.2 EVALUATION (reasoning_effort=${REASONING_EFFORT})"
echo "============================================================"
echo "Model: $MODEL"
echo "Backend: $MODEL_BACKEND"
echo "Reasoning effort: $REASONING_EFFORT"
echo "Max samples: $MAX_SAMPLES"
echo "Seed: $SEED"
echo "Timeout: ${TIMEOUT_S}s"
echo "Max output tokens: $MAX_OUTPUT_TOKENS (client min: 16384)"
echo "Dry run: $DRY_RUN"
echo ""

# Ensure logs directory exists
mkdir -p logs

# Text benchmarks — same tiers as gpt-5-mini
TEXT_TIER1="bbeh simpleqa prbench tutorbench healthbench"
TEXT_TIER2="hle arc_agi chembench"
TEXT_TIER3="gpqa livebench omnimath"
TEXT_LC="oolong babilong"
ALL_TEXT="$TEXT_TIER1 $TEXT_TIER2 $TEXT_TIER3 $TEXT_LC"

# VLM benchmarks
VLM_HARD="mathvision mathverse hallusionbench"
VLM_MED="mathvista mmmu mmstar charxiv realworldqa aokvqa vizwiz mmvet vsr"
ALL_VLM="$VLM_HARD $VLM_MED"

# Parallelism — reasoning models with high effort are slower, use fewer parallel jobs
# to avoid rate limits. Adjust based on your API tier.
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
echo "Parallel jobs: $PARALLEL_JOBS"

run_benchmark_bg() {
    local bench=$1
    local extra_args=$2
    local logfile="logs/gpt52_${bench}.log"
    # Fixed output dir so --resume works across smoke test -> full run
    local out_dir="${OUTPUT_BASE}/gpt52_high_${bench}"

    cmd="python -m uq_eval.cli --bench $bench --model_backend $MODEL_BACKEND --model_name $MODEL"
    cmd="$cmd --timeout_s $TIMEOUT_S --max_output_tokens $MAX_OUTPUT_TOKENS"
    cmd="$cmd --reasoning_effort $REASONING_EFFORT"
    cmd="$cmd --out_dir $out_dir"

    if [[ -n "$MAX_SAMPLES" ]]; then
        cmd="$cmd --max_examples $MAX_SAMPLES"
    fi

    if [[ -n "$SEED" ]]; then
        cmd="$cmd --seed $SEED"
    fi

    if [[ -n "$extra_args" ]]; then
        cmd="$cmd $extra_args"
    fi

    echo "  [BG] $bench -> $logfile"
    echo "       $cmd"

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

echo ""
echo "============================================================"
echo "TEXT BENCHMARKS (parallel: $PARALLEL_JOBS jobs)"
echo "============================================================"

for bench in $ALL_TEXT; do
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
echo "============================================================"
echo "VLM BENCHMARKS (parallel: $PARALLEL_JOBS jobs)"
echo "============================================================"

for bench in $ALL_VLM; do
    wait_for_jobs $PARALLEL_JOBS
    run_benchmark_bg "$bench"
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
echo "Runs saved to: $OUTPUT_BASE/"
echo "Logs saved to: logs/gpt52_*.log"
echo ""
echo "Check results:"
echo "  for f in logs/gpt52_*.log; do echo \"=== \$f ===\"; tail -3 \$f; done"
