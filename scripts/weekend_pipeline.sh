#!/bin/bash
# Weekend Pipeline: GPT-5.2 grading + Qwen3.5 inference + cross-model evals + ablation
#
# Usage:
#   bash scripts/weekend_pipeline.sh                 # Full run
#   bash scripts/weekend_pipeline.sh --smoke_test    # Quick smoke test (~20 min)
#   bash scripts/weekend_pipeline.sh --phase 3       # Resume from phase 3
#
# Phases:
#   0: Grade GPT-5.2 open-ended benchmarks (API, 0 GPU) — runs in background
#   1: Qwen3.5-397B inference via vLLM (8 GPUs, TP=8)
#   2: Cross-model eval on GPT-5.2 (text + VLM, 1 GPU)
#   3: Cross-model eval on Qwen3.5 (text + VLM, 1 GPU)
#   4: VLM training size ablation with VSR fix (4 GPUs)
#   5: Summary
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Ensure conda environment is active (bashrc first, then conda)
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "uq_eval" ]]; then
    source /home/khayes/.bashrc 2>/dev/null || true
    eval "$(/scratch/khayes/anaconda3/bin/conda shell.bash hook)" 2>/dev/null || true
    conda activate uq_eval 2>/dev/null || true
fi

STATE_DIR="logs/weekend_pipeline_state"
LOGFILE="logs/weekend_pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$STATE_DIR" logs

# Parse arguments
SMOKE_TEST=0
START_PHASE=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke_test) SMOKE_TEST=1; shift ;;
        --phase) START_PHASE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Logging helper
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGFILE"; }

# Cleanup: kill ALL child processes on exit (prevents orphaned GPU processes)
VLLM_PID=""
cleanup() {
    log "Cleaning up all child processes..."
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        log "  Killing vLLM server (PID $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    # Kill entire process group to prevent orphaned GPU processes
    kill -- -$$ 2>/dev/null || true
    log "Pipeline finished."
}
trap cleanup EXIT SIGTERM SIGINT

phase_done() { [[ -f "$STATE_DIR/phase_${1}_done" ]]; }
mark_done()  { date > "$STATE_DIR/phase_${1}_done"; log "Phase $1 complete."; }

# ============================================================
# PHASE 0: Grade GPT-5.2 open-ended benchmarks
# ============================================================
run_phase_0() {
    if phase_done 0; then log "Phase 0 already done, skipping."; return; fi
    log "=== PHASE 0: Grading GPT-5.2 open-ended benchmarks ==="

    # Grading is API-only — hide GPUs so it doesn't consume VRAM
    export CUDA_VISIBLE_DEVICES=""

    local BENCHMARKS_TO_GRADE=(healthbench prbench tutorbench)
    # mmvet is also open-ended but has its own grading

    if [[ $SMOKE_TEST -eq 1 ]]; then
        BENCHMARKS_TO_GRADE=(healthbench)
    fi

    for bench in "${BENCHMARKS_TO_GRADE[@]}"; do
        local pred_file="runs/gpt52_high_${bench}/predictions.jsonl"
        if [[ ! -f "$pred_file" ]]; then
            log "  Skipping $bench: no predictions file"
            continue
        fi

        # Check if already graded (score.correct exists and != -1)
        local needs_grading
        needs_grading=$(python3 -c "
import json
with open('$pred_file') as f:
    for line in f:
        d = json.loads(line)
        s = d.get('score', {})
        c = s.get('correct', -1) if isinstance(s, dict) else s
        if c not in (0, 1):
            print('yes')
            break
    else:
        print('no')
" 2>/dev/null || echo "yes")

        if [[ "$needs_grading" == "no" ]]; then
            log "  $bench already graded, skipping."
            continue
        fi

        log "  Grading $bench..."
        local grade_args="--predictions $pred_file --output $pred_file --judge_backend openai --judge_model gpt-5-mini"
        if [[ $SMOKE_TEST -eq 1 ]]; then
            grade_args="$grade_args --max_examples 3"
        fi
        python -m uq_eval.grader $grade_args 2>&1 | tee -a "$LOGFILE" || {
            log "  WARNING: Grading $bench failed, continuing..."
        }
    done

    mark_done 0
}

# ============================================================
# PHASE 1: Qwen3.5-397B inference via vLLM
# ============================================================
run_phase_1() {
    if phase_done 1; then log "Phase 1 already done, skipping."; return; fi
    log "=== PHASE 1: Qwen3.5-397B inference (8 GPUs, TP=8) ==="

    # Set up vLLM environment
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export NCCL_DEBUG=WARN
    export LD_PRELOAD=/scratch/khayes/nccl-src/build/lib/libnccl.so.2.27.5

    local MODEL_PATH="/scratch/khayes/.cache/huggingface/hub/Qwen3.5-397B-A17B-FP8"
    local PORT=8100

    if [[ ! -d "$MODEL_PATH" ]]; then
        log "ERROR: Qwen3.5 model not found at $MODEL_PATH"
        log "Skipping Phase 1."
        return
    fi

    # Start vLLM server in background
    log "  Starting vLLM server on port $PORT..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --tensor-parallel-size 8 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.85 \
        --enforce-eager \
        --dtype auto \
        > logs/vllm_qwen35_server.log 2>&1 &
    VLLM_PID=$!
    log "  vLLM PID: $VLLM_PID"

    # Wait for server readiness
    log "  Waiting for server..."
    if ! python scripts/smoke_test_qwen35_server.py --base_url "http://localhost:${PORT}/v1" --wait_timeout 900; then
        log "ERROR: vLLM server failed to start."
        kill "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
        return 1
    fi

    # Run benchmarks
    local runner_args="--base_url http://localhost:${PORT}/v1 --parallel 4"
    if [[ $SMOKE_TEST -eq 1 ]]; then
        runner_args="$runner_args --smoke_test"
    fi

    log "  Running text benchmarks..."
    python scripts/run_all_qwen35_397b.py $runner_args 2>&1 | tee -a "$LOGFILE" || {
        log "WARNING: Some Qwen3.5 text benchmarks may have failed."
    }

    # Run VLM benchmarks (Qwen3.5 has a vision encoder)
    local vlm_args="--base_url http://localhost:${PORT}/v1 --parallel 2"
    if [[ $SMOKE_TEST -eq 1 ]]; then
        vlm_args="$vlm_args --smoke_test"
    fi

    log "  Running VLM benchmarks..."
    python scripts/run_all_qwen35_397b_vlm.py $vlm_args 2>&1 | tee -a "$LOGFILE" || {
        log "WARNING: Some Qwen3.5 VLM benchmarks may have failed."
    }

    # Shutdown vLLM server
    log "  Shutting down vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""

    # Free GPU memory
    sleep 10

    mark_done 1
}

# ============================================================
# PHASE 2: Cross-model eval on GPT-5.2
# ============================================================
run_phase_2() {
    if phase_done 2; then log "Phase 2 already done, skipping."; return; fi
    log "=== PHASE 2: Cross-model eval on GPT-5.2 ==="

    export CUDA_VISIBLE_DEVICES=0,1

    # Text calibrator
    log "  Running text calibrator on GPT-5.2..."
    local text_args="--target gpt52"
    if [[ $SMOKE_TEST -eq 1 ]]; then
        text_args="$text_args --smoke_test"
    fi
    python scripts/cross_model_eval_gpt52.py $text_args 2>&1 | tee -a "$LOGFILE" || {
        log "WARNING: Text cross-model eval on GPT-5.2 failed."
    }

    # VLM judge
    log "  Running VLM judge on GPT-5.2..."
    local vlm_args="--target gpt52"
    if [[ $SMOKE_TEST -eq 1 ]]; then
        vlm_args="$vlm_args --smoke_test"
    fi
    python scripts/vlm_judge_cross_model_eval.py $vlm_args 2>&1 | tee -a "$LOGFILE" || {
        log "WARNING: VLM cross-model eval on GPT-5.2 failed."
    }

    mark_done 2
}

# ============================================================
# PHASE 3: Cross-model eval on Qwen3.5
# ============================================================
run_phase_3() {
    if phase_done 3; then log "Phase 3 already done, skipping."; return; fi
    log "=== PHASE 3: Cross-model eval on Qwen3.5 ==="

    # Check if Qwen3.5 predictions exist
    local has_data=0
    for d in runs/qwen35_397b_*/predictions.jsonl; do
        if [[ -f "$d" ]]; then has_data=1; break; fi
    done

    if [[ $has_data -eq 0 ]]; then
        log "  No Qwen3.5 predictions found. Skipping Phase 3."
        mark_done 3
        return
    fi

    export CUDA_VISIBLE_DEVICES=0,1

    # Text calibrator
    log "  Running text calibrator on Qwen3.5..."
    local text_args="--target qwen35"
    if [[ $SMOKE_TEST -eq 1 ]]; then
        text_args="$text_args --smoke_test"
    fi
    python scripts/cross_model_eval_gpt52.py $text_args 2>&1 | tee -a "$LOGFILE" || {
        log "WARNING: Text cross-model eval on Qwen3.5 failed."
    }

    # VLM judge
    log "  Running VLM judge on Qwen3.5..."
    local vlm_args="--target qwen35"
    if [[ $SMOKE_TEST -eq 1 ]]; then
        vlm_args="$vlm_args --smoke_test"
    fi
    python scripts/vlm_judge_cross_model_eval.py $vlm_args 2>&1 | tee -a "$LOGFILE" || {
        log "WARNING: VLM cross-model eval on Qwen3.5 failed."
    }

    mark_done 3
}

# ============================================================
# PHASE 4: VLM training size ablation (VSR-fixed)
# ============================================================
run_phase_4() {
    if phase_done 4; then log "Phase 4 already done, skipping."; return; fi
    log "=== PHASE 4: VLM training size ablation (VSR-fixed) ==="

    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    local ablation_args="--run-all-subprocess --skip-transfer"
    if [[ $SMOKE_TEST -eq 1 ]]; then
        ablation_args="$ablation_args --quick"
    fi

    python scripts/vlm_training_size_ablation.py $ablation_args 2>&1 | tee -a "$LOGFILE" || {
        log "WARNING: VLM ablation failed or partially completed."
    }

    mark_done 4
}

# ============================================================
# PHASE 5: Summary
# ============================================================
run_phase_5() {
    log "=== PHASE 5: Summary ==="

    python3 -c "
import json
from pathlib import Path

print()
print('=' * 70)
print('WEEKEND PIPELINE RESULTS')
print('=' * 70)

# Cross-model results
for name in ['text_v3_on_gpt52', 'text_v3_on_qwen35',
             'vlm_judge_vsr_fixed_on_gpt52', 'vlm_judge_vsr_fixed_on_qwen35']:
    path = Path(f'data/cross_model/{name}.json')
    if path.exists():
        d = json.load(open(path))
        auroc = d.get('auroc', '?')
        n = d.get('n_samples', '?')
        print(f'  {name}: AUROC={auroc:.4f}, n={n}')
    else:
        print(f'  {name}: NOT FOUND')

# Ablation results
ablation_path = Path('data/ablations/vlm_training_size_vsr_fixed/ablation_results.json')
if ablation_path.exists():
    d = json.load(open(ablation_path))
    print()
    print('VLM Ablation Results:')
    for r in d.get('results', []):
        print(f\"  size={r['size']}: AUROC={r['vision_auroc']:.4f}\")
else:
    print('  VLM Ablation: NOT FOUND')

# Qwen3.5 benchmark counts
qwen_dirs = list(Path('runs').glob('qwen35_397b_*'))
if qwen_dirs:
    print()
    print(f'Qwen3.5-397B benchmarks: {len(qwen_dirs)} dirs')
    for d in sorted(qwen_dirs):
        pred = d / 'predictions.jsonl'
        if pred.exists():
            n = sum(1 for _ in open(pred))
            print(f'  {d.name}: {n} predictions')

print()
print('=' * 70)
" 2>&1 | tee -a "$LOGFILE"

    mark_done 5
}

# ============================================================
# MAIN: Run phases
# ============================================================
log "Weekend Pipeline starting (smoke_test=$SMOKE_TEST, start_phase=$START_PHASE)"
log "Log file: $LOGFILE"
log ""

# Phase 0 (grading) runs in background with GPUs hidden so it doesn't steal VRAM.
# Phase 1 (vLLM) needs all 8 GPUs and starts in parallel.
if [[ $START_PHASE -le 0 ]]; then
    CUDA_VISIBLE_DEVICES="" run_phase_0 &
    GRADE_PID=$!
fi

PHASE1_OK=0
if [[ $START_PHASE -le 1 ]]; then
    if run_phase_1; then
        PHASE1_OK=1
    else
        log "Phase 1 FAILED — skipping subsequent phases that depend on it."
    fi
fi

# Wait for grading to finish before cross-model eval
if [[ -n "${GRADE_PID:-}" ]]; then
    log "Waiting for Phase 0 (grading) to finish..."
    wait "$GRADE_PID" 2>/dev/null || true
fi

if [[ $START_PHASE -le 2 ]]; then run_phase_2; fi
if [[ $START_PHASE -le 3 ]]; then run_phase_3; fi
if [[ $START_PHASE -le 4 ]]; then run_phase_4; fi

run_phase_5

log "All phases complete!"
