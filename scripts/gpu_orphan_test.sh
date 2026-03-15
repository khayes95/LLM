#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/khayes/LLM/logs/gpu_test_%j.out
#SBATCH --error=/scratch/khayes/LLM/logs/gpu_test_%j.err
#
# GPU Orphan and Brick Test Suite
#
# Usage (submit with sbatch, pass --mode and optionally --fix):
#
#   PART 1 - ORPHAN TESTS (2-min time limit, test one at a time):
#
#     sbatch --time=00:02:00 -J orphan_repro    gpu_orphan_test.sh --mode repro
#     sbatch --time=00:02:00 -J fix1_srun       gpu_orphan_test.sh --mode repro    --fix srun
#     sbatch --time=00:02:00 -J fix2_signal     gpu_orphan_test.sh --mode repro    --fix signal
#     sbatch --time=00:02:00 -J fix3_python     gpu_orphan_test.sh --mode fixed
#     sbatch --time=00:02:00 -J fix_all         gpu_orphan_test.sh --mode fixed    --fix all
#     sbatch --time=00:10:00 -J normal_job      gpu_orphan_test.sh --mode normal   --fix all
#
#   PART 2 - BRICK TESTS (off-hours, dedicated GPU):
#
#     sbatch --time=00:10:00 -J brick_kernel    gpu_orphan_test.sh --mode brick_kernel
#     sbatch --time=00:10:00 -J brick_teardown  gpu_orphan_test.sh --mode brick_teardown
#     sbatch --time=04:00:00 -J brick_loop      gpu_orphan_test.sh --mode brick_loop
#
# After each test check:
#     ps aux | grep -E "vllm|VLLM" | grep -v grep
#     nvidia-smi

# ---- Parse arguments ----
MODE=""
FIX=""
KILL_DELAY=30
ITERATIONS=50

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)       MODE="$2";       shift 2 ;;
        --fix)        FIX="$2";        shift 2 ;;
        --kill_delay) KILL_DELAY="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "ERROR: --mode is required"
    echo "Modes: repro, fixed, normal, brick_kernel, brick_teardown, brick_loop"
    exit 1
fi

# ---- Setup ----
conda activate gputest
cd /scratch/khayes/LLM
source scripts/gpu_test_helpers.sh

echo "=========================================="
echo "GPU Orphan/Brick Test"
echo "Mode: $MODE"
echo "Fix:  ${FIX:-none}"
echo "Start: $(date)"
echo "SLURM assigned CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "=========================================="
echo ""

gpu_health_check || exit 1

# ---- Build the python command ----
PYCMD="python scripts/gpu_orphan_test.py --mode $MODE"

if [ "$MODE" = "brick_kernel" ] || [ "$MODE" = "brick_loop" ]; then
    PYCMD="$PYCMD --kill_delay $KILL_DELAY"
fi
if [ "$MODE" = "brick_loop" ]; then
    PYCMD="$PYCMD --gpu $CUDA_VISIBLE_DEVICES --iterations $ITERATIONS"
fi

# ---- Apply fix strategy ----
case "$FIX" in
    srun)
        # Fix 1: srun only
        echo "Applying fix: srun (SLURM tracks full process tree)"
        srun $PYCMD
        ;;
    signal)
        # Fix 2: shell signal trap only
        echo "Applying fix: shell signal trap"
        cleanup() {
            echo "Shell trap: caught signal, killing process group..."
            kill -TERM -$$ 2>/dev/null || true
            wait
            echo "Shell cleanup done: $(date)"
            post_test_report
        }
        trap cleanup USR1 TERM EXIT
        $PYCMD &
        wait $!
        ;;
    all)
        # Fix 1+2+3: srun + shell trap + Python handlers
        echo "Applying fix: srun + shell trap + Python handlers"
        cleanup() {
            echo "Shell trap: caught signal, forwarding to process group..."
            kill -TERM -$$ 2>/dev/null || true
            wait
            echo "Shell cleanup done: $(date)"
            post_test_report
        }
        trap cleanup USR1 TERM EXIT
        srun $PYCMD &
        wait $!
        ;;
    ""|none)
        # No fix: bare python
        echo "No fix applied (bare python)"
        $PYCMD
        ;;
    *)
        echo "ERROR: Unknown fix '$FIX'. Options: srun, signal, all"
        exit 1
        ;;
esac

echo ""
echo "Finished: $(date)"
post_test_report
