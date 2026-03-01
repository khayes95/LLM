#!/bin/bash
#SBATCH --job-name=uq_pipeline
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/overnight_pipeline_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/overnight_pipeline_%j.log

# === Full UQ Pipeline: Generate → Grade → Retrain → Score → Use Cases ===
#
# Stages:
#   1. Start Qwen3.5-397B vLLM server
#   2. Generate missing Qwen3.5 text predictions (~1,357 samples)
#   3. Generate missing Qwen3.5 VLM predictions (~43 samples)
#   4. Kill vLLM server
#   5. Grade LLM-judged benchmarks (healthbench, tutorbench, prbench, mmvet)
#   6. Retrain unified model on all data
#   7. Score all predictions with retrained model
#   8. Run all use cases (UC1-UC9)
#
# Estimated time: 6-8 hours total
# GPU requirements: 8x A100-80GB (stages 1-4, 6-7), CPU-only (stages 5, 8)

set -e
cd /scratch/khayes/LLM

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate uq_eval

mkdir -p logs

echo "=========================================="
echo "UQ Overnight Pipeline"
echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Environment for Qwen3.5 vLLM server
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export LD_PRELOAD=/scratch/khayes/nccl-src/build/lib/libnccl.so.2.27.5

# ===========================================================================
# STAGE 1: Start vLLM server for Qwen3.5-397B
# ===========================================================================
echo ""
echo "==========================================="
echo "STAGE 1: Starting Qwen3.5-397B vLLM server"
echo "==========================================="

python -m vllm.entrypoints.openai.api_server \
    --model /scratch/khayes/.cache/huggingface/hub/Qwen3.5-397B-A17B-FP8 \
    --port 8100 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --dtype auto &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for server to start..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8100/v1/models > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server died during startup"
        exit 1
    fi
    sleep 5
done

# Final check
if ! curl -s http://localhost:8100/v1/models > /dev/null 2>&1; then
    echo "ERROR: Server not ready after 10 minutes"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo "Server confirmed running"

# ===========================================================================
# STAGE 2: Generate missing Qwen3.5 TEXT predictions
# ===========================================================================
echo ""
echo "==========================================="
echo "STAGE 2: Generating missing Qwen3.5 text predictions"
echo "==========================================="

# Run all text benchmarks (script handles resume / existing predictions via --include_ids)
python scripts/run_all_qwen35_397b.py \
    --base_url http://localhost:8100/v1 \
    --benchmarks bbeh,healthbench,hle,livebench,omnimath,prbench,tutorbench,arc_agi \
    --parallel 4 \
    2>&1 | tee logs/qwen35_text_stage2.log

echo "Text predictions done: $(date)"

# ===========================================================================
# STAGE 3: Generate missing Qwen3.5 VLM predictions
# ===========================================================================
echo ""
echo "==========================================="
echo "STAGE 3: Generating missing Qwen3.5 VLM predictions"
echo "==========================================="

python scripts/run_all_qwen35_397b_vlm.py \
    --base_url http://localhost:8100/v1 \
    --benchmarks mathvista,mathverse,mathvision,mmvet \
    --parallel 2 \
    2>&1 | tee logs/qwen35_vlm_stage3.log

echo "VLM predictions done: $(date)"

# ===========================================================================
# STAGE 4: Kill vLLM server, free GPUs
# ===========================================================================
echo ""
echo "==========================================="
echo "STAGE 4: Stopping vLLM server"
echo "==========================================="

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
sleep 10
echo "Server stopped"

# Show prediction counts
echo ""
echo "=== Qwen3.5 prediction counts ==="
for bench in bbeh healthbench hle livebench mathverse mathvision omnimath prbench tutorbench arc_agi mathvista mmvet; do
    f="runs/qwen35_397b_${bench}/predictions.jsonl"
    if [ -f "$f" ]; then
        n=$(wc -l < "$f")
        echo "  $bench: $n"
    else
        echo "  $bench: MISSING"
    fi
done

# ===========================================================================
# STAGE 5: Grade LLM-judged benchmarks
# ===========================================================================
echo ""
echo "==========================================="
echo "STAGE 5: Grading LLM-judged benchmarks"
echo "==========================================="

# Grade Qwen3.5 predictions that need LLM judge
GRADE_FILES=(
    "runs/qwen35_397b_healthbench/predictions.jsonl"
    "runs/qwen35_397b_tutorbench/predictions.jsonl"
    "runs/qwen35_397b_prbench/predictions.jsonl"
    "runs/qwen35_397b_mmvet/predictions.jsonl"
)

for f in "${GRADE_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "SKIP: $f (not found)"
        continue
    fi

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

    echo "Grading: $f ($ungraded ungraded)"
    python -m uq_eval.grader \
        --predictions "$f" \
        --output "$f" \
        --judge_backend openai \
        --judge_model gpt-5-mini \
        2>&1 | tee -a logs/grading_stage5.log
    echo "Done: $f"
done

echo "Grading complete: $(date)"

# ===========================================================================
# STAGE 6: Retrain unified model
# ===========================================================================
echo ""
echo "==========================================="
echo "STAGE 6: Retraining unified UQ model"
echo "==========================================="

export CUDA_VISIBLE_DEVICES=0,1,2,3

python scripts/train_best_uq.py \
    --output_dir uq_models/best_unified_v2 \
    --num_epochs 3 \
    --lora_r 16 \
    --learning_rate 2e-5 \
    --test_split 0.15 \
    2>&1 | tee logs/retrain_stage6.log

echo "Training complete: $(date)"

# Check results
if [ -f "uq_models/best_unified_v2/results.json" ]; then
    echo "Results:"
    python3 -c "
import json
r = json.load(open('uq_models/best_unified_v2/results.json'))
print(f'  AUROC: {r[\"auroc\"]:.4f}')
print(f'  VLM:   {r.get(\"vlm_auroc\", \"N/A\")}')
print(f'  Text:  {r.get(\"text_auroc\", \"N/A\")}')
"
fi

# ===========================================================================
# STAGE 7: Score all predictions with new model
# ===========================================================================
echo ""
echo "==========================================="
echo "STAGE 7: Scoring all predictions"
echo "==========================================="

export CUDA_VISIBLE_DEVICES=0

python scripts/score_all_unified.py \
    --checkpoint uq_models/best_unified_v2 \
    --target all \
    --output_dir data/use_cases/scored_unified_v2 \
    2>&1 | tee logs/scoring_stage7.log

echo "Scoring complete: $(date)"

# ===========================================================================
# STAGE 8: Run all use cases
# ===========================================================================
echo ""
echo "==========================================="
echo "STAGE 8: Running all use cases"
echo "==========================================="

SCORED_DIR="data/use_cases/scored_unified_v2"
RESULTS_DIR="data/use_cases/results_unified_v2"
FIGURES_DIR="figures/use_cases_unified_v2"
mkdir -p "$RESULTS_DIR" "$FIGURES_DIR"

for uc in uc1_selective_prediction uc2_model_routing uc3_hallucination_detection uc4_difficulty_estimation uc5_uq_reward_model uc7_self_improvement uc8_ood_detection uc9_annotation_prioritization; do
    echo ""
    echo "--- Running $uc ---"
    python "scripts/${uc}.py" \
        --scored_dir "$SCORED_DIR" \
        --output_dir "$RESULTS_DIR" \
        --fig_dir "$FIGURES_DIR" \
        2>&1 || echo "WARNING: $uc failed"
done

echo ""
echo "==========================================="
echo "PIPELINE COMPLETE"
echo "End: $(date)"
echo "==========================================="
echo ""
echo "Outputs:"
echo "  Model: uq_models/best_unified_v2/"
echo "  Scores: data/use_cases/scored_unified_v2/"
echo "  Results: data/use_cases/results_unified_v2/"
echo "  Figures: figures/use_cases_unified_v2/"
