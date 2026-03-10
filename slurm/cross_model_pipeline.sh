#!/bin/bash
#SBATCH --job-name=xmodel_uq
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/cross_model_pipeline_%j.log
#SBATCH --error=logs/cross_model_pipeline_%j.log

# Full cross-model evaluation pipeline:
#   Step 1: Run cross-model eval with BOTH calibrators on all targets (4 parallel GPUs)
#   Step 2: Run analysis script (CPU)
#   Step 3: Generate paper figures (CPU)
#
# Uses GPUs 0-3 (daytime safe)
# Total time: ~45 min (eval) + ~5 min (analysis+figures)
#
# Smoke test: sbatch --export=SMOKE_TEST=1 slurm/cross_model_pipeline.sh

echo "=========================================="
echo "Cross-Model Evaluation Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Smoke test: ${SMOKE_TEST:-0}"
echo "=========================================="

cd /scratch/khayes/LLM

source ~/.bashrc
conda activate uq_eval

nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader

SMOKE_ARGS=""
if [ "${SMOKE_TEST:-0}" = "1" ]; then
    SMOKE_ARGS="--smoke_test"
fi

mkdir -p data/cross_model figures/paper

# ============================================================
# STEP 1: Cross-model evaluations (parallel on 4 GPUs)
# ============================================================
# We run 4 evals at once, then another 4, etc.
# Combined calibrator on 3 targets + GPT-5.2 calibrator on 1

echo ""
echo "=========================================="
echo "STEP 1a: Cross-model eval — Batch 1 (4 parallel)"
echo "Start: $(date)"
echo "=========================================="

# Batch 1: Combined calibrator on gpt52, qwen3vl, qwen35 + GPT-5.2 cal on qwen3vl
CUDA_VISIBLE_DEVICES=0 python scripts/cross_model_eval_v3.py \
    --calibrator uq_models/text_calibrator_combined \
    --target gpt52 \
    --output data/cross_model/text_combined_on_gpt52.json \
    $SMOKE_ARGS &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python scripts/cross_model_eval_v3.py \
    --calibrator uq_models/text_calibrator_combined \
    --target qwen3vl \
    --output data/cross_model/text_combined_on_qwen3vl.json \
    $SMOKE_ARGS &
PID2=$!

CUDA_VISIBLE_DEVICES=2 python scripts/cross_model_eval_v3.py \
    --calibrator uq_models/text_calibrator_combined \
    --target qwen35 \
    --output data/cross_model/text_combined_on_qwen35.json \
    $SMOKE_ARGS &
PID3=$!

CUDA_VISIBLE_DEVICES=3 python scripts/cross_model_eval_v3.py \
    --calibrator uq_models/text_calibrator_gpt52 \
    --target qwen3vl \
    --output data/cross_model/text_gpt52cal_on_qwen3vl.json \
    $SMOKE_ARGS &
PID4=$!

echo "Waiting for batch 1: PIDs $PID1 $PID2 $PID3 $PID4"
wait $PID1; RC1=$?
wait $PID2; RC2=$?
wait $PID3; RC3=$?
wait $PID4; RC4=$?
echo "Batch 1 done: $(date)"
echo "  combined→gpt52: exit=$RC1"
echo "  combined→qwen3vl: exit=$RC2"
echo "  combined→qwen35: exit=$RC3"
echo "  gpt52cal→qwen3vl: exit=$RC4"

echo ""
echo "=========================================="
echo "STEP 1b: Cross-model eval — Batch 2 (3 parallel)"
echo "Start: $(date)"
echo "=========================================="

# Batch 2: GPT-5.2 calibrator on gpt52, qwen35 + old v3 calibrator re-eval on gpt52 (for comparison)
CUDA_VISIBLE_DEVICES=0 python scripts/cross_model_eval_v3.py \
    --calibrator uq_models/text_calibrator_gpt52 \
    --target gpt52 \
    --output data/cross_model/text_gpt52cal_on_gpt52.json \
    $SMOKE_ARGS &
PID5=$!

CUDA_VISIBLE_DEVICES=1 python scripts/cross_model_eval_v3.py \
    --calibrator uq_models/text_calibrator_gpt52 \
    --target qwen35 \
    --output data/cross_model/text_gpt52cal_on_qwen35.json \
    $SMOKE_ARGS &
PID6=$!

# Re-eval old v3 calibrator on gpt52 for fair comparison (if it exists)
if [ -d "uq_models/text_calibrator_v3" ]; then
    CUDA_VISIBLE_DEVICES=2 python scripts/cross_model_eval_v3.py \
        --calibrator uq_models/text_calibrator_v3 \
        --target gpt52 \
        --output data/cross_model/text_v3_on_gpt52_reeval.json \
        $SMOKE_ARGS &
    PID7=$!
else
    echo "Skipping v3 re-eval (checkpoint not found)"
    PID7=""
fi

echo "Waiting for batch 2: PIDs $PID5 $PID6 $PID7"
wait $PID5; RC5=$?
wait $PID6; RC6=$?
if [ -n "$PID7" ]; then
    wait $PID7; RC7=$?
else
    RC7="skipped"
fi
echo "Batch 2 done: $(date)"
echo "  gpt52cal→gpt52: exit=$RC5"
echo "  gpt52cal→qwen35: exit=$RC6"
echo "  v3→gpt52 reeval: exit=$RC7"

# ============================================================
# STEP 2: Analysis (CPU only)
# ============================================================

echo ""
echo "=========================================="
echo "STEP 2: Cross-model analysis"
echo "Start: $(date)"
echo "=========================================="

python scripts/analyze_cross_model.py --output_dir figures/analysis
RC_ANALYSIS=$?
echo "Analysis exit code: $RC_ANALYSIS"

# ============================================================
# STEP 3: Paper figures (CPU only)
# ============================================================

echo ""
echo "=========================================="
echo "STEP 3: Generate paper figures"
echo "Start: $(date)"
echo "=========================================="

python scripts/generate_paper_figures.py --output_dir figures/paper
RC_FIGURES=$?
echo "Figures exit code: $RC_FIGURES"

# ============================================================
# SUMMARY
# ============================================================

echo ""
echo "=========================================="
echo "PIPELINE SUMMARY"
echo "=========================================="
echo "Cross-model evals:"
echo "  combined→gpt52: $RC1 | combined→qwen3vl: $RC2 | combined→qwen35: $RC3"
echo "  gpt52cal→qwen3vl: $RC4 | gpt52cal→gpt52: $RC5 | gpt52cal→qwen35: $RC6"
echo "  v3→gpt52 reeval: $RC7"
echo "Analysis: $RC_ANALYSIS"
echo "Figures: $RC_FIGURES"
echo ""
echo "Result files:"
ls -la data/cross_model/text_combined_on_*.json data/cross_model/text_gpt52cal_on_*.json 2>/dev/null
echo ""

# Print AUROC summary
echo "AUROC Summary:"
for f in data/cross_model/text_combined_on_*.json data/cross_model/text_gpt52cal_on_*.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .json)
        auroc=$(python3 -c "import json; print(f'{json.load(open(\"$f\"))[\"auroc\"]:.4f}')" 2>/dev/null || echo "error")
        echo "  $name: AUROC=$auroc"
    fi
done

echo ""
echo "Done: $(date)"
