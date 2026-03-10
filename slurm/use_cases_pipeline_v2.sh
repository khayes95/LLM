#!/bin/bash
#SBATCH --job-name=uc_pipeline_v2
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/khayes/LLM/logs/uc_pipeline_v2_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/uc_pipeline_v2_%j.log

# Run all use case analyses on scored_test_only_v2 (test-only data)
# All CPU-only, no GPU needed

set -e

cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "=========================================="
echo "Use Cases Pipeline v2 (model-specific calibrators)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

SCORED_DIR="data/use_cases/scored_test_only_v2"
OUTPUT_DIR="data/use_cases/results"
FIG_DIR="figures/use_cases"

mkdir -p "$OUTPUT_DIR" "$FIG_DIR"

echo ""
echo "--- UC1: Selective Prediction ---"
python scripts/uc1_selective_prediction.py --scored_dir "$SCORED_DIR" --output_dir "$OUTPUT_DIR" --fig_dir "$FIG_DIR"

echo ""
echo "--- UC2: Model Routing ---"
python scripts/uc2_model_routing.py --scored_dir "$SCORED_DIR" --output_dir "$OUTPUT_DIR" --fig_dir "$FIG_DIR"

echo ""
echo "--- UC3: Error Flagging + Labeling Efficiency ---"
python scripts/uc3_hallucination_detection.py --scored_dir "$SCORED_DIR" --output_dir "$OUTPUT_DIR" --fig_dir "$FIG_DIR"

echo ""
echo "--- UC4: Difficulty Estimation ---"
python scripts/uc4_difficulty_estimation.py --scored_dir "$SCORED_DIR" --output_dir "$OUTPUT_DIR" --fig_dir "$FIG_DIR"

echo ""
echo "--- UC5: Reward Model (fixed with normalization) ---"
python scripts/uc5_uq_reward_model.py --scored_dir "$SCORED_DIR" --output_dir "$OUTPUT_DIR" --fig_dir "$FIG_DIR"

echo ""
echo "--- UC7: Self-Improvement ---"
python scripts/uc7_self_improvement.py --scored_dir "$SCORED_DIR" --output_dir "$OUTPUT_DIR" --fig_dir "$FIG_DIR"

echo ""
echo "--- UC8: Deployment Monitoring (new) ---"
python scripts/uc8_deployment_monitoring.py --scored_dir "$SCORED_DIR" --output_dir "$OUTPUT_DIR" --fig_dir "$FIG_DIR"

echo ""
echo "=========================================="
echo "All use cases complete"
echo "Results: $OUTPUT_DIR/uc{1,2,3,4,5,7,8}_results.json"
echo "Figures: $FIG_DIR/uc*.pdf"
echo "End: $(date)"
echo "=========================================="
