#!/bin/bash
#SBATCH --job-name=uc_all_v2
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/uc_all_v2_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/uc_all_v2_%j.log

# Run ALL improved use case scripts on v2 test-only scored data.
# These are ALL CPU-only — no GPU needed.
#
# Smoke test:
#   python scripts/uc1_selective_prediction.py --smoke_test
#   python scripts/uc3_error_discovery.py --smoke_test

set -e
cd /scratch/khayes/LLM

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate uq_eval

# No GPU needed — CPU-only analyses
export CUDA_VISIBLE_DEVICES=""

SCORED="data/use_cases/scored_test_only_v2"
RESULTS="data/use_cases/results_test_only_v2"
FIGS="figures/use_cases_v2"

echo "============================================"
echo "ALL USE CASES V2 PIPELINE"
echo "Job: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "============================================"

# UC1: Selective Prediction (enhanced: response length baseline, bootstrap CIs)
echo ""
echo ">>> UC1: Selective Prediction ($(date))"
python scripts/uc1_selective_prediction.py \
    --scored_dir "$SCORED" --output_dir "$RESULTS" --fig_dir "$FIGS" \
    --n_bootstrap 1000 2>&1

# UC2: Model Routing (enhanced: oracle router, break-even analysis)
echo ""
echo ">>> UC2: Model Routing ($(date))"
python scripts/uc2_model_routing.py \
    --scored_dir "$SCORED" --output_dir "$RESULTS" --fig_dir "$FIGS" 2>&1

# UC3: Error Discovery (MERGED UC3+UC9: binary detection + annotation efficiency)
echo ""
echo ">>> UC3: Error Discovery [merged UC3+UC9] ($(date))"
python scripts/uc3_error_discovery.py \
    --scored_dir "$SCORED" --output_dir "$RESULTS" --fig_dir "$FIGS" 2>&1

# UC4: Difficulty Estimation (enhanced: Kendall tau, Fisher z-test)
echo ""
echo ">>> UC4: Difficulty Estimation ($(date))"
python scripts/uc4_difficulty_estimation.py \
    --scored_dir "$SCORED" --output_dir "$RESULTS" --fig_dir "$FIGS" 2>&1

# UC5: Response Selection (MERGED UC5+UC-B: pairwise + best-of-N)
echo ""
echo ">>> UC5: Response Selection [merged UC5+UC-B] ($(date))"
python scripts/uc5_response_selection.py \
    --scored_dir "$SCORED" --output_dir "$RESULTS" --fig_dir "$FIGS" 2>&1

# UC6: Cascade Inference (enhanced: 2-tier cost curve, break-even)
echo ""
echo ">>> UC6: Cascade Inference ($(date))"
python scripts/uc6_cascade_inference.py \
    --scored_dir "$SCORED" --output_dir "$RESULTS" --fig_dir "$FIGS" 2>&1

# UC7: Self-Improvement (enhanced: simulated retry experiment)
echo ""
echo ">>> UC7: Self-Improvement ($(date))"
python scripts/uc7_self_improvement.py \
    --scored_dir "$SCORED" --output_dir "$RESULTS" --fig_dir "$FIGS" 2>&1

# UC8: OOD Detection (enhanced: bootstrap CIs, KS test, calibration drift)
echo ""
echo ">>> UC8: OOD Detection ($(date))"
python scripts/uc8_ood_detection.py \
    --scored_dir "$SCORED" --output_dir "$RESULTS" --fig_dir "$FIGS" 2>&1

# UC-A: DPO Reward (enhanced: bootstrap CIs, reward quality analysis)
echo ""
echo ">>> UC-A: DPO Reward ($(date))"
python scripts/uc_a_dpo_reward.py \
    --scored_dir "$SCORED" --output_dir "$RESULTS" --fig_dir "$FIGS" 2>&1

# UC-C: Data Curation (enhanced: bootstrap CIs, data efficiency ratio)
echo ""
echo ">>> UC-C: Data Curation ($(date))"
python scripts/uc_c_data_curation.py \
    --scored_dir "$SCORED" --output_dir "$RESULTS" --fig_dir "$FIGS" 2>&1

# UC-D: Agent Steps (HONEST NEGATIVE — step-level UQ near random)
echo ""
echo ">>> UC-D: Agent Steps [honest negative] ($(date))"
python scripts/uc_d_agent_steps.py \
    --scored_dir "$SCORED" --output_dir "$RESULTS" --fig_dir "$FIGS" 2>&1

echo ""
echo "============================================"
echo "ALL USE CASES COMPLETE — $(date)"
echo "============================================"
echo ""
echo "Results: $RESULTS/"
ls -la "$RESULTS"/uc*.json
echo ""
echo "Figures: $FIGS/"
ls -la "$FIGS"/uc*.pdf
