#!/bin/bash
#SBATCH --job-name=uq_novel
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/overnight_novel_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/overnight_novel_%j.log

# Overnight GPU work: UC-D S2 (step truncation) then UC-C S2 (student training)
# Starts immediately on GPU 2, then uses more GPUs after size ablation finishes
# Expected: ~2-3h for UC-D, ~2-3h for UC-C, done well before 9 AM

set -e
cd /scratch/khayes/LLM

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate uq_eval

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Overnight Novel Use Cases — GPU Pipeline"
echo "Job: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# ============================================================
# PHASE 1: UC-D S2 — Step Truncation Scoring (GPU 2)
# Runs immediately in parallel with size ablation
# ============================================================
echo ""
echo "=========================================="
echo "PHASE 1: UC-D S2 — Step Truncation Scoring"
echo "Start: $(date)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=2
python scripts/uc_d_step_truncation.py \
    --checkpoint uq_models/best_unified \
    --output_dir data/use_cases/results_unified \
    --fig_dir figures/use_cases_unified

echo "UC-D S2 done: $(date)"

# ============================================================
# PHASE 2: UC-C S2 — Student Model Training (GPU 2)
# Runs after UC-D S2 finishes
# ============================================================
echo ""
echo "=========================================="
echo "PHASE 2: UC-C S2 — Student Model Training"
echo "Start: $(date)"
echo "=========================================="

# Check if more GPUs are free (size ablation may have finished)
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
export CUDA_VISIBLE_DEVICES=2

python scripts/uc_c_train_students.py \
    --epochs 2 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --output_dir data/use_cases/results_unified \
    --fig_dir figures/use_cases_unified \
    --model_dir uq_models/uc_c_students

echo "UC-C S2 done: $(date)"

# ============================================================
# DONE
# ============================================================
echo ""
echo "=========================================="
echo "All overnight novel UCs complete"
echo "Done: $(date)"
echo "=========================================="
