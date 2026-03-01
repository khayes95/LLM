#!/bin/bash
#SBATCH --job-name=uq_analysis
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1M
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/khayes/LLM/logs/uq_analysis_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/uq_analysis_%j.log

# CPU-only analysis and figure generation
# Runs on debug partition — no GPU needed

set -e

echo "=========================================="
echo "UQ Analysis + Figures"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

# Activate conda
eval "$(conda shell.bash hook)"
conda activate uq_eval

which python
python --version

mkdir -p figures/paper figures/analysis

echo ""
echo "Step 1: Aggregate analysis..."
python scripts/analysis_aggregate.py --output_dir figures/analysis
RC1=$?
echo "Analysis exit code: $RC1"

echo ""
echo "Step 2: Paper figures..."
python scripts/generate_paper_figures.py --output_dir figures/paper
RC2=$?
echo "Figures exit code: $RC2"

echo ""
echo "=========================================="
echo "SUMMARY"
echo "  Analysis: $RC1"
echo "  Figures: $RC2"
echo "=========================================="

echo ""
echo "Generated files:"
ls -la figures/paper/*.pdf figures/analysis/*.json 2>/dev/null

echo "Done: $(date)"
