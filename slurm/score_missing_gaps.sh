#!/bin/bash
#SBATCH --job-name=score_gaps
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/score_gaps_%j.log
#SBATCH --error=logs/score_gaps_%j.log

# Score 2 missing heatmap gaps + regenerate breakdown + heatmap
# Smoke test: SMOKE=1 sbatch slurm/score_missing_gaps.sh

cd /scratch/khayes/LLM
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=4

EXTRA=""
if [[ -n "$SMOKE" ]]; then
    EXTRA="--smoke_test"
    echo "*** SMOKE TEST ***"
fi

echo "Step 1: Score missing predictions..."
python scripts/score_heatmap_gaps.py $EXTRA

echo ""
echo "Step 2: Regenerate per-benchmark breakdown..."
python scripts/per_benchmark_breakdown.py \
    --scored_dir data/use_cases/scored_test_only_v2 \
    --output data/use_cases/results_test_only_v2/per_benchmark_breakdown.json

echo ""
echo "Step 3: Regenerate heatmap figure..."
python scripts/cpu_generate_all_figures.py \
    --figures fig_per_benchmark_heatmap \
    --scored_dir data/use_cases/scored_test_only_v2 \
    --per_benchmark_breakdown data/use_cases/results_test_only_v2/per_benchmark_breakdown.json \
    --fig_dir figures/paper

echo "Done: $(date)"
