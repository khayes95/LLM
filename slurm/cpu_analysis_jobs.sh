#!/bin/bash
# =============================================================================
# CPU Analysis Jobs — Submit all CPU-intensive analysis to debug partition
#
# Usage:
#   bash slurm/cpu_analysis_jobs.sh           # Submit all jobs
#   bash slurm/cpu_analysis_jobs.sh --smoke   # Submit smoke tests only
# =============================================================================

set -e

SMOKE=""
if [[ "$1" == "--smoke" ]]; then
    SMOKE="--smoke_test"
    echo "=== SMOKE TEST MODE ==="
fi

LOGDIR="/scratch/khayes/LLM/logs"
PYTHON="/scratch/khayes/.conda/envs/uq_eval/bin/python"
WORKDIR="/scratch/khayes/LLM"

mkdir -p "$LOGDIR"

echo "Submitting CPU analysis jobs to debug partition..."
echo "Python: $PYTHON"
echo ""

# --- Job 1: Exhaustive Bootstrap (100K iterations, BCa) ---
JOB1=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name=bootstrap
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=${LOGDIR}/bootstrap_%j.out

cd ${WORKDIR}

echo "=== Exhaustive Bootstrap (100K iterations, BCa) ==="
echo "Start: \$(date)"
echo "CPUs: \$SLURM_CPUS_PER_TASK"
echo "Node: \$(hostname)"

${PYTHON} scripts/cpu_exhaustive_bootstrap.py \
    --scored_dir data/use_cases/scored_test_only \
    --output data/use_cases/results_test_only/exhaustive_bootstrap.json \
    $SMOKE

echo "End: \$(date)"
EOF
)
echo "  [Job $JOB1] Exhaustive Bootstrap"

# --- Job 2: Contamination Check ---
JOB2=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name=contam
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=${LOGDIR}/contamination_%j.out

cd ${WORKDIR}

echo "=== Contamination & Deduplication Check ==="
echo "Start: \$(date)"
echo "CPUs: \$SLURM_CPUS_PER_TASK"
echo "Node: \$(hostname)"

${PYTHON} scripts/cpu_contamination_check.py \
    --train_file data/finetune/train_v2.jsonl \
    --test_file data/finetune/test_v2.jsonl \
    --scored_dir data/use_cases/scored_test_only \
    --output data/use_cases/results_test_only/contamination_report.json \
    $SMOKE

echo "End: \$(date)"
EOF
)
echo "  [Job $JOB2] Contamination Check"

# --- Job 3: Feature Analysis ---
JOB3=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name=features
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=${LOGDIR}/features_%j.out

cd ${WORKDIR}

echo "=== Feature Extraction & Error Analysis ==="
echo "Start: \$(date)"
echo "CPUs: \$SLURM_CPUS_PER_TASK"
echo "Node: \$(hostname)"

${PYTHON} scripts/cpu_feature_analysis.py \
    --scored_dir data/use_cases/scored_test_only \
    --output data/use_cases/results_test_only/feature_analysis.json \
    $SMOKE

echo "End: \$(date)"
EOF
)
echo "  [Job $JOB3] Feature Analysis"

# --- Job 4: Prompt Perturbation Generation ---
JOB4=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name=perturb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=${LOGDIR}/perturbation_%j.out

cd ${WORKDIR}

echo "=== Prompt Perturbation Generation ==="
echo "Start: \$(date)"
echo "CPUs: \$SLURM_CPUS_PER_TASK"
echo "Node: \$(hostname)"

${PYTHON} scripts/cpu_prompt_perturbation.py \
    --scored_dir data/use_cases/scored_test_only \
    --output data/use_cases/perturbations/all_perturbations.jsonl \
    $SMOKE

echo "End: \$(date)"
EOF
)
echo "  [Job $JOB4] Prompt Perturbation"

# --- Job 5: Figure Generation ---
JOB5=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name=figures
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=${LOGDIR}/figures_%j.out

cd ${WORKDIR}

echo "=== Paper Figure Generation ==="
echo "Start: \$(date)"
echo "CPUs: \$SLURM_CPUS_PER_TASK"
echo "Node: \$(hostname)"

${PYTHON} scripts/cpu_generate_all_figures.py \
    --fig_dir figures/paper \
    $SMOKE

echo "End: \$(date)"
EOF
)
echo "  [Job $JOB5] Figure Generation"

echo ""
echo "All 5 jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOGDIR}/bootstrap_*.out"
echo ""
echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4, $JOB5"
