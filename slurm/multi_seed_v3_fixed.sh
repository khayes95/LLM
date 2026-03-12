#!/bin/bash
# Multi-seed training for v3 (question-level split), one SLURM job per seed.
#
# DIAGNOSIS OF OOM (jobs 8741, 8810, 8811):
#
#   Root cause: CUDA context corruption from sequential subprocess runs.
#   multi_seed_training.py runs retrain_best_v2.py sequentially via subprocess.run()
#   in a SINGLE SLURM job. The first seed (42) trains successfully using
#   device_map="auto" across 4 GPUs (~75.77 GiB consumed). When that subprocess
#   exits, the parent Python process (multi_seed_training.py) retains the leaked
#   CUDA context — there is NO cleanup (no del model, no torch.cuda.empty_cache(),
#   no gc.collect() anywhere in retrain_best_v2.py).
#
#   When seed 123 starts, it tries to load a fresh Qwen3-VL-8B model, but
#   GPU 0 has only 126 MiB free (of 79 GiB). Every batch fails with OOM.
#   Additionally, device_map="auto" triggers a CUDA internal assert:
#     "device >= 0 && device < num_gpus INTERNAL ASSERT FAILED, device=4, num_gpus=4"
#   because the stale context confuses PyTorch's device indexing.
#
#   Result: seed 42 succeeded (AUROC=0.889), seeds 123/456/789/314 all FAILED.
#   Jobs 8810 and 8811 were resubmissions that hit the exact same issue.
#
# FIX: Submit each seed as a SEPARATE SLURM job with its own clean CUDA context.
#   Uses retrain_best_v2.py which correctly:
#     - Accepts --seed and uses it for train_test_split (random_state=args.seed)
#     - Has the question-level split fix (splits by question ID, not sample index)
#     - Uses r=32, alpha=64, combined prompt (same config as successful v3 job 8527)
#
# Usage:
#   bash slurm/multi_seed_v3_fixed.sh              # submit all 4 remaining seeds
#   bash slurm/multi_seed_v3_fixed.sh 123 456       # submit specific seeds
#   bash slurm/multi_seed_v3_fixed.sh 42 123 456 789 314  # all 5 (re-runs seed 42 too)

set -e
cd /scratch/khayes/LLM

# Default: only the 4 seeds that failed (seed 42 already succeeded in job 8741)
SEEDS="${@:-123 456 789 314}"
OUTPUT_BASE="data/ablations/multi_seed_v3"

mkdir -p "$OUTPUT_BASE" logs

echo "=== Multi-Seed v3 Training (separate SLURM jobs) ==="
echo "Seeds: $SEEDS"
echo "Output: $OUTPUT_BASE"
echo "Config: r=32, alpha=64, lr=1e-4, 3 epochs, 4x A100, ~4-5h each"
echo ""

for SEED in $SEEDS; do
    # Skip seeds with existing results (delete results.json to force re-run)
    if [ -f "$OUTPUT_BASE/seed_${SEED}/results.json" ]; then
        echo "Seed $SEED: results.json exists — skipping (delete to re-run)"
        continue
    fi

    SCRIPT=$(mktemp /tmp/mseed_v3_${SEED}_XXXX.sh)
    cat > "$SCRIPT" <<SLURM_EOF
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --job-name=msv3_s${SEED}
#SBATCH --output=logs/multi_seed_v3_s${SEED}_%j.out

# Smoke test (uncomment to test before full run):
# CUDA_VISIBLE_DEVICES=0 python scripts/retrain_best_v2.py \\
#   --output_dir ${OUTPUT_BASE}/seed_${SEED} --smoke_test --seed ${SEED} \\
#   --lora_r 32 --lora_alpha 64 --learning_rate 1e-4

cd /scratch/khayes/LLM
source activate uq_eval

echo "=== Multi-Seed v3: Seed ${SEED} ==="
echo "Start: \$(date)"
echo "GPUs: \$(nvidia-smi -L | wc -l)x A100"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/retrain_best_v2.py \\
    --output_dir ${OUTPUT_BASE}/seed_${SEED} \\
    --epochs 3 \\
    --lora_r 32 \\
    --lora_alpha 64 \\
    --learning_rate 1e-4 \\
    --seed ${SEED}

echo "Done: \$(date)"

# Print results
if [ -f "${OUTPUT_BASE}/seed_${SEED}/results.json" ]; then
    echo "=== Results ==="
    python -m json.tool "${OUTPUT_BASE}/seed_${SEED}/results.json" | head -20
else
    echo "ERROR: No results.json produced"
    exit 1
fi
SLURM_EOF

    JOB_ID=$(sbatch "$SCRIPT" | awk '{print $4}')
    echo "Seed $SEED: submitted as job $JOB_ID"
    rm -f "$SCRIPT"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo ""
echo "After all seeds complete, aggregate results with:"
echo "  python3 -c \""
echo "import json, numpy as np, glob"
echo "results = []"
echo "for f in sorted(glob.glob('${OUTPUT_BASE}/seed_*/results.json')):"
echo "    with open(f) as fh: r = json.load(fh)"
echo "    seed = f.split('seed_')[1].split('/')[0]"
echo "    auroc = r.get('auroc', r.get('overall_auroc'))"
echo "    results.append((seed, auroc))"
echo "    print(f'  Seed {seed}: AUROC={auroc:.4f}')"
echo "aurocs = [r[1] for r in results if r[1] is not None]"
echo "print(f'\\nMean: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f} (n={len(aurocs)})')"
echo "\""
