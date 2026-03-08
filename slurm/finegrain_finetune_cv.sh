#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --job-name=fg_ft_cv
#SBATCH --output=logs/fg_finetune_cv_%j.out

cd /scratch/khayes/LLM
export CUDA_VISIBLE_DEVICES=2,3

echo "=== Starting FineGRAIN 5-fold CV finetuning at $(date) ==="

PYTHON=/scratch/khayes/.conda/envs/uq_eval/bin/python
OUTPUT_BASE=data/finegrain_uq/exp1_human_cv

# 5-fold leave-one-model-out cross-validation
for MODEL in flux sd3.5_large sd3.5_medium sd3_m sd3_xl; do
    echo ""
    echo "=== Fold: held-out model = ${MODEL} ==="
    echo "=== Started at $(date) ==="

    $PYTHON scripts/finegrain_finetune.py \
        --experiment human_cv \
        --held_out_model "$MODEL" \
        --output_dir "${OUTPUT_BASE}" \
        --epochs 2 \
        --learning_rate 2e-5

    echo "=== Fold ${MODEL} finished at $(date) ==="
done

echo ""
echo "=== All 5 folds complete at $(date) ==="

# Summarize results
$PYTHON -c "
import json, os, glob
import numpy as np

base = '${OUTPUT_BASE}'
results = {}
for fold_dir in sorted(glob.glob(os.path.join(base, 'fold_*'))):
    res_path = os.path.join(fold_dir, 'results.json')
    if os.path.exists(res_path):
        with open(res_path) as f:
            r = json.load(f)
        model = os.path.basename(fold_dir).replace('fold_', '')
        results[model] = r
        print(f'{model}: AUROC={r.get(\"auroc\", \"N/A\")}')

if results:
    aurocs = [r['auroc'] for r in results.values() if 'auroc' in r]
    print(f'\\nMean AUROC across folds: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}')

    summary = {'folds': results, 'mean_auroc': float(np.mean(aurocs)), 'std_auroc': float(np.std(aurocs))}
    with open(os.path.join(base, 'cv_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary saved to {base}/cv_summary.json')
else:
    print('No results found!')
"
