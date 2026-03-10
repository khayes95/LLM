#!/bin/bash
#SBATCH --job-name=size_abl
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/size_ablation_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/size_ablation_%j.log

# Training data size ablation: how many samples needed for good AUROC?
# Uses v2 config (r=32, alpha=64, combined prompt) on GPUs 2-5
#
# Sizes: 100, 250, 500, 1000, 2000, 5000, full (~10047)
# Each run: 3 epochs, 4 GPUs
#
# Estimated time: ~5-6 hours total

set -e
cd /scratch/khayes/LLM

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=0,2,4,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "============================================"
echo "TRAINING SIZE ABLATION"
echo "Job: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "============================================"

SIZES=(100 250 500 1000 2000 5000)
BASE_DIR="uq_models/size_ablation"
RESULTS_DIR="data/ablations/training_size"
mkdir -p "$RESULTS_DIR"

# Run each size
for N in "${SIZES[@]}"; do
    echo ""
    echo ">>> Training with N=$N samples ($(date))"
    OUTPUT="$BASE_DIR/n_${N}"

    python scripts/train_best_uq.py \
        --output_dir "$OUTPUT" \
        --max_train_samples $N \
        --epochs 3 \
        --lora_r 32 \
        --lora_alpha 64 \
        --learning_rate 1e-4 \
        --prompt_variant combined \
        --seed 42 \
        2>&1

    echo "--- N=$N done ($(date)) ---"

    # Copy results
    if [ -f "$OUTPUT/results.json" ]; then
        cp "$OUTPUT/results.json" "$RESULTS_DIR/results_n${N}.json"
    fi
done

# Full training (no --max_train_samples)
echo ""
echo ">>> Training with FULL dataset ($(date))"
OUTPUT="$BASE_DIR/n_full"

python scripts/train_best_uq.py \
    --output_dir "$OUTPUT" \
    --epochs 3 \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 1e-4 \
    --prompt_variant combined \
    --seed 42 \
    2>&1

echo "--- Full done ($(date)) ---"

if [ -f "$OUTPUT/results.json" ]; then
    cp "$OUTPUT/results.json" "$RESULTS_DIR/results_n_full.json"
fi

# Aggregate results
echo ""
echo ">>> Aggregating results..."
python3 -c "
import json, glob, os
results = {}
for f in sorted(glob.glob('$RESULTS_DIR/results_n*.json')):
    name = os.path.basename(f).replace('results_', '').replace('.json', '')
    with open(f) as fh:
        d = json.load(fh)
    n = name.replace('n', '').replace('_', '')
    results[name] = {
        'auroc': d.get('auroc'),
        'brier': d.get('brier'),
        'vlm_auroc': d.get('vlm_auroc'),
        'text_auroc': d.get('text_auroc'),
        'n_train': d.get('n_samples'),
    }
    print(f'  {name}: AUROC={d.get(\"auroc\", \"?\"):.4f}')

with open('$RESULTS_DIR/summary.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved summary to $RESULTS_DIR/summary.json')
"

echo ""
echo "============================================"
echo "TRAINING SIZE ABLATION COMPLETE — $(date)"
echo "============================================"
echo ""
echo "Outputs: $RESULTS_DIR/"
echo "Checkpoints: $BASE_DIR/"
