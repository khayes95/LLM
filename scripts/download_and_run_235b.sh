#!/bin/bash
# Download Qwen3-VL-235B and run full evaluation after download completes

set -e

cd /scratch/khayes/LLM

echo "=== Downloading Qwen3-VL-235B-A22B-Thinking ==="
echo "Started at: $(date)"

# Download model with 16 parallel workers
/scratch/khayes/anaconda3/envs/finegrain_vlm/bin/python -c "
from huggingface_hub import snapshot_download
print('Downloading Qwen3-VL-235B-A22B-Thinking with 16 workers...')
snapshot_download(
    'Qwen/Qwen3-VL-235B-A22B-Thinking',
    max_workers=16,
    resume_download=True,
)
print('Download complete!')
"

echo ""
echo "=== Download complete at: $(date) ==="
echo ""
echo "=== Starting full evaluation ==="

# Run the full evaluation on 8 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /scratch/khayes/anaconda3/envs/finegrain_vlm/bin/python \
    scripts/run_qwen3_vl_235b_full.py \
    --model Qwen/Qwen3-VL-235B-A22B-Thinking \
    --tensor_parallel_size 8 \
    2>&1 | tee logs/qwen3_vl_235b_full_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=== Finished at: $(date) ==="
