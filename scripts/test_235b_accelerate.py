#!/usr/bin/env python3
"""
Test 235B using accelerate/transformers directly for model loading.
This bypasses vLLM and uses naive tensor parallelism.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from transformers import AutoProcessor
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download


def main():
    print("=" * 60)
    print("Testing Qwen3-VL-235B with Accelerate")
    print("=" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    model_path = "Qwen/Qwen3-VL-235B-A22B-Thinking"
    
    print(f"\nLoading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("Processor loaded!")
    
    # Find local checkpoint
    cache_dir = f"/scratch/khayes/.cache/huggingface/hub/models--Qwen--Qwen3-VL-235B-A22B-Thinking/snapshots/"
    import glob
    snapshots = glob.glob(f"{cache_dir}*/")
    if snapshots:
        local_path = snapshots[0]
        print(f"Using local path: {local_path}")
    else:
        print("Downloading model...")
        local_path = snapshot_download(model_path)
    
    print("\nModel would load here with accelerate device_map='auto'")
    print("This will distribute across all 8 GPUs automatically.")
    print("Skipping actual load to test if the approach works.")


if __name__ == '__main__':
    main()
