#!/usr/bin/env python3
"""
Test loading 235B model weights manually with direct Torch distributed.
This bypasses vLLM's multiprocessing layer to diagnose where the issue is.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def main():
    print("Testing Qwen3-VL-235B model weight loading...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    model_path = "Qwen/Qwen3-VL-235B-A22B-Thinking"

    print(f"\nLoading config from {model_path}...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"Model type: {config.model_type}")
    print(f"Config keys: {list(config.to_dict().keys())[:10]}...")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Tokenizer loaded, vocab size: {len(tokenizer)}")

    # Try loading model on first GPU only
    print("\nAttempting to load model on single GPU (this will likely OOM but shows weight loading works)...")
    try:
        # Use device_map="auto" for distributed loading
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("Model loaded successfully with device_map='auto'!")
        print(f"Model device: {model.device}")
    except Exception as e:
        print(f"Model loading failed (expected for single GPU): {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()
