#!/usr/bin/env python3
"""
Test 235B with verbose debugging to find the actual error.
"""
import os
import sys
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'
os.environ['VLLM_TRACE_FUNCTION'] = '1'

import logging
logging.basicConfig(level=logging.DEBUG)

import torch
from vllm import LLM, SamplingParams


def main():
    print("Testing Qwen3-VL-235B with verbose debugging...")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

    print("\nLoading model with verbose logging...")

    try:
        llm = LLM(
            model='Qwen/Qwen3-VL-235B-A22B-Thinking',
            dtype='bfloat16',
            tensor_parallel_size=8,
            gpu_memory_utilization=0.80,
            max_model_len=2048,
            trust_remote_code=True,
            enforce_eager=True,
            limit_mm_per_prompt={"video": 0, "image": 1},
            disable_custom_all_reduce=True,
        )
        print("Model loaded successfully!")

        sampling_params = SamplingParams(temperature=0, max_tokens=32)
        outputs = llm.generate(["What is 2 + 2?"], sampling_params)
        print(f"Output: {outputs[0].outputs[0].text}")

    except Exception as e:
        print(f"\n\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
