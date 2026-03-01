#!/usr/bin/env python3
"""
Test 235B with V0 (legacy) engine which is more stable.
"""
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VLLM_USE_V1'] = '0'  # Force V0 engine

import torch
from vllm import LLM, SamplingParams


def main():
    print("Testing Qwen3-VL-235B with V0 (legacy) engine...")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    llm = LLM(
        model='Qwen/Qwen3-VL-235B-A22B-Thinking',
        dtype='bfloat16',
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
        enforce_eager=True,
        limit_mm_per_prompt={"video": 0, "image": 8},
    )

    print("Model loaded successfully!")

    # Test with a simple prompt
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=64,
    )

    outputs = llm.generate(["What is 2 + 2?"], sampling_params)
    print(f"Output: {outputs[0].outputs[0].text}")


if __name__ == '__main__':
    main()
