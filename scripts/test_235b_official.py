#!/usr/bin/env python3
"""
Test 235B with official vLLM configuration for A100.

Based on official vLLM docs:
- Use mm_encoder_tp_mode="data" for data-parallel vision encoder
- Use enable_expert_parallel=True for MoE models
- Use limit_mm_per_prompt for image-only mode
- Use async_scheduling for better performance
"""
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['OMP_NUM_THREADS'] = '1'  # Avoid CPU contention

import torch
from vllm import LLM, SamplingParams


def main():
    print("Testing Qwen3-VL-235B with official A100 configuration...")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    llm = LLM(
        model='Qwen/Qwen3-VL-235B-A22B-Thinking',
        dtype='bfloat16',
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.85,
        max_model_len=4096,  # Reduced for A100
        trust_remote_code=True,
        mm_encoder_tp_mode="data",  # Data-parallel vision encoder
        enable_expert_parallel=True,  # For MoE models
        limit_mm_per_prompt={"video": 0},  # Image-only mode
        enforce_eager=True,
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
