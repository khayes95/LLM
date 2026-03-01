#!/usr/bin/env python3
"""
Test 235B using Ray distributed executor instead of multiprocessing.
Ray handles process spawning differently and may avoid the vLLM V1 bug.
"""
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from vllm import LLM, SamplingParams


if __name__ == '__main__':
    print("Testing Qwen3-VL-235B with Ray distributed executor...")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    # Use our downloaded Thinking model
    model_path = "Qwen/Qwen3-VL-235B-A22B-Thinking"

    print(f"Loading {model_path}...")
    llm = LLM(
        model=model_path,
        dtype='bfloat16',
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=True,
        distributed_executor_backend="ray",  # Use Ray instead of MP
        limit_mm_per_prompt={"video": 0},
    )

    print("Model loaded successfully!")

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=64,
    )

    outputs = llm.generate(["What is 2 + 2?"], sampling_params)
    print(f"Output: {outputs[0].outputs[0].text}")
