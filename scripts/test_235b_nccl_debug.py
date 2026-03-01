#!/usr/bin/env python3
"""
Test 235B with NCCL debug flags to diagnose the segfault.
"""
import os

# NCCL debug flags
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['NCCL_DEBUG_SUBSYS'] = 'INIT,NET'
os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand
os.environ['NCCL_P2P_LEVEL'] = 'NVL'  # Use NVLink for P2P

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from vllm import LLM, SamplingParams


if __name__ == '__main__':
    print("Testing Qwen3-VL-235B with NCCL debugging...")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

    # Use our downloaded Thinking model
    model_path = "Qwen/Qwen3-VL-235B-A22B-Thinking"

    print(f"Loading {model_path}...")
    llm = LLM(
        model=model_path,
        dtype='bfloat16',
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.80,
        max_model_len=2048,
        trust_remote_code=True,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=True,
        enforce_eager=True,
        limit_mm_per_prompt={"video": 0, "image": 1},
    )

    print("Model loaded successfully!")

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=64,
    )

    outputs = llm.generate(["What is 2 + 2?"], sampling_params)
    print(f"Output: {outputs[0].outputs[0].text}")
