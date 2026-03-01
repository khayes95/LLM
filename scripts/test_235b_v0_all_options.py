#!/usr/bin/env python3
"""
Test 235B with V0 engine and ALL known workarounds combined.
"""
import os
# CRITICAL: Set these BEFORE any torch/vllm imports
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['VLLM_USE_V1'] = '0'  # Force V0 engine
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P which can cause issues
os.environ['NCCL_IB_DISABLE'] = '1'   # Disable InfiniBand
os.environ['NCCL_SHM_DISABLE'] = '0'  # Keep shared memory

import torch
from vllm import LLM, SamplingParams


def main():
    print("=" * 60)
    print("Qwen3-VL-235B Test with V0 Engine + All Workarounds")
    print("=" * 60)
    print(f"VLLM_USE_V1: {os.environ.get('VLLM_USE_V1')}")
    print(f"NCCL_P2P_DISABLE: {os.environ.get('NCCL_P2P_DISABLE')}")
    print(f"NCCL_IB_DISABLE: {os.environ.get('NCCL_IB_DISABLE')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    model_path = "Qwen/Qwen3-VL-235B-A22B-Instruct"

    print(f"\nLoading {model_path} with TP=8...")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=8,
        max_model_len=4096,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_num_seqs=4,
        enforce_eager=True,
        disable_custom_all_reduce=True,  # Use standard NCCL
    )

    print("\nMODEL LOADED SUCCESSFULLY!")

    sampling_params = SamplingParams(temperature=0, max_tokens=64)
    outputs = llm.generate(["What is 2 + 2?"], sampling_params)
    print(f"Output: {outputs[0].outputs[0].text}")


if __name__ == "__main__":
    main()
