#!/usr/bin/env python3
"""
Test 235B-Instruct (not Thinking) model with TP=8.
The official vLLM docs show Instruct working, not Thinking.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from vllm import LLM, SamplingParams


def main():
    print("=" * 60)
    print("Qwen3-VL-235B-A22B-Instruct Test (TP=8)")
    print("=" * 60)
    print(f"Available GPUs: {torch.cuda.device_count()}")

    # Try Instruct model instead of Thinking
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
        enable_expert_parallel=True,
    )

    print("\nMODEL LOADED SUCCESSFULLY!")

    sampling_params = SamplingParams(temperature=0, max_tokens=64)
    outputs = llm.generate(["What is 2 + 2?"], sampling_params)
    print(f"Output: {outputs[0].outputs[0].text}")


if __name__ == "__main__":
    main()
