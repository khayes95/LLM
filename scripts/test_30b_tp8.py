#!/usr/bin/env python3
"""
Test the 30B model with TP=8 to see if the issue is specific to 235B or multi-GPU in general.
"""
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from vllm import LLM, SamplingParams


if __name__ == '__main__':
    print("Testing Qwen3-VL-30B with TP=8...")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    model_path = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    print(f"Loading {model_path} with TP=8...")
    try:
        llm = LLM(
            model=model_path,
            dtype='bfloat16',
            tensor_parallel_size=8,
            gpu_memory_utilization=0.50,  # Low since model is small
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True,
        )

        print("\nMODEL LOADED SUCCESSFULLY WITH TP=8!")

        sampling_params = SamplingParams(temperature=0, max_tokens=64)
        outputs = llm.generate(["What is 2 + 2?"], sampling_params)
        print(f"Output: {outputs[0].outputs[0].text}")

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
