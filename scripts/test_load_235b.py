#!/usr/bin/env python3
"""Test loading Qwen3-VL-235B with verbose error output."""

import os
import sys
import traceback

# Enable debug output
os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'
os.environ['NCCL_DEBUG'] = 'INFO'

from vllm import LLM, SamplingParams

print("Loading model...")
try:
    llm = LLM(
        model='Qwen/Qwen3-VL-235B-A22B-Thinking',
        dtype='bfloat16',
        tensor_parallel_size=8,
        gpu_memory_utilization=0.8,
        max_model_len=4096,  # Very small
        trust_remote_code=True,
        max_num_seqs=1,  # Minimal
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},  # Limit multimodal
    )
    print("Model loaded successfully!")

    # Test simple generation
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    outputs = llm.generate(["Hello, world!"], sampling_params)
    print(f"Test output: {outputs[0].outputs[0].text}")

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1)
