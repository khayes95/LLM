#!/usr/bin/env python3
"""Test 235B with Ray distributed executor."""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams

print("Testing Ray distributed executor...")
llm = LLM(
    model='Qwen/Qwen3-VL-235B-A22B-Thinking',
    dtype='bfloat16',
    tensor_parallel_size=8,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    trust_remote_code=True,
    max_num_seqs=4,
    enforce_eager=True,
    distributed_executor_backend="ray",
)

print("Model loaded! Testing inference...")
outputs = llm.generate(["Hello, how are you?"], SamplingParams(max_tokens=32))
print(f"Output: {outputs[0].outputs[0].text}")
