#!/usr/bin/env python3
"""Test 235B with expert parallelism for MoE."""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

from vllm import LLM, SamplingParams

print("Testing with expert parallelism...")
print("Model: Qwen/Qwen3-VL-235B-A22B-Thinking")
print("TP=4, EP=2 (4*2=8 GPUs)")

llm = LLM(
    model='Qwen/Qwen3-VL-235B-A22B-Thinking',
    dtype='bfloat16',
    tensor_parallel_size=4,  # TP=4
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    trust_remote_code=True,
    max_num_seqs=4,
    enforce_eager=True,
    distributed_executor_backend="mp",
    enable_expert_parallel=True,  # EP for MoE
)

print("Model loaded! Testing inference...")
outputs = llm.generate(["Hello, how are you?"], SamplingParams(max_tokens=32))
print(f"Output: {outputs[0].outputs[0].text}")
