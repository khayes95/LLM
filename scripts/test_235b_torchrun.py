#!/usr/bin/env python3
"""Test 235B using torchrun for distributed."""
import os
import torch
import torch.distributed as dist

# Initialize distributed
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

from vllm import LLM, SamplingParams

print(f"Rank {local_rank}: Testing with torchrun...")
if local_rank == 0:
    llm = LLM(
        model='Qwen/Qwen3-VL-235B-A22B-Thinking',
        dtype='bfloat16',
        tensor_parallel_size=8,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
        max_num_seqs=4,
        enforce_eager=True,
    )

    print("Model loaded! Testing inference...")
    outputs = llm.generate(["Hello, how are you?"], SamplingParams(max_tokens=32))
    print(f"Output: {outputs[0].outputs[0].text}")
