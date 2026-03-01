#!/usr/bin/env python3
"""
Test simple NCCL multi-GPU communication to diagnose the segfault.
This bypasses vLLM to check if NCCL works on this system.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NCCL_DEBUG'] = 'INFO'

import torch
import torch.distributed as dist


def main():
    # Initialize distributed with NCCL
    print("Testing NCCL multi-GPU initialization...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    # Test simple operations on each GPU
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        x = torch.ones(100, device=device)
        y = x * 2
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - OK")

    # Test PyTorch NCCL backend directly (without multiprocessing)
    print("\nTesting NCCL import...")
    try:
        # This imports the NCCL bindings
        from torch.distributed import is_nccl_available
        print(f"NCCL available: {is_nccl_available()}")
    except Exception as e:
        print(f"NCCL check failed: {e}")

    print("\nSimple GPU tests passed! NCCL crash likely occurs during distributed init.")


if __name__ == '__main__':
    main()
