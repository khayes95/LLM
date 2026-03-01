"""Test multi-GPU communication: gloo and nccl backends."""
import os
import sys
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_NET"] = "Socket"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def test_gloo(rank, world_size):
    """Test with gloo backend (CPU-based, always works)."""
    try:
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29501",
            world_size=world_size,
            rank=rank,
        )
        tensor = torch.ones(10) * (rank + 1)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"[gloo rank {rank}] all_reduce result: {tensor[0].item()}")
        dist.destroy_process_group()
        print(f"[gloo rank {rank}] PASSED")
    except Exception as e:
        print(f"[gloo rank {rank}] FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


def test_nccl_manual(rank, world_size):
    """Test NCCL with manual cuda device set."""
    try:
        torch.cuda.set_device(rank)
        print(f"[nccl rank {rank}] CUDA device set to {rank}")
        print(f"[nccl rank {rank}] Testing basic CUDA op...")
        x = torch.randn(100, device=f"cuda:{rank}")
        print(f"[nccl rank {rank}] CUDA tensor created OK")

        print(f"[nccl rank {rank}] Initializing NCCL process group...")
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29502",
            world_size=world_size,
            rank=rank,
        )
        print(f"[nccl rank {rank}] NCCL process group OK!")

        tensor = torch.ones(10, device=f"cuda:{rank}") * (rank + 1)
        print(f"[nccl rank {rank}] Before all_reduce: {tensor[0].item()}")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"[nccl rank {rank}] After all_reduce: {tensor[0].item()}")

        dist.destroy_process_group()
        print(f"[nccl rank {rank}] PASSED")
    except Exception as e:
        print(f"[nccl rank {rank}] FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 50)
    print("Test 1: Gloo backend (CPU)")
    print("=" * 50)
    mp.spawn(test_gloo, args=(2,), nprocs=2, join=True)
    print("GLOO PASSED\n")

    print("=" * 50)
    print("Test 2: NCCL backend (GPU)")
    print("=" * 50)
    mp.spawn(test_nccl_manual, args=(2,), nprocs=2, join=True)
    print("NCCL PASSED")
