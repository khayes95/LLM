"""Test NCCL multi-GPU communication directly."""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size):
    try:
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        print(f"[rank {rank}] Initializing process group...")
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29500",
            world_size=world_size,
            rank=rank,
        )
        print(f"[rank {rank}] Process group initialized!")

        torch.cuda.set_device(rank)
        tensor = torch.ones(10, device=f"cuda:{rank}") * (rank + 1)
        print(f"[rank {rank}] Before all_reduce: {tensor[:3]}")

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"[rank {rank}] After all_reduce: {tensor[:3]}")

        dist.destroy_process_group()
        print(f"[rank {rank}] NCCL TEST PASSED")

    except Exception as e:
        print(f"[rank {rank}] FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    world_size = 2
    print(f"Testing NCCL with {world_size} GPUs")
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
    print("\nNCCL MULTI-GPU TEST PASSED")
