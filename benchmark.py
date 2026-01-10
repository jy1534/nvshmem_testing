# benchmark.py
import os
import time
import subprocess

import torch
import torch.distributed as dist

from utils import gen_tensor
from alltoall_persistent_gemm import run as CustomA2A
from baseline import run as TorchA2A


def _env_int(*names, default=None):
    for n in names:
        v = os.getenv(n, None)
        if v is not None:
            return int(v)
    return default


def infer_rank_info():
    # Works for torchrun and for srun.
    rank = _env_int("RANK", "SLURM_PROCID", default=0)
    world_size = _env_int("WORLD_SIZE", "SLURM_NTASKS", default=1)
    local_rank = _env_int("LOCAL_RANK", "SLURM_LOCALID", default=0)
    return rank, world_size, local_rank


def preflight(rank):
    print(f"[rank {rank}] HOST={os.uname().nodename}")
    print(f"[rank {rank}] CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"[rank {rank}] LD_LIBRARY_PATH={os.getenv('LD_LIBRARY_PATH')}")
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        print(f"[rank {rank}] nvidia-smi -L:\n{out}")
    except Exception as e:
        print(f"[rank {rank}] nvidia-smi unavailable: {e}")
    print(f"[rank {rank}] torch={torch.__version__}")
    print(f"[rank {rank}] torch.cuda.is_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[rank {rank}] torch.version.cuda={torch.version.cuda}")
        print(f"[rank {rank}] device_count={torch.cuda.device_count()}")


def main():
    rank, world_size, local_rank = infer_rank_info()
    preflight(rank)

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"[rank {rank}] torch.cuda.is_available is False. "
            "Common causes: wrong container/env mixing, CUDA compat libs in LD_LIBRARY_PATH, or GPU not allocated."
        )

    # Decide CUDA device index robustly under srun 1 task = 1 GPU
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    vis_list = [x.strip() for x in visible.split(",") if x.strip() != ""]
    nvis = len(vis_list)

    if nvis == 1:
        device_index = 0
    else:
        device_index = local_rank

    print(f"[rank {rank}] local_rank={local_rank} CUDA_VISIBLE_DEVICES={visible} -> set_device({device_index})")
    torch.cuda.set_device(device_index)



    # Make env:// happy (works for srun and for plain python)
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", os.getenv("MASTER_ADDR", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", os.getenv("MASTER_PORT", "29500"))

    # env:// expects MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
    dist.init_process_group(backend="nccl", init_method="env://")

    # ---- Config (keep your original defaults) ----
    batch, seq, hidden_dim, topk = 4, 1024, 768, 2
    num_experts = world_size * 32
    run_custom_a2a = (os.getenv("RUN_CUSTOM", "0") == "1")

    tokens, chunk_size = gen_tensor(batch, seq, hidden_dim, world_size, num_experts, rank, topk)
    if rank == 0:
        print(f"[rank {rank}] world_size={world_size} chunk_size={chunk_size} tokens={tuple(tokens.shape)}")

    # correctness check
    if rank == 0:
        print(">> Running correctness verification...")

    #  one Baseline (NCCL)
     
    baseline_out = TorchA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, False, None)
    
    #  OneCustom (SymmMem)
    # barrier for clearence
    dist.barrier()
    custom_out = CustomA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, None, False)
    dist.barrier()

    # comaparison
    
    match = torch.allclose(baseline_out, custom_out, rtol=0, atol=0) 
    
    if match:
        if rank == 0:
            print(f"[rank {rank}] PASS: Custom SymmMem output matches PyTorch Baseline!")
    else:
        # diff count
        diff = (baseline_out - custom_out).abs().max()
        mismatch_cnt = (baseline_out != custom_out).sum()
        print(f"[rank {rank}]  FAIL: Output mismatch! Max diff: {diff}, Mismatch elements: {mismatch_cnt}")
       
        # raise RuntimeError("Correctness check failed!")

    # release
    del baseline_out, custom_out
    torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        print(">> Verification done. Starting benchmark loops...")
    
    # Warmup 
    for _ in range(5):
        if run_custom_a2a:
            CustomA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, None, False)
        else:
            TorchA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, False, None)
        dist.barrier()
    torch.cuda.synchronize()

    #  Timing (simple first pass) 
    start = time.time()
    for _ in range(10):
        if run_custom_a2a:
            CustomA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, None, False)
        else:
            TorchA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, False, None)
    dist.barrier()
    torch.cuda.synchronize()
    end = time.time()

    print(f"[rank {rank}] mode={'custom(symm_mem)' if run_custom_a2a else 'torch(all_to_all_single)'} "
          f"time={(end - start):.6f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
