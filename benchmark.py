import os
import time
import subprocess

import torch
import torch.distributed as dist

from utils import gen_tensor, gen_weights
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
    #nvis = len(vis_list)

    #if nvis == 1:
    #    device_index = 0
    #else:
    #    device_index = local_rank

    #print(f"[rank {rank}] local_rank={local_rank} CUDA_VISIBLE_DEVICES={visible} -> set_device({device_index})")
    #torch.cuda.set_device(device_index)

    device_index = 0 if len(vis_list) == 1 else local_rank
    torch.cuda.set_device(device_index)

    # Make env:// happy (works for srun and for plain python)
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", os.getenv("MASTER_ADDR", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", os.getenv("MASTER_PORT", "29500"))

    # env:// expects MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
    dist.init_process_group(backend="nccl", init_method="env://")

    # Config
    batch, seq, hidden_dim, topk = 4, 1024, 768, 2
    out_dim = hidden_dim # Assume output dim equals input dim
    num_experts = world_size * 32
    run_custom_a2a = (os.getenv("RUN_CUSTOM", "0") == "1")

    # Generate Tokens
    # sent_lengths: list of int, length = num_experts. 
    # Token count sent from Rank i to Expert j
    tokens, chunk_size, sent_lengths = gen_tensor(batch, seq, hidden_dim, world_size, num_experts, rank, topk)
    
    # Prepare Counts
    num_local_experts = num_experts // world_size
    sent_counts_tensor = torch.tensor(sent_lengths, dtype=torch.int32, device="cuda")

    # Reshape logic: Expert e belongs to Rank (e // NumLocalExperts).
    # sent_counts_tensor is already ordered by destination rank.
    # reshape to [WorldSize, NumLocalExperts].
    sent_counts_reshaped = sent_counts_tensor.view(world_size, num_local_experts)

    # random for correctness check
    #torch.manual_seed(rank + 10086) 
    #local_weights = gen_weights(num_local_experts, hidden_dim, out_dim, dtype=tokens.dtype)
    
    
    #if rank == 0:
        #print(f"[rank {rank}] world_size={world_size} chunk_size={chunk_size} tokens={tuple(tokens.shape)}")
        #print(f"[rank {rank}] local_weights={tuple(local_weights.shape)}")


    # Prepare to recv buffer
    recv_counts = torch.empty_like(sent_counts_reshaped)
    
    # swap counts via All-to-All
    # recv_counts[j, k] denotes the number of tokens sent from Rank j to my k-th local expert
    dist.all_to_all_single(recv_counts, sent_counts_reshaped)
    
    # Generate Weights
    torch.manual_seed(rank + 10086)
    local_weights = gen_weights(num_local_experts, hidden_dim, out_dim, dtype=tokens.dtype)


    # Correctness Check
    if rank == 0:
        print(">> Running correctness verification (Comm + GEMM)...")

    # Baseline
    baseline_out = TorchA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, False, None, local_weights)
    
    dist.barrier()
    
    # Custom
    #custom_out = CustomA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, None, False, local_weights)
    # recv_counts for kernel to know the data range
    custom_out = CustomA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, None, False, local_weights, 
                           recv_counts=recv_counts)

    dist.barrier()

    # comparsion
    match = torch.allclose(baseline_out, custom_out, rtol=1e-2, atol=1e-2) # BF16 
    
    if match:
        if rank == 0:
            print(f"[rank {rank}]  PASS: Custom Output (Comm+GEMM) matches Baseline!")
    else:
        diff = (baseline_out - custom_out).abs().max()
        print(f"[rank {rank}]  FAIL: Max diff {diff}")

    # Release memo
    del baseline_out, custom_out
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        print(">> Verification done. Starting benchmark loops...")
    
    func_args = (rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, None, False, local_weights)
    
# Warmup
    for _ in range(5):
        if run_custom_a2a:
            
            CustomA2A(*func_args, recv_counts=recv_counts)
        else:
            TorchA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, False, None, local_weights)
        dist.barrier()

    torch.cuda.synchronize()

    # Timing 
    start = time.time()
    for _ in range(10):
        if run_custom_a2a:
            
            CustomA2A(*func_args, recv_counts=recv_counts)
        else:
            TorchA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, False, None, local_weights)
    dist.barrier()
    torch.cuda.synchronize()
    end = time.time()

    print(f"[rank {rank}] mode={'custom' if run_custom_a2a else 'torch'} time={(end - start):.6f}s")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
