# benchmark.py
import os
import time
import subprocess

import torch
import torch.distributed as dist

from utils import gen_tensor, gen_weights
from alltoall_persistent_gemm import run as CustomA2A
from baseline import run as TorchA2A
import argparse
from torch.profiler import profile, record_function, ProfilerActivity


# new 4 reproducible seed
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def main(chunk_size, batch, seq, hidden_dim, num_experts, topk):
    rank, world_size, local_rank = infer_rank_info()
    preflight(rank)

    #new seed
    set_seed(42 + rank)

    if not torch.cuda.is_available():
        raise RuntimeError(f"[rank {rank}] No CUDA detected.")

    # Device setup
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    vis_list = [x.strip() for x in visible.split(",") if x.strip() != ""]
    device_index = 0 if len(vis_list) == 1 else local_rank
    torch.cuda.set_device(device_index)

    # Dist setup
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", os.getenv("MASTER_ADDR", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", os.getenv("MASTER_PORT", "29500"))
    dist.init_process_group(backend="nccl", init_method="env://")

    # now totally based on paras from arges
    #batch, seq, hidden_dim, topk = 4, 2048, 4096, 2  
    out_dim = hidden_dim # Assume output dim equals input dim
    #num_experts = world_size * 32                    
    
    run_custom_a2a = (os.getenv("RUN_CUSTOM", "0") == "1")

    # Generate Tokens
    # sent_lengths: list of int, length = num_experts. 
    # Token count sent from Rank i to Expert j
    tokens, chunk_size, sent_lengths = gen_tensor(batch, seq, hidden_dim, world_size, num_experts, rank, topk)
    
    # Prepare Weights / Counts
    num_local_experts = num_experts // world_size
    sent_counts_tensor = torch.tensor(sent_lengths, dtype=torch.int32, device="cuda")
    
    # Reshape logic: Expert e belongs to Rank (e // NumLocalExperts).
    # sent_counts_tensor is already ordered by destination rank.
    # reshape to [WorldSize, NumLocalExperts].
    sent_counts_reshaped = sent_counts_tensor.view(world_size, num_local_experts)
    
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
    
    # New to CPU
    # carry to CPU
    baseline_cpu = baseline_out.to("cpu", non_blocking=True)
    # del those on GPU 
    del baseline_out 
    # clear GPU memory
    torch.cuda.empty_cache() 
    dist.barrier()
    
    # Custom
    custom_out = CustomA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, None, False, local_weights, 
                           recv_counts=recv_counts)
    dist.barrier()

    # New comparison logic (all with cpu data)
    custom_cpu = custom_out.to("cpu") # 1 time carry only
    match = torch.allclose(baseline_cpu, custom_cpu, rtol=1e-2, atol=1e-2) # BF16 nothing different I think
    
    if match:
        if rank == 0:
            print(f"[rank {rank}]  PASS: Custom Output (Comm+GEMM) matches Baseline!")
    else:
        # only change is use cpu data as above
        diff = (baseline_cpu - custom_cpu).abs().max()
        print(f"[rank {rank}]  FAIL: Max diff {diff}")

    # release memo
    # same sort of changes as above
    del custom_out, baseline_cpu, custom_cpu
    torch.cuda.empty_cache()
    dist.barrier()
    ### I think its nothing different, will put data into cpu change that lot ? Also it's a change in case of OOM as I typed in slack today 1/14/26

    if rank == 0:
        print(">> Verification done. Starting benchmark loops...")
    
    func_args = (rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, None, False, local_weights)
    
    # Warmup
    for _ in range(5):
        if run_custom_a2a:
            #torch.cuda.empty_cache() ### try to use it also for OOM, BUT after delete it, at least the results are the results are monotonically increasing and comparable 
            CustomA2A(*func_args, recv_counts=recv_counts)
        else:
            TorchA2A(*func_args)
        dist.barrier()

    torch.cuda.synchronize()

    print(f"[rank {rank}] Starting Profiler & Benchmark...") ### Thios one do no harm to timing right
    
    # Timing
    ### I think the problem mainly comes here, but I asked GPT about my code's shortcomes, it said this timing is more reliable and I think it make sense

    # NEW: enable CPU+CUDA profiling and shape recording
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                 record_shapes=True) as prof:
        
        # NEW: label region in trace for easy locating
        with record_function("My_Triton_Benchmark"):
            
            dist.barrier()  # NEW/CHANGED: align all ranks before timing window
            torch.cuda.synchronize()  # NEW/CHANGED: flush pending CUDA work before timing
            start = time.perf_counter() # CHANGED: higher-res timer than time.time()
            for i in range(10):
                # Optional: per-iteration region label in trace (disabled by default to reduce overhead)
                # with record_function(f"step_{i}"):
                    
                    if run_custom_a2a:
                        CustomA2A(*func_args, recv_counts=recv_counts)
                    else:
                        TorchA2A(*func_args)
            
            torch.cuda.synchronize()   # NEW/CHANGED: ensure all GPU work finishes before end timestamp
            dist.barrier() # NEW/CHANGED: ensure ranks complete loop before timing ends
            end = time.perf_counter()  # CHANGED: paired with perf_counter()
    
    # NEW: export Chrome trace for offline inspection (rank0 only to avoid file write collisions)
    if rank == 0:
        trace_name = "trace_custom.json" if run_custom_a2a else "trace_baseline.json"
        prof.export_chrome_trace(trace_name)
        print(f"[rank {rank}] Trace exported to: {trace_name}")


    print(f"[rank {rank}] mode={'custom' if run_custom_a2a else 'torch'} time={(end - start):.6f}s")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # new para lists ez for sweep
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
    parser.add_argument('--num_experts', type=int, default=16, help='Number of Experts')
    parser.add_argument('--topk', type=int, default=2, help='Top K experts')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Hidden dimension')
    
    # chunk size stays as default
    parser.add_argument('--chunk_size', type=int, default=32, help='Chunk size')

    args = parser.parse_args()

    # config output
    print(f">> Config: B={args.batch}, S={args.seq_len}, E={args.num_experts}, K={args.topk}, H={args.hidden_dim}")

    #new main for sweep
    main(args.chunk_size, args.batch, args.seq_len, args.hidden_dim, args.num_experts, args.topk)