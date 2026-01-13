import os
import time
import subprocess
import argparse

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
    # print(f"[rank {rank}] CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
    # print(f"[rank {rank}] LD_LIBRARY_PATH={os.getenv('LD_LIBRARY_PATH')}")
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        # print(f"[rank {rank}] nvidia-smi -L:\n{out}")
    except Exception as e:
        print(f"[rank {rank}] nvidia-smi unavailable: {e}")
    # print(f"[rank {rank}] torch={torch.__version__}")
    if torch.cuda.is_available():
        pass
        # print(f"[rank {rank}] device_count={torch.cuda.device_count()}")


def main(chunk_size, batch, seq, hidden_dim, num_experts, topk):
    rank, world_size, local_rank = infer_rank_info()
    preflight(rank)

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"[rank {rank}] torch.cuda.is_available is False. "
        )

    # Decide CUDA device index robustly under srun
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    vis_list = [x.strip() for x in visible.split(",") if x.strip() != ""]
    device_index = 0 if len(vis_list) == 1 else local_rank
    torch.cuda.set_device(device_index)

    # Init Process Group
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", os.getenv("MASTER_ADDR", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", os.getenv("MASTER_PORT", "29500"))

    dist.init_process_group(backend="nccl", init_method="env://")

    # === Config (已修改：直接使用传入参数) ===
    out_dim = hidden_dim 
    # 确保 experts 能被整除 (可选)
    if num_experts % world_size != 0:
        if rank == 0:
            print(f"Warning: num_experts ({num_experts}) is not divisible by world_size ({world_size})")

    # 1. 生成 Tokens
    # sent_lengths: [E0, E1, ..., E_total]
    tokens, chunk_size, sent_lengths = gen_tensor(batch, seq, hidden_dim, world_size, num_experts, rank, topk)
    
    # 2. 准备通信元数据
    num_local_experts = num_experts // world_size
    sent_counts_tensor = torch.tensor(sent_lengths, dtype=torch.int32, device="cuda")
    sent_counts_reshaped = sent_counts_tensor.view(world_size, num_local_experts)

    # 准备接收 buffer
    recv_counts = torch.empty_like(sent_counts_reshaped)
    dist.all_to_all_single(recv_counts, sent_counts_reshaped)
    
    # 3. 生成 Weights
    torch.manual_seed(rank + 10086)
    local_weights = gen_weights(num_local_experts, hidden_dim, out_dim, dtype=tokens.dtype)


    # === Correctness Check ===
    if rank == 0:
        print(">> Running correctness verification (Comm + GEMM)...")

    # 跑 Baseline
    baseline_out = TorchA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, False, None, local_weights)
    
    dist.barrier()
    
    # 跑 Custom (Triton) - 确保传入 recv_counts
    custom_out = CustomA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, None, False, local_weights, 
                           recv_counts=recv_counts)

    dist.barrier()

    # 对比
    match = torch.allclose(baseline_out, custom_out, rtol=1e-2, atol=1e-2) # BF16 tolerance
    
    if match:
        if rank == 0:
            print(f"[rank {rank}]  PASS: Custom Output (Comm+GEMM) matches Baseline!")
    else:
        diff = (baseline_out - custom_out).abs().max()
        if rank == 0:
            print(f"[rank {rank}]  FAIL: Max diff {diff}")

    # 清理显存
    del baseline_out, custom_out
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        print(">> Verification done. Starting benchmark loops...")
    
    # === Warmup ===
    # 预热两边，确保 cache ready
    for _ in range(5):
        TorchA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, False, None, local_weights)
        dist.barrier()
        CustomA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, None, False, local_weights, recv_counts=recv_counts)
        dist.barrier()

    torch.cuda.synchronize()

    # === 1. Timing Baseline (Torch) ===
    dist.barrier()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        TorchA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, False, None, local_weights)
    
    dist.barrier()
    torch.cuda.synchronize()
    end = time.time()
    
    # 打印格式必须包含 "mode=torch time=..." 方便 grep
    if rank == 0:
        print(f"[rank {rank}] mode=torch time={(end - start):.6f}s")

    # === 2. Timing Custom (Triton) ===
    dist.barrier()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        CustomA2A(rank, tokens, chunk_size, batch, seq, hidden_dim, num_experts, world_size, None, False, local_weights, recv_counts=recv_counts)
    
    dist.barrier()
    torch.cuda.synchronize()
    end = time.time()
    
    # 打印格式必须包含 "mode=custom time=..." 方便 grep
    if rank == 0:
        print(f"[rank {rank}] mode=custom time={(end - start):.6f}s")
        
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 参数定义
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
    parser.add_argument('--num_experts', type=int, default=16, help='Number of Experts')
    parser.add_argument('--topk', type=int, default=2, help='Top K experts')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Hidden dimension')
    parser.add_argument('--chunk_size', type=int, default=32, help='Chunk size')

    args = parser.parse_args()

    # 打印配置
    print(f">> Config: B={args.batch}, S={args.seq_len}, E={args.num_experts}, K={args.topk}, H={args.hidden_dim}")

    # 调用 main
    main(args.chunk_size, args.batch, args.seq_len, args.hidden_dim, args.num_experts, args.topk)