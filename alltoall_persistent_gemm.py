import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton import Config

try:
    import torch.distributed._symmetric_memory as symm_mem
except Exception as e:
    symm_mem = None
    _SYMM_IMPORT_ERR = repr(e)

# Cache rendezvoused symmetric buffers
_SYMM_CACHE = {}

def get_configs():
    """
    Generate a list of configurations for the Hopper architecture (GH200).
    Focus on larger tiles to utilize SRAM and aggressive pipelining for HBM3e.
    """
    configs = []
    
      
    # Structure: (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
    settings = [
        # Balanced / Large Tile (Good for large GEMM phases)
        (128, 256, 64, 8, 4),
        (128, 128, 64, 8, 4),
        (256, 128, 64, 8, 5), # High throughput
        (128, 64,  64, 4, 4),
        
        # Latency sensitive (Good for smaller effective batches per expert)
        (64,  256, 64, 8, 4),
        (64,  128, 64, 4, 3),
        (64,  64,  64, 4, 3),
        
        # Conservative / Smaller K (If K is small or shapes are weird)
        (128, 128, 32, 4, 4),
        (32,  128, 64, 4, 2), # Fallback for very small tasks
        (32, 128, 64, 4, 2),  
        (32, 64,  64, 4, 2),
        (16, 128, 64, 4, 2),  
        (16, 64,  64, 2, 2),
    ]

    for (bm, bn, bk, w, s) in settings:
        configs.append(
            Config(
                {'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk, 'GROUP_SIZE_M': 8},
                num_warps=w,
                num_stages=s
            )
        )
    return configs


@triton.autotune(
    configs=get_configs(),
    key=['world_size', 'global_max_tokens', 'hidden_dim', 'out_dim'],
)
@triton.jit
def fused_grouped_gemm_kernel(
    # Pointers
    recv_ptr, signal_ptr, weights_ptr, output_ptr, counts_ptr,
    # Dimensions
    world_size, global_max_tokens, hidden_dim, out_dim,
    # Strides
    stride_recv_rank, stride_recv_exp, stride_recv_tok, stride_recv_h,
    stride_w_exp, stride_w_h, stride_w_out,
    stride_out_exp, stride_out_tok, stride_out_d,
    stride_counts_rank, stride_counts_exp,
    # Meta-parameters (Injected by Autotuner)
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Kernel logic remains largely identical, but relies on constexpr for Blocks
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(global_max_tokens, BLOCK_M)
    num_pid_n = tl.cdiv(out_dim, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    expert_idx = tl.program_id(axis=1)

    # Pre-calc Weight Pointers
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % out_dim
    
    # Weights: [Expert, K, N]
    w_ptrs_base = weights_ptr + expert_idx * stride_w_exp + \
                  (offs_k[:, None] * stride_w_h + offs_n[None, :] * stride_w_out)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over ranks (Fused Loop)
    for src_rank in range(world_size):
        # 1. Busy Wait
        sig_addr = signal_ptr + src_rank
        while tl.load(sig_addr, volatile=True) != 1:
            pass 

        # 2. Check Valid Length
        count_offset = src_rank * stride_counts_rank + expert_idx * stride_counts_exp
        valid_len = tl.load(counts_ptr + count_offset)
        
        start_m = pid_m * BLOCK_M
        
        # Computation Branch
        if start_m < valid_len:
            base_recv = recv_ptr + (src_rank * stride_recv_rank) + (expert_idx * stride_recv_exp)
            offs_m = start_m + tl.arange(0, BLOCK_M)
            m_mask = offs_m < valid_len
            
            # Loop over K
            for k in range(0, tl.cdiv(hidden_dim, BLOCK_K)):
                # Load A
                k_start = k * BLOCK_K
                k_mask = (k_start + offs_k) < hidden_dim
                
                a_ptrs = base_recv + (offs_m[:, None] * stride_recv_tok + \
                                      (k_start + offs_k)[None, :] * stride_recv_h)
                a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
                
                # Load B
                curr_k_inds = k_start + offs_k
                # Recalculate W ptrs to avoid complex pointer arithmetic accumulation errors
                w_curr_ptrs = weights_ptr + expert_idx * stride_w_exp + \
                              (curr_k_inds[:, None] * stride_w_h + offs_n[None, :] * stride_w_out)
                b = tl.load(w_curr_ptrs, mask=(curr_k_inds[:, None] < hidden_dim), other=0.0)
                
                acc += tl.dot(a, b)
    
    pass #

    # Refined Logic for Rank-Stacked Output
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    for src_rank in range(world_size):
        # Wait
        sig_addr = signal_ptr + src_rank
        while tl.load(sig_addr, volatile=True) != 1:
            pass 

        # Check Valid
        count_offset = src_rank * stride_counts_rank + expert_idx * stride_counts_exp
        valid_len = tl.load(counts_ptr + count_offset)
        start_m = pid_m * BLOCK_M
        
        if start_m < valid_len:
            # Re-init accumulator for THIS rank's slice
            acc_local = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            
            base_recv = recv_ptr + (src_rank * stride_recv_rank) + (expert_idx * stride_recv_exp)
            m_mask = offs_m < valid_len
            
            for k in range(0, tl.cdiv(hidden_dim, BLOCK_K)):
                k_start = k * BLOCK_K
                k_mask = (k_start + offs_k) < hidden_dim
                
                a_ptrs = base_recv + (offs_m[:, None] * stride_recv_tok + \
                                      (k_start + offs_k)[None, :] * stride_recv_h)
                a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
                
                curr_k_inds = k_start + offs_k
                w_curr_ptrs = weights_ptr + expert_idx * stride_w_exp + \
                              (curr_k_inds[:, None] * stride_w_h + offs_n[None, :] * stride_w_out)
                b = tl.load(w_curr_ptrs, mask=(curr_k_inds[:, None] < hidden_dim), other=0.0)
                
                acc_local += tl.dot(a, b)
            
            # Store Result for THIS rank
            c = acc_local.to(tl.bfloat16)
            
            global_row_start = (src_rank * global_max_tokens) + start_m
            out_base = output_ptr + expert_idx * stride_out_exp
            
            offs_global_m = global_row_start + tl.arange(0, BLOCK_M)
            out_row_mask = offs_m < valid_len 
            mask_n = offs_n < out_dim
            
            out_ptrs = out_base + (offs_global_m[:, None] * stride_out_tok + offs_n[None, :] * stride_out_d)
            tl.store(out_ptrs, c, mask=out_row_mask[:, None] & mask_n[None, :])


def _cache_key(shape, dtype, device, group):
    gname = getattr(group, "group_name", "WORLD")
    return (tuple(shape), str(dtype), str(device), gname)


def _get_symm_buffers(shape, dtype, device, group, world_size):
    if symm_mem is None:
        raise RuntimeError(f"torch.distributed._symmetric_memory import failed: {_SYMM_IMPORT_ERR}.")

    key = _cache_key(shape, dtype, device, group)
    if key in _SYMM_CACHE:
        return _SYMM_CACHE[key]

    gname = getattr(group, "group_name", "WORLD")
    if not symm_mem.is_symm_mem_enabled_for_group(gname):
        symm_mem.enable_symm_mem_for_group(gname)
    
    SM_Class = symm_mem._SymmetricMemory

    stride = torch.empty(shape, device="meta").stride()
    recv_data = SM_Class.empty_strided_p2p(shape, stride, dtype, device, gname)
    
    signal_shape = (world_size,)
    signal_stride = (1,)
    recv_signal = SM_Class.empty_strided_p2p(signal_shape, signal_stride, torch.int32, device, gname)

    hdl_data = SM_Class.rendezvous(recv_data)
    hdl_signal = SM_Class.rendezvous(recv_signal)

    _SYMM_CACHE[key] = (recv_data, hdl_data, recv_signal, hdl_signal)
    return recv_data, hdl_data, recv_signal, hdl_signal


def run(
    rank: int,
    tokens: torch.Tensor,
    transmit_size: int,
    batch: int,
    seq: int,
    hidden_dim: int,
    num_experts: int,
    world_size: int,
    shmem=None,                 
    general_a2a: bool = False, 
    local_weights: torch.Tensor = None,
    recv_counts: torch.Tensor = None
):
    if general_a2a:
        raise NotImplementedError("general/unbalanced a2a not implemented.")

    assert dist.is_initialized()
    assert tokens.is_cuda

    group = dist.group.WORLD
    recv_shape = (transmit_size * world_size, tokens.shape[1])
    
    recv, hdl, recv_signal, hdl_signal = _get_symm_buffers(
        recv_shape, tokens.dtype, tokens.device, group, world_size
    )

    hdl.barrier(channel=0) 
    recv_signal.zero_() 
    torch.cuda.current_stream().synchronize() 
    hdl.barrier(channel=0)

    if not hasattr(run, "comm_stream"):
        run.comm_stream = torch.cuda.Stream()
    comm_stream = run.comm_stream

    num_local_experts = local_weights.shape[0]
    global_max = transmit_size // num_local_experts 
    out_dim = local_weights.shape[-1]
    
    reshaped_recv = recv.view(world_size, num_local_experts, global_max, hidden_dim)
    output_shape = (num_local_experts, world_size * global_max, out_dim)
    output = torch.zeros(output_shape, dtype=tokens.dtype, device=tokens.device)

    # Producer (Comm Stream)
    with torch.cuda.stream(comm_stream):
        for dst in range(world_size):
            peer_recv = hdl.get_buffer(dst, recv.shape, recv.dtype)
            send_chunk = tokens[dst * transmit_size : (dst + 1) * transmit_size]
            dst_off = rank * transmit_size
            peer_recv[dst_off : dst_off + transmit_size].copy_(send_chunk)
            
            peer_signal = hdl_signal.get_buffer(dst, recv_signal.shape, recv_signal.dtype)
            one = torch.tensor(1, dtype=torch.int32, device=tokens.device)
            peer_signal[rank].copy_(one, non_blocking=True)

    # onsumer (Main Stream / Triton)
    if recv_counts is None:
        recv_counts = torch.ones((world_size, num_local_experts), dtype=torch.int32, device=tokens.device) * global_max

    # Grid now decided by Lambda func,Triton will be fused into META
    grid = lambda META: (
        triton.cdiv(global_max, META['BLOCK_M']) * triton.cdiv(out_dim, META['BLOCK_N']),
        num_local_experts
    )

    fused_grouped_gemm_kernel[grid](
        reshaped_recv,      # Data
        recv_signal,        # Signals
        local_weights,      # Weights
        output,             # Output
        recv_counts,        # Counts
        
        world_size, global_max, hidden_dim, out_dim,
        
        reshaped_recv.stride(0), reshaped_recv.stride(1), reshaped_recv.stride(2), reshaped_recv.stride(3),
        local_weights.stride(0), local_weights.stride(1), local_weights.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        recv_counts.stride(0), recv_counts.stride(1),
        
        # Now BLOCK paras will be arranged by Autotuner
    
    )

    return output