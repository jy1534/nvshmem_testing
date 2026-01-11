import os
import torch
import torch.distributed as dist
import triton
import triton.language as tl

try:
    import torch.distributed._symmetric_memory as symm_mem
except Exception as e:
    symm_mem = None
    _SYMM_IMPORT_ERR = repr(e)

# Cache rendezvoused symmetric buffers to avoid re-rendezvous every iteration
_SYMM_CACHE = {}

@triton.jit
def fused_grouped_gemm_kernel(
    # Pointers to buffers
    recv_ptr,           # [WorldSize, NumExperts, GlobalMax, Hidden]
    signal_ptr,         # [WorldSize]
    weights_ptr,        # [NumExperts, Hidden, OutDim]
    output_ptr,         # [NumExperts, WorldSize*GlobalMax, OutDim]
    counts_ptr,         # [WorldSize, NumExperts] (int32)
    
    # Dimensions
    world_size,
    global_max_tokens,
    hidden_dim,
    out_dim,
    
    # Strides (Check layouts!)
    stride_recv_rank, stride_recv_exp, stride_recv_tok, stride_recv_h,
    stride_w_exp, stride_w_h, stride_w_out,
    stride_out_exp, stride_out_tok, stride_out_d,
    stride_counts_rank, stride_counts_exp,
    
    # Meta-parameters
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    Fused Kernel:
    1. Iterate through each src_rank (0..world_size-1).
    2. BUSY WAIT until signal[src_rank] == 1.
    3. Perform GEMM for the specific expert on that rank's data.
    """
    
    # Grid Logic: 
    # pid_m: Handles the M dimension (Tokens)
    # pid_n: Handles the N dimension (Output Features)
    # expert_idx: Which local expert we are computing for
    
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

    # Pre-calculate pointers dependent on Expert and N (Weights don't change per rank)
    # Weights: [Expert, K, N]
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % out_dim
    
    # Weight ptr: base + expert_offset + (k, n)
    # Assuming Weights are [NumExperts, Hidden, OutDim] -> Row Major in (H, Out)
    # Ptr = weights_ptr + expert_idx * stride_w_exp
    w_ptrs = weights_ptr + expert_idx * stride_w_exp + \
             (offs_k[:, None] * stride_w_h + offs_n[None, :] * stride_w_out)

    # Iterate over ranks (The "Fusion" Loop)
    for src_rank in range(world_size):
        
        # BUSY WAIT for Signal
        # Poll the signal flag from the source rank
        sig_addr = signal_ptr + src_rank # signal is [WorldSize]
        
        # volatile load to ensure we actually read memory
        while tl.load(sig_addr, volatile=True) != 1:
            pass 

        # Check Valid Length (Grouped Logic)
        # Get token count for (src_rank, expert_idx)
        count_offset = src_rank * stride_counts_rank + expert_idx * stride_counts_exp
        valid_len = tl.load(counts_ptr + count_offset)
        
        #Skip computation entirely if the current block corresponds to padding
        
        start_m = pid_m * BLOCK_M
        
        # Only compute if this block has at least some valid tokens
        if start_m < valid_len:
            
            # Calculate base pointers for Input (A) and Weights (B)
            base_recv = recv_ptr + (src_rank * stride_recv_rank) + (expert_idx * stride_recv_exp)
            
            offs_m = start_m + tl.arange(0, BLOCK_M)
            # Mask for M dimension (handle boundary of global_max AND valid_len)
            # valid_len is the tighter bound for computation
            m_mask = offs_m < valid_len
            
            # Accumulator
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            
            # Loop over K dimension (Hidden Dim)
            for k in range(0, tl.cdiv(hidden_dim, BLOCK_K)):
                # Load A (Input)
                # Input is [M, K]. Stride M = stride_recv_tok, Stride K = stride_recv_h
                # We need to handle K masking if hidden_dim is not multiple of BLOCK_K
                k_mask = (k * BLOCK_K + offs_k) < hidden_dim
                
                a_ptrs = base_recv + (offs_m[:, None] * stride_recv_tok + \
                                      (k * BLOCK_K + offs_k)[None, :] * stride_recv_h)
                
                # Load A with masking. Pad with 0
                # Note: m_mask handles rows > valid_len. k_mask handles cols > hidden
                a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
                
                
                # Recalc for clarity:
                curr_k_inds = k * BLOCK_K + offs_k
                w_curr_ptrs = weights_ptr + expert_idx * stride_w_exp + \
                              (curr_k_inds[:, None] * stride_w_h + offs_n[None, :] * stride_w_out)
                
                b = tl.load(w_curr_ptrs, mask=(curr_k_inds[:, None] < hidden_dim), other=0.0)
                
                # Compute
                acc += tl.dot(a, b)

            # Result
            # Output: [Expert, (Rank * GlobalMax + M), N]
            # We need to calculate the correct global row index
            # Start Row for this rank = src_rank * global_max_tokens
            global_row_start = (src_rank * global_max_tokens) + start_m
            
            offs_global_m = global_row_start + tl.arange(0, BLOCK_M)
            out_row_mask = offs_m < valid_len # Only write valid rows
            
            # Ptr = output_ptr + expert_off + row_off + col_off
            out_base = output_ptr + expert_idx * stride_out_exp
            out_ptrs = out_base + (offs_global_m[:, None] * stride_out_tok + offs_n[None, :] * stride_out_d)
            
            c = acc.to(tl.bfloat16) # Assume BF16 output
            
            # Mask N dimension
            mask_n = offs_n < out_dim
            
            tl.store(out_ptrs, c, mask=out_row_mask[:, None] & mask_n[None, :])



def _cache_key(shape, dtype, device, group):
    gname = getattr(group, "group_name", "WORLD")
    return (tuple(shape), str(dtype), str(device), gname)


def _get_symm_buffers(shape, dtype, device, group, world_size):
    
    if symm_mem is None:
        raise RuntimeError(
            f"torch.distributed._symmetric_memory import failed: {_SYMM_IMPORT_ERR}. "
        )

    key = _cache_key(shape, dtype, device, group)
    if key in _SYMM_CACHE:
        return _SYMM_CACHE[key]

    gname = getattr(group, "group_name", "WORLD")

    # Fix for NGC 24.09: Explicitly Enable SymmMem
    if not symm_mem.is_symm_mem_enabled_for_group(gname):
        symm_mem.enable_symm_mem_for_group(gname)
    

    SM_Class = symm_mem._SymmetricMemory

    # Data Buffer
    stride = torch.empty(shape, device="meta").stride()
    recv_data = SM_Class.empty_strided_p2p(shape, stride, dtype, device, gname)
    
    
    # Signal Buffer （for flag sync）
    
    signal_shape = (world_size,)
    signal_stride = (1,)
    recv_signal = SM_Class.empty_strided_p2p(signal_shape, signal_stride, torch.int32, device, gname)

    # Rendezvous (handshaking)
    # Simple try 
    hdl_data = SM_Class.rendezvous(recv_data)
    hdl_signal = SM_Class.rendezvous(recv_signal)

    # into Cache
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
    assert tokens.shape[0] == transmit_size * world_size

    group = dist.group.WORLD
    recv_shape = (transmit_size * world_size, tokens.shape[1])
    
    # Get Buffers
    recv, hdl, recv_signal, hdl_signal = _get_symm_buffers(
        recv_shape, tokens.dtype, tokens.device, group, world_size
    )

    # Reset Signals (Sync Logic)
    hdl.barrier(channel=0) 
    recv_signal.zero_() 
    torch.cuda.current_stream().synchronize() # Important
    hdl.barrier(channel=0)

    # Setup Streams
    if not hasattr(run, "comm_stream"):
        run.comm_stream = torch.cuda.Stream()
    comm_stream = run.comm_stream

    # View Setup
    num_local_experts = local_weights.shape[0]
    global_max = transmit_size // num_local_experts 
    out_dim = local_weights.shape[-1]
    
    # Reshape Recv: [WorldSize, LocalExperts, GlobalMax, Hidden]
    reshaped_recv = recv.view(world_size, num_local_experts, global_max, hidden_dim)
    
    # Output: [LocalExperts, WorldSize * GlobalMax, OutDim]
    
    output_shape = (num_local_experts, world_size * global_max, out_dim)
    output = torch.zeros(output_shape, dtype=tokens.dtype, device=tokens.device)

    # producer stream
    with torch.cuda.stream(comm_stream):
        for dst in range(world_size):
            peer_recv = hdl.get_buffer(dst, recv.shape, recv.dtype)
            send_chunk = tokens[dst * transmit_size : (dst + 1) * transmit_size]
            dst_off = rank * transmit_size
            peer_recv[dst_off : dst_off + transmit_size].copy_(send_chunk)
            
            peer_signal = hdl_signal.get_buffer(dst, recv_signal.shape, recv_signal.dtype)
            one = torch.tensor(1, dtype=torch.int32, device=tokens.device)
            peer_signal[rank].copy_(one) 

    # Consumer stream
    
    # Grid Config
    BLOCK_M = 32 # Token block size
    BLOCK_N = 32 # Output dim block size
    BLOCK_K = 32 # Inner dim
    
    # Grid: (M/BLOCK_M * N/BLOCK_N, NumLocalExperts)
    grid = (lambda META: (
        triton.cdiv(global_max, META['BLOCK_M']) * triton.cdiv(out_dim, META['BLOCK_N']),
        num_local_experts
    ))
    
    # Fallback to pure GEMM if no counts (sanity check)
    if recv_counts is None:
        
        recv_counts = torch.ones((world_size, num_local_experts), dtype=torch.int32, device=tokens.device) * global_max

    # Launch Kernel
    fused_grouped_gemm_kernel[grid](
        reshaped_recv,      # Data
        recv_signal,        # Signals
        local_weights,      # Weights
        output,             # Output
        recv_counts,        # Counts
        
        world_size, global_max, hidden_dim, out_dim,
        
        # Strides
        reshaped_recv.stride(0), reshaped_recv.stride(1), reshaped_recv.stride(2), reshaped_recv.stride(3),
        local_weights.stride(0), local_weights.stride(1), local_weights.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        recv_counts.stride(0), recv_counts.stride(1),
        
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_SIZE_M=8
    )

    return output
