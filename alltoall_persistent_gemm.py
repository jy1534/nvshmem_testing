# alltoall_persistent_gemm.py
import os
import torch
import torch.distributed as dist

try:
    import torch.distributed._symmetric_memory as symm_mem
except Exception as e:
    symm_mem = None
    _SYMM_IMPORT_ERR = repr(e)

# Cache rendezvoused symmetric buffers to avoid re-rendezvous every iteration
_SYMM_CACHE = {}


def _cache_key(shape, dtype, device, group):
    gname = getattr(group, "group_name", "WORLD")
    return (tuple(shape), str(dtype), str(device), gname)


def _get_symm_recv(shape, dtype, device, group):
    """
    Allocate + rendezvous a symmetric recv buffer once, then reuse.
    """
    if symm_mem is None:
        raise RuntimeError(
            f"torch.distributed._symmetric_memory import failed: {_SYMM_IMPORT_ERR}. "
            "Are you running inside the PyTorch container build that includes SymmetricMemory?"
        )

    key = _cache_key(shape, dtype, device, group)
    if key in _SYMM_CACHE:
        return _SYMM_CACHE[key]

    # Retrieve the group name (the error confirmed this is required).
    gname = getattr(group, "group_name", "WORLD")

    # ▼▼▼ Fix for NGC 24.09: Explicitly Enable SymmMem ▼▼▼
    # In this version, the NVSHMEM context is not initialized by default.
    # We must explicitly enable Symmetric Memory for the group before any allocation.
    if not symm_mem.is_symm_mem_enabled_for_group(gname):
        # This initializes the backend and registers the group info.
        symm_mem.enable_symm_mem_for_group(gname)
    # ▲▲▲ End Fix ▲▲▲

    # Access the internal class where low-level static methods reside in this build.
    SM_Class = symm_mem._SymmetricMemory
    
    # 1. Compute contiguous row-major strides.
    #    (Required because `empty_strided_p2p` does not infer strides automatically).
    stride = torch.empty(shape, device="meta").stride()
    
    # 2. Manually allocate P2P-accessible symmetric memory.
    #    Signature: (size, stride, dtype, device, group_name, alloc_id)
    #    This is analogous to `nvshmem_malloc` in the raw C++ API.
    recv = SM_Class.empty_strided_p2p(shape, stride, dtype, device, gname)

    # 3. Register the handle.
    #    Signature: (tensor) -> handle
    #    Note: In this version, `rendezvous` only accepts the tensor; the group context 
    #    is inferred from the tensor's metadata established during allocation.
    hdl = SM_Class.rendezvous(recv)

    _SYMM_CACHE[key] = (recv, hdl)
    return recv, hdl


def run(
    rank: int,
    tokens: torch.Tensor,
    transmit_size: int,
    batch: int,
    seq: int,
    hidden_dim: int,
    num_experts: int,
    world_size: int,
    shmem=None,                 # kept for callsite compatibility; unused in SymmMem path
    general_a2a: bool = False,  # keep signature; not implemented here
):
    """
    SymmMem-based a2a-only path (no grouped GEMM).
    Semantics match dist.all_to_all_single for equal splits:
      - tokens is split into world_size chunks along dim0, each of size transmit_size
      - rank i sends chunk j to dst=j
      - dst receives chunks from all src ranks stacked in rank order
    """
    if general_a2a:
        raise NotImplementedError("general/unbalanced a2a is not implemented in the SymmMem path yet.")

    assert dist.is_initialized(), "dist process group must be initialized before calling run()."
    assert tokens.is_cuda, "tokens must be on CUDA for SymmMem P2P copy."
    assert tokens.shape[0] == transmit_size * world_size, (
        f"Expected tokens.shape[0] == transmit_size * world_size, got {tokens.shape[0]} vs {transmit_size * world_size}"
    )

    group = dist.group.WORLD
    recv_shape = (transmit_size * world_size, tokens.shape[1])
    recv, hdl = _get_symm_recv(recv_shape, tokens.dtype, tokens.device, group)

    # Make sure previous iteration's remote writes are not racing with this iteration.
    # hdl.barrier(channel=0) is used in public examples. :contentReference[oaicite:6]{index=6}
    hdl.barrier(channel=0)

    src_off = rank * transmit_size

    # For each destination, write my chunk into the destination's recv slice [src_off : src_off+transmit_size].
    # get_buffer + copy_ is a typical SymmMem usage pattern. :contentReference[oaicite:7]{index=7}
    for dst in range(world_size):
        peer_recv = hdl.get_buffer(dst, recv.shape, recv.dtype)
        send_chunk = tokens[dst * transmit_size : (dst + 1) * transmit_size]
        peer_recv[src_off : src_off + transmit_size].copy_(send_chunk)

    hdl.barrier(channel=0)
    return recv
