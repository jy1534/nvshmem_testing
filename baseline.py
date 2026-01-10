"""
Baseline a2a implementation to compare perf against.
"""

import torch
import torch.distributed as dist

def run(
    rank: int, tokens: torch.Tensor, chunk_size: int, 
    batch: int, seq: int, hidden_dim: int, 
    num_experts: int, world_size: int, 
    general_a2a: bool, shmem
):
    if general_a2a:
        pass ## Not implemented yet. ##
    else:
        # ALLOC buffer
        tokens_recv = torch.zeros(chunk_size * world_size, hidden_dim, dtype=tokens.dtype, device=tokens.device)
        
        # NCCL All-to-All
        dist.all_to_all_single(tokens_recv, tokens) 

        torch.cuda.synchronize()
        
        
        return tokens_recv