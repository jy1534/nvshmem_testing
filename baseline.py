"""
Baseline a2a implementation to compare perf against.
"""

import torch
import torch.distributed as dist

def run(
    rank: int, tokens: torch.Tensor, chunk_size: int, 
    batch: int, seq: int, hidden_dim: int, 
    num_experts: int, world_size: int, 
    general_a2a: bool, shmem,
    local_weights: torch.Tensor = None #local expert weights
):
    if general_a2a:
        pass 
    else:
        # Communication (NCCL)
        # alloc recv buffer
        tokens_recv = torch.zeros(chunk_size * world_size, hidden_dim, dtype=tokens.dtype, device=tokens.device)
        
        # NCCL All-to-All
        dist.all_to_all_single(tokens_recv, tokens) 
        

        # Computation (Grouped GEMM)
        # tokens_recv struct: [Rank0 data] [Rank1 data]....
        #  [Rank X data] contains the data for Local Experts
        # Shape:
        # Total Size = World_Size * Chunk_Size
        # Chunk_Size = Num_Local_Experts * Global_Max_Tokens
        
        num_local_experts = local_weights.shape[0]
        # Derive global_max from chunk_size (assuming padding)
        global_max = chunk_size // num_local_experts
        
        # Reshape view: Split into [WorldSize, LocalExperts, GlobalMax, Hidden]
        # Note: Dim 1 (LocalExperts) is interleaved here, as each Rank sends data for all Local Experts
        reshaped = tokens_recv.view(world_size, num_local_experts, global_max, hidden_dim)
        
        # Permute dimensions: Move LocalExperts to the front -> [LocalExperts, WorldSize, GlobalMax, Hidden]
        permuted = reshaped.permute(1, 0, 2, 3)
        
        # Flatten: Concatenate tokens for the same Expert across all Ranks -> [LocalExperts, TotalTokensPerExpert, Hidden]
        # TotalTokensPerExpert = WorldSize * GlobalMax
        input_for_gemm = permuted.flatten(1, 2)
        
        # do Batch Matrix Multiply
        # [LocalExperts, N, H] @ [LocalExperts, H, Out] -> [LocalExperts, N, Out]
        output = torch.bmm(input_for_gemm, local_weights)


        torch.cuda.synchronize()
        
       
        return output