import torch
import torch.nn.functional as F
import torch.distributed as dist
import math


def gen_tensor(
    batch: int, seq: int, hidden_dim: int, 
    world_size: int, num_experts: int, 
    rank: int, topk: int) -> tuple[torch.Tensor, int, list]:

    torch.manual_seed(rank)
    assert num_experts % world_size == 0, "Incorrect EP_SIZE, world_size should evenly divide num_experts."

     # mimic Input Tokens
    # Shape: [batch * seq, hidden_dim
    tokens = torch.rand(batch*seq, hidden_dim, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")
   
    # stimulate router/gating
    router = torch.zeros(hidden_dim, num_experts, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")
    # Add small random perturbation
    router = router + torch.randn_like(router) * 0.001
    # Compute Routing: Logits -> Softmax -> TopK
    routed_values = F.softmax(torch.einsum('bh, he -> be', tokens, router), dim=-1)
    top_vals, top_idxs = torch.topk(routed_values, topk)

     # Bucketize tokens per Expert
    expert_tokens = []
    real_expert_lengths = [] # records actual valid length for zero-skipping kernel
    for ex in range(num_experts):
        mask = (top_idxs == ex).any(dim=-1)
        tkns = tokens[mask] # (num tokens routed to an expert, hidden dimension).
        if tkns.numel() > 0:
            real_expert_lengths.append(tkns.shape[0])
            expert_tokens.append(tkns)
        else:
            real_expert_lengths.append(0) # 0 if empty
           # Handle empty buckets: record 0 length, but append a 1-row dummy 
            # to simplify the padding logic (avoids concatenation errors with empty tensors)
            
            tmp_tkns = torch.zeros(1, hidden_dim, dtype=torch.bfloat16, device="cuda")
            expert_tokens.append(tmp_tkns)
    
    # Global Padding Logic
    # Calculate global max token count to ensure uniform shapes for NCCL All-to-All
    max_tkn_cnt = max([i.shape[0] for i in expert_tokens]) # max_tkn_cnt in local

    # Exhcange between devices
    global_max = torch.tensor([max_tkn_cnt], dtype=torch.int32).to("cuda" if torch.cuda.is_available() else "cpu") 
    
    dist.all_reduce(global_max, dist.ReduceOp.MAX) # New: then pad token bucket of each expert to global_max
   
    if rank == 0:
       
        print(f'[rank: {rank}]: global_max_tkn_cnt: {global_max.item()}')

    # perform padding
    padded_experts = []
    for i, t in enumerate(expert_tokens):
        # target length
        target_len = global_max.item()
        current_len = t.shape[0]
        
        if current_len < target_len:
            pad = torch.zeros(target_len - current_len, hidden_dim, dtype=tokens.dtype, device=tokens.device)
            padded_experts.append(torch.cat((t, pad)))
        else:
            padded_experts.append(t)


    # coalesce for All-to-All
    # concatenate all padded buckets along dim 0
    coalesced_experts = torch.cat(padded_experts, dim=0) 
    
    # Define Chunk Size for EP
    expert_per_device = num_experts // world_size
    chunk_size = global_max.item() * expert_per_device
    
    return coalesced_experts, chunk_size, real_expert_lengths

    # new gen weights
def gen_weights(num_experts, hidden_dim, out_dim, dtype=torch.bfloat16):
    # Generate weights for all experts: [E, H, H_out]
    return torch.randn(num_experts, hidden_dim, out_dim, dtype=dtype, device="cuda")