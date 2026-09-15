import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Define the linear layers for the expert FFN
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X: Tensor) -> Tensor:
        # Apply the SwiGLU activation function to the input X
        gated = F.silu(self.gate_proj(X)) * self.up_proj(X)
        output = self.down_proj(gated)
        return self.dropout(output)
    
class Experts(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int, capacity_factor: float = 1.25, dropout: float = 0.1):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d_model, d_ff, dropout) for _ in range(num_experts)])
        self.num_experts = num_experts        
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
    def Token_Experts_Dispatch(self, flattened_X: Tensor, flattened_token_ids: Tensor, flattened_top_k_experts: Tensor, flattened_top_k_probs: Tensor) -> tuple[Tensor, float]:
        num_tokens = flattened_X.size(0)

        # Capacity limit used in Mixture-of-Experts routing.
        # Each expert is allowed to process at most this many tokens.
        capacity = int(self.capacity_factor * num_tokens * self.top_k // self.num_experts)

        output = torch.zeros_like(flattened_X)

        # Track how many selected assignments were dropped due to capacity overflow
        dropped = 0

        for expert in range(self.num_experts):
            # Select all token assignments that route to this expert
            mask = (flattened_top_k_experts == expert)
            token_ids = flattened_token_ids[mask]
            weights = flattened_top_k_probs[mask]

            # keep only the highest-probability ones if exceed capacity
            if token_ids.size(0) > capacity:
                keep = weights.topk(capacity).indices
                dropped += token_ids.size(0) - capacity
                token_ids = token_ids[keep]
                weights = weights[keep]

            # Nothing to do for this expert if no tokens routed here
            if token_ids.numel() == 0:
                continue

            # Pass to expert
            expert_output = self.experts[expert](flattened_X[token_ids])

            # Accumulate weighted expert outputs back into their original token positions 
            output.index_add_(0, token_ids, expert_output * weights.unsqueeze(-1))

        # Fraction of routed assignments that were dropped because of capacity constraints
        drop_rate = dropped / (num_tokens * self.top_k)

        return output, drop_rate
    
    def forward(self, X: Tensor) -> Tensor:
        pass # Abdelali khdmtk hadi hhhh
                
