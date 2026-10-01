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
        
    def Token_Experts_Dispatch(
        self,
        flattened_X,
        flattened_token_ids,
        flattened_top_k_experts,
        flattened_top_k_probs
    ):
        num_tokens = flattened_X.size(0)

        capacity = max(
            1,
            int(
                self.capacity_factor
                * num_tokens
                * self.top_k
                // self.num_experts
            )
        )

        # Keep the dispatch buffer in BF16 to match A100 autocast.
        output = torch.zeros(
            flattened_X.shape,
            device=flattened_X.device,
            dtype=torch.bfloat16
        )

        dropped = 0

        for expert in range(self.num_experts):

            mask = flattened_top_k_experts == expert

            token_ids = flattened_token_ids[mask]
            weights = flattened_top_k_probs[mask]

            if token_ids.size(0) > capacity:
                keep = weights.topk(capacity).indices

                dropped += token_ids.size(0) - capacity

                token_ids = token_ids[keep]
                weights = weights[keep]

            if token_ids.numel() == 0:
                continue

            expert_output = self.experts[expert](
                flattened_X[token_ids]
            )

            weighted_output = (
                expert_output.to(torch.bfloat16)
                * weights.to(torch.bfloat16).unsqueeze(-1)
            )

            output.index_add_(
                0,
                token_ids,
                weighted_output
            )

        drop_rate = (
            dropped / (num_tokens * self.top_k)
            if num_tokens
            else 0.0
        )

        return output, drop_rate

    def forward(self, X: Tensor, top_k_experts: Tensor, top_k_probs: Tensor) -> tuple[Tensor, float]:

        original_shape = X.shape # (batch_size, seq_len, d_model)

        flattened_X = X.reshape(-1, X.size(-1)) # ((batch_size*seq_len), d_model)
        flattened_top_k_experts = top_k_experts.reshape(-1, self.top_k) # now each row represents one token and the two values tell us which two experts were selected !
        flattened_top_k_probs = top_k_probs.reshape(-1, self.top_k) # give us the probability of these two values 

        flattened_token_ids = torch.arange(flattened_X.size(0), device=X.device).unsqueeze(1).expand(-1, self.top_k) # Track which original token each selected expert assignment belongs to

        output, drop_rate = self.Token_Experts_Dispatch(flattened_X, flattened_token_ids, flattened_top_k_experts, flattened_top_k_probs)   # Dispatch tokens to their selected experts and combine their outputs

        output = output.reshape(original_shape) # # Reshape the output back to the original batch and sequence dimensions

        return output, drop_rate

        # Salit khdamti :)
        # 3ab3ali sbe3
                
