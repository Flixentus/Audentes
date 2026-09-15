import torch
import torch.nn as nn
from torch import Tensor

from Attention import GroupedQueryAttention
from Experts import Experts

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, max_seq_len: int, d_ff: int, num_experts: int, top_k: int, capacity_factor: float, dropout: float):
        super().__init__()
        self.self_attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads, max_seq_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.moe = Experts(d_model, d_ff, num_experts, top_k, capacity_factor, dropout)

    def forward(self, X: Tensor) -> Tensor:
        # Ankhdmo b Experts.py hia MLP dial MoE
        pass