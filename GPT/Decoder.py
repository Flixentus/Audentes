import torch
import torch.nn as nn
import json

from torch import Tensor
from Attention import GroupedQueryAttention
from Experts import Experts
from Router import Router

import os

print(os.getcwd())
print(os.listdir())

with open("config.json", "r") as f:
    config = json.load(f)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, max_seq_len: int, d_ff: int, num_experts: int, top_k: int, capacity_factor: float, dropout: float):
        super().__init__()
        self.self_attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads, max_seq_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.router = Router(d_model, num_experts, top_k)
        self.moe = Experts(d_model, d_ff, num_experts, top_k, capacity_factor, dropout)

    def forward(self, X: Tensor) -> Tensor:

        #Self-attention
        attn_out, new_cache = self.self_attn(X)

        # Residual connection + LayerNorm
        X = self.norm1(X + attn_out)

        # Ankhdmo b Experts.py hia MLP dial MoE
        top_k_experts, top_k_probs, aux_loss, z_loss = self.router(X)

        output, drop_rate = self.moe(X, top_k_experts, top_k_probs)

        # Residual connection + LayerNorm
        X = self.norm2(X + output)

        return X

if __name__ == "__main__":
    d_model = config["d_model"]
    num_heads = config["num_heads"]
    num_kv_heads = config["num_kv_heads"]
    max_seq_len = config["max_seq_len"]
    d_ff = config["d_ff"]
    num_experts = config["num_experts"]
    top_k = config["top_k"]
    capacity_factor = config["capacity_factor"]
    dropout = config["dropout"]

    block = DecoderBlock(
        d_model,
        num_heads,
        num_kv_heads,
        max_seq_len,
        d_ff,
        num_experts,
        top_k,
        capacity_factor,
        dropout
    )

    X = torch.randn(2, 16, d_model)

    top_k_experts, top_k_probs, aux_loss, z_loss = block.router(X)

    print("Top-k experts:")
    print(top_k_experts)

    print("Top-k probabilities:")
    print(top_k_probs)

    print("Aux loss:", aux_loss.item())
    print("Z loss:", z_loss.item())

    output = block(X)

    print("Input shape:", X.shape)
    print("Output shape:", output.shape)