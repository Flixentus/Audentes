import torch
import torch.nn as nn
from torch import Tensor

from Attention import GroupedQueryAttention
from MoE import MoELayer


import yaml
from pathlib import Path

print(Path().absolute())
print(Path().absolute().parent)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(d_model))
        
    def forward(self, X: Tensor) -> Tensor:
        # Formula for RMSNorm: x / sqrt(mean(x²) + eps)
        norm = torch.rsqrt(X.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * X * norm


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, max_seq_len: int, d_ff: int, num_experts: int, top_k: int, capacity_factor: float = 1.25, rope_base: float = 10_000, dropout: float = 0.1, eps: float = 1e-6):
            super().__init__()
            self.attn_norm = RMSNorm(d_model, eps)
            self.attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads, max_seq_len, rope_base)
            
            self.moe = MoELayer(d_model, d_ff, num_experts, top_k, capacity_factor, dropout)
            self.moe_norm = RMSNorm(d_model, eps)
            
            self.dropout = nn.Dropout(dropout)

    def forward(self, X: Tensor, offset: int = 0, kv_cache: dict | None = None) -> tuple[Tensor, dict, Tensor, Tensor, float]:
        #Self-attention
        attn_out, new_cache = self.attention(X, offset=offset, kv_cache=kv_cache)
        
        # Residual connection + RMSNorm
        X = X + self.dropout(attn_out)
        X = self.attn_norm(X)
        
        # MoE
        output, aux_loss, z_loss, drop_rate = self.moe(X)
        # Residual connection + RMSNorm
        X = X + self.dropout(output)
        X = self.moe_norm(X)
        
        return (X, new_cache, aux_loss, z_loss, drop_rate)
    

if __name__ == "__main__":
    
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)["Model"]

    block = DecoderBlock(d_model=config["d_model"], num_heads=config["num_heads"], num_kv_heads=config["num_kv_heads"],  max_seq_len=config["max_seq_len"], d_ff=config["d_ff"], num_experts=config["num_experts"], top_k=config["top_k"], capacity_factor=config["capacity_factor"], rope_base=config["rope_base"], dropout=config["dropout"], eps=float(config["eps"]))

    inputs = torch.randn(2, 16, config["d_model"])
    output, kv_cache, aux_loss, z_loss, drop_rate = block(inputs)

    print("Input shape:", inputs.shape)
    print("Output shape:", output.shape)
    print("Key cache shape:", kv_cache["K"].shape)
    print("Value cache shape:", kv_cache["V"].shape)
    print("Aux loss:", aux_loss.item())
    print("Z loss:", z_loss.item())
    print("Drop rate:", drop_rate)