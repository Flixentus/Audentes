import torch, torch.nn as nn
from Attention import GroupedQueryAttention
from mlp import MLP

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, max_seq_len, d_ff):
        super().__init__()
        self.self_attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads, max_seq_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, cache = self.self_attn(normed, offset=0, kv_cache=None) 
        x = x + attn_out

        normed2 = self.norm2(x)
        mlp_out = self.mlp(normed2)
        x = x + mlp_out

        return x
    