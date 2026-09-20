import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rope import RoPE

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_base: float = 10_000) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.kv_group = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        
        # Set Q, K, V and W_o projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.W_o = nn.Linear(d_model, d_model)
        
        #  Initialize RoPE encoding        
        self.rope = RoPE(self.head_dim, max_seq_len, rope_base)
        
    def forward(self, X: Tensor, offset: int = 0, kv_cache: dict | None = None) -> tuple[Tensor, dict]:
        
        #TODO: Apply Flash-Attention
        
        batch_size, seq_len, _ = X.shape
        
        # Project Q, K, V
        Q = self.q_proj(X).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(X).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        V = self.v_proj(X).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE encoding to Q and K
        Q = self.rope.apply_rotary(Q, offset=offset)
        K = self.rope.apply_rotary(K, offset=offset)
        
        # Transpose Q, K, V
        Q = Q.transpose(1,2) # (batch_size, num_heads, seq_len, head_dim)
        K = K.transpose(1,2) # (batch_size, num_kv_heads, seq_len, head_dim)
        V = V.transpose(1,2) # (batch_size, num_kv_heads, seq_len, head_dim)
        
        if kv_cache is not None:
            # Concat on seq_len
            K = torch.cat((kv_cache["K"], K), dim=2)
            V = torch.cat((kv_cache["V"], V), dim=2)
            
        new_cache = {"K": K, "V": V}
        
        # Apply a causal mask that accounts for cached keys.
        Q_len = Q.size(2)
        K_len = K.size(2)
        past_len = K_len - Q_len
        if past_len < 0:
            raise ValueError("kv_cache cannot contain fewer tokens than the current input")

        if past_len == 0:
            attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, enable_gqa=True)
        else:
            query_positions = torch.arange(Q_len, device=X.device).unsqueeze(1)
            key_positions = torch.arange(K_len, device=X.device).unsqueeze(0)
            causal_mask = key_positions <= past_len + query_positions
            attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=causal_mask, enable_gqa=True)
        
        # Transpose back and reshape
        attn_out = attn_out.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        attn_out = self.W_o(attn_out)
        
        return (attn_out, new_cache)
