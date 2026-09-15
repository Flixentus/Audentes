import torch
import torch.nn as nn
from torch import Tensor

class RoPE(nn.Module):
    cos: Tensor
    sin: Tensor
    
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000):
        super().__init__()
        
        assert head_dim % 2 == 0, "head_dim must be even"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        # Precompute the theta values
        i = torch.arange(0, head_dim, 2).float()
        theta = base ** (-i/head_dim)
        
        pos = torch.arange(max_seq_len).float()
        angles = torch.outer(pos, theta)
        
        # Register the computed values as buffers
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
        
    def apply_rotary(self, X: Tensor, offset: int = 0):
        """
        Apply the rotary encoding to the input tensor X.
        """
        seq_len = X.size(1)
        assert offset + seq_len <= self.max_seq_len, (f"offset + seq_len ({offset + seq_len}) must be <= max_seq_len ({self.max_seq_len})")
        
        cos = self.cos[offset: offset + seq_len].to(X.dtype).unsqueeze(0).unsqueeze(2)
        sin = self.sin[offset: offset + seq_len].to(X.dtype).unsqueeze(0).unsqueeze(2)
        
        # Apply the rotary encoding
        x1, x2 = X.chunk(2, dim=-1)

        x1_new = x1 * cos - x2 * sin
        x2_new = x1 * sin + x2 * cos  # ← Use original x1
        
        return torch.cat((x1_new, x2_new), dim=-1)
