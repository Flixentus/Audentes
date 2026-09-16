import torch
import torch.nn as nn
from torch import Tensor

from Router import Router
from Experts import Experts

class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int, capacity_factor: float = 1.25, dropout: float = 0.1):
        super().__init__()
        self.router = Router(d_model, num_experts, top_k)
        self.experts = Experts(d_model, d_ff, num_experts, top_k, capacity_factor, dropout)
        
    def forward(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        top_k_experts, top_k_probs, aux_loss, z_loss = self.router(X)
        output, drop_rate = self.experts(X, top_k_experts, top_k_probs)
        
        return (output, aux_loss, z_loss, drop_rate)