import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        assert top_k <= num_experts, "top_k must be <= num_experts"
        
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.top_k = top_k
        
    def forward(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Compute the logits for each expert
        with torch.autocast(device_type=X.device.type, enabled=False):
            logits = self.router(X.float())  # (batch_size, seq_len, num_experts)
        
        # Compute softmax over the last dimension
        probs = F.softmax(logits, dim= -1)  # (batch_size, seq_len, num_experts)
        
        # Get the top-k experts for each token
        top_k_probs, top_k_experts = probs.topk(self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True) # Normalize to get probabilities

        # Compute the aux/ z losses
        aux_loss = self._load_balance_loss(probs, top_k_experts)
        z_loss = self._z_loss(logits)
        
        return top_k_experts, top_k_probs, aux_loss, z_loss
    
    def _load_balance_loss(self, probs: Tensor, experts: Tensor) -> Tensor:
        """
        loss = num_experts * Σ(f_i * P_i)
        """
        num_experts = probs.size(-1)
        
        # Get f_i fraction of tokens assigned to expert i
        mask = F.one_hot(experts, num_classes=num_experts).sum(dim=-2).float()
        f_i = mask.mean(dim= (0, 1))
        
        # Get P_i average probability assigned to expert i
        P_i = probs.mean(dim= (0, 1))
        
        # Return the load balance loss
        return num_experts * (f_i * P_i).sum()
    
    def _z_loss(self, logits: Tensor) -> Tensor:
        """
        loss = (1/seq_len) * Σ_t (logsumexp(logits_t))²
        """
        
        log_z = torch.logsumexp(logits, dim=-1)
        return (log_z ** 2).mean()


