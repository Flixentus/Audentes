import torch 
import torch.nn as nn 
from torch import Tensor

from Decoder import RMSNorm, DecoderBlock

import yaml
from pathlib import Path
from typing import cast

class Audentes(nn.Module):
    def __init__(self, config_path: str | Path):
        super().__init__()

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)["Model"]
        
        self.token_embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        
        self.blocks = nn.ModuleList([DecoderBlock(d_model= config["d_model"], num_heads= config["num_heads"], num_kv_heads= config["num_kv_heads"], max_seq_len= config["max_seq_len"], d_ff= config["d_ff"], num_experts= config["num_experts"], top_k= config["top_k"], capacity_factor= config["capacity_factor"], rope_base= config["rope_base"] , dropout= config["dropout"], eps= float(config["eps"]))
                                     for _ in range(config["num_layers"])])
        
        self.final_norm = RMSNorm(config["d_model"], eps= float(config["eps"]))
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias= False)
        
        # Share the embedding weights
        self.lm_head.weight = self.token_embedding.weight

    
    def forward(self, X: Tensor, offset: int = 0, kv_cache: list[dict | None] | None = None) -> tuple[Tensor, list[dict | None], Tensor, Tensor, float]:
        
        if kv_cache is None:
            kv_cache = cast(list[dict | None], [None] * len(self.blocks))
            
        elif len(kv_cache) != len(self.blocks):
            raise ValueError("kv_cache must contain one entry per decoder block")
            
        # Token Embedding
        X = self.token_embedding(X) # (batch_size, seq_len, d_model)
        
        new_cache = []
        total_aux_loss, total_z_loss = X.new_zeros(()), X.new_zeros(())
        drop_rates = []
        
        # Forward pass
        for block, block_cache in zip(self.blocks, kv_cache):
            X, block_new_cache, aux_loss, z_loss, drop_rate = block(X, offset=offset, kv_cache=block_cache)
            
            # Update the cache
            new_cache.append(block_new_cache)
            
            # Save the metrics
            total_aux_loss += aux_loss
            total_z_loss += z_loss
            drop_rates.append(drop_rate)
        
        avg_drop_rate = sum(drop_rates) / len(drop_rates)
            
        # Apply the final norm
        X = self.final_norm(X)

        # Apply the linear head
        logits = self.lm_head(X) # (batch_size, seq_len, vocab_size)
        
        return (logits, new_cache, total_aux_loss, total_z_loss, avg_drop_rate)

    # nawfallllljma33sawwfallll
    # 3tih l3assiiiir
    # 3ab3ali 3awed dwz lpirmiiiiiiiii 

    