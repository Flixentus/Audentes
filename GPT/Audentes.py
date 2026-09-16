import torch.nn as nn 
import torch 
from Decoder import DecoderBlock
import json

with open("config.json", "r") as f:
    config = json.load(f)


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.d_model = d_model
        self.vocab = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder = nn.ModuleList([DecoderBlock(d_model, num_heads = config["num_heads"], num_kv_heads=config["num_kv_heads"], max_seq_len=config["max_seq_len"], d_ff=config["d_ff"], num_experts=config["num_experts"], top_k=config["top_k"], capacity_factor=config["capacity_factor"], dropout=config["dropout"]) for _ in range(config["num_layers"])])

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_tokens):
        x = self.embedding(tgt_tokens)

        seq_len = tgt_tokens.shape[1]

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device),
            diagonal=1) 
        causal_mask[:tgt_tokens.shape[1], :tgt_tokens.shape[1]] = 0

        for block in self.decoder:
            x = block(x, mask=causal_mask)


        logits = self.output_proj(x)
        return logits

    #nawfallllljma33sawwfallll 

    