import torch.nn as nn
import torch

class MHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.d_k = d_model // num_heads 
        #------------
        # RoPE na9ssa
        #------------

    def forward(self, x, mask=None, use_rope = True):
        batch_size = x.size(0)

        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        q_seq_len = x.shape[1]
        
        Q = Q.view(batch_size, q_seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, q_seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, q_seq_len, self.num_heads, self.d_k)

        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        if use_rope:
            Q = self.rope(Q)
            K = self.rope(K)

        scores = Q @ K.transpose(-2, -1) / self.d_k**0.5 #lformule

        attn_weights = torch.softmax(scores, dim=-1) # softmax lkola ra9em QK product ftable
        output = attn_weights @ V
        output = output.transpose(1,2).contiguous.view(batch_size, q_seq_len, self.d_model) #output irje3 l original dimension 9bel QK^T
        output = self.W_o(output) # dik le3ba dial concat heads

        return output

class MLP(nn.Module):

    def __init__(self, *args, **kwargs):
        pass

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MHA(d_model, num_heads)
        pass


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_decoder_layers):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        pass
