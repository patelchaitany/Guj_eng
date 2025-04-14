import sys
import os

sys.path.append(os.path.abspath("../"))
current_dir = os.path.dirname(os.path.abspath(__file__))  # Path to model/train.py

import torch
from torch import nn
from torch.nn import functional as F
import math as maths
from model.config import Config

# this is transformer encoderblock with attention and feedforward
# with residual connection and layer normalization
class EncAttention(nn.Module):

    def __init__(self,config) -> None:
        super(EncAttention, self).__init__()

        assert config.num_embeddings_encoder % config.nhead == 0
        self.attention = nn.Linear(config.num_embeddings_encoder,config.num_embeddings_encoder*3)
        self.proj = nn.Linear(config.num_embeddings_encoder,config.num_embeddings_encoder)
        self.proj.NANO_TRANS_E = 1.0
        self.nhead = config.nhead
        self.num_embeddings_encoder = config.num_embeddings_encoder

    def forward(self,x):
        # x : [batch_size,seq_len,embeddings]
        # attention formula : softmax((Q @ K.T)/sqrt(d_k)) @ V
        # Q,K,V : [batch,nhead,seq_len,embeddings//nhead]
        # output : [batch,nhead,seq_len,embeddings//nhead]

        B,T,C = x.size()

        qkv = self.attention(x) # [batch,seq_len,embeddings] @ [embeddings,embeddings*3] = [batch,seq_len,embeddings*3]

        q,k,v = qkv.split(self.num_embeddings_encoder,dim=-1) # [batch,seq_len,embeddings]

        q = q.view(B,T,self.nhead,self.num_embeddings_encoder//self.nhead).transpose(1,2) # [batch,nhead,seq_len,embeddings//nhead]
        k = k.view(B,T,self.nhead,self.num_embeddings_encoder//self.nhead).transpose(1,2) # [batch,nhead,seq_len,embeddings//nhead]
        v = v.view(B,T,self.nhead,self.num_embeddings_encoder//self.nhead).transpose(1,2) # [batch,nhead,seq_len,embeddings//nhead]

        y = F.scaled_dot_product_attention(q,k,v) # [batch,nhead,seq_len,embeddings//nhead]

        y = y.transpose(1,2).contiguous().view(B,T,C) # [batch,seq_len,embeddings]

        y = self.proj(y) # [batch,seq_len,embeddings]
        return y


class encoderMLP(nn.Module):

    def __init__(self,config)-> None:
        super(encoderMLP, self).__init__()

        self.linear1 = nn.Linear(config.num_embeddings_encoder,config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward,config.num_embeddings_encoder)
        self.gelu = nn.GELU()
        self.linear2.NANO_TRANS_E = 1.0
    def forward(self,x):
        # x : [batch_size,seq_len,embeddings]
        # output : [batch_size,seq_len,embeddings]
        # activation : GeLU

        x = self.gelu(self.linear1(x))
        x = self.linear2(x)
        return x

class EncoderBlock(nn.Module):

    def __init__(self,config) -> None:
        super(EncoderBlock,self).__init__()

        self.attention = EncAttention(config)
        self.mlp = encoderMLP(config)
        self.norm1 = nn.LayerNorm(config.num_embeddings_encoder)
        self.norm2 = nn.LayerNorm(config.num_embeddings_encoder)
    def forward(self,x):
        assert x is not None,print("Input is None")

        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class Encoder(nn.Module):

    def __init__(self,config) -> None:
        super(Encoder,self).__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
            wte = nn.Embedding(config.vocab_size,config.num_embeddings_encoder),
            h = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_encoder_layers)]),
            ln_f = nn.LayerNorm(config.num_embeddings_encoder)
            ))

        self.proj_c = nn.Linear(config.num_embeddings_encoder,config.num_embeddings_decoder)

        # positional embeddings

        div_term = torch.exp(torch.arange(0,self.config.num_embeddings_encoder,2)*(-1*maths.log(10000.0)/self.config.num_embeddings_encoder)).to(idx.device) # [embeddings//2]
        k = torch.arange(self.config.max_len).unsqueeze(1) # [T,1]
        pos_embd = torch.zeros(self.config.max_len,self.config.num_embeddings_encoder) # [T,embeddings]
        pos_embd[:,0::2] = torch.sin(k*div_term) # -> k*div_term : [T,1] * [embeddings//2] = [T,1] * [T,embeddings//2] = [T,embeddings//2]
        pos_embd[:,1::2] = torch.cos(k*div_term)
        pos_embd = pos_embd.unsqueeze(0) # [1,T,embeddings]
        self.register_buffer('pos_embd',pos_embd)

    def forward(self,idx):

        assert idx is not None,print("Input is None")
        B,T = idx.size()
        assert T <= self.config.max_len , f"Input sequence length {T} is greater than maximum length {self.config.max_len}"

        token_embd = self.transformer.wte(idx)
        x = token_embd + self.pos_embd[:,:T].requires_grad_(False) # [batch,seq_len,embeddings]

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        x = self.proj_c(x)
        return x

if __name__ == "__main__":
    config = Config(
        d_model = 64,
        nhead = 4,
        num_encoder_layers = 20,
        num_decoder_layers = 10,
        dim_feedforward = 128,
        dropout = 0.1,
        vocab_size = 100,
        max_len = 32,
        num_embeddings_decoder = 64,
        num_embeddings_encoder = 64
    )

    idx = torch.randint(0,100,(1,32))
    print(idx)
    encoder = Encoder(config)
    out = encoder(idx)
    print(out.shape)
    print(out)
