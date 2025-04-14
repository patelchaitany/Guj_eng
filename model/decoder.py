import sys
import os

sys.path.append(os.path.abspath("../"))
current_dir = os.path.dirname(os.path.abspath(__file__))  # Path to model/train.py

import torch
from torch import nn
from torch.nn import functional as F
import math as maths
from model.config import Config

class DecAttention(nn.Module):

    def __init__(self,config):
        super(DecAttention,self).__init__()
        self.config = config
        self.attention = nn.Linear(config.num_embeddings_decoder,config.num_embeddings_decoder*3)
        self.proj_c = nn.Linear(config.num_embeddings_decoder,config.num_embeddings_decoder)
        self.proj_c.NANO_TRANS_D = 1.0
        self.nhead = config.nhead
        self.num_embeddings_decoder = config.num_embeddings_decoder

    def forward(self,x):

        B,T,C = x.size()
        qkv = self.attention(x) # B,T,3C
        q,k,v = torch.split(qkv,C,dim= -1)
        q = q.view(B,T,self.nhead,C//self.nhead).transpose(1,2)
        k = k.view(B,T,self.nhead,C//self.nhead).transpose(1,2)
        v = v.view(B,T,self.nhead,C//self.nhead).transpose(1,2)
        y = F.scaled_dot_product_attention(q,k,v,is_causal = True) # B,nhead,T,C//nhead
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.proj_c(y)
        return y

class CrossAttention(nn.Module):

    def __init__(self,config):
        super(CrossAttention,self).__init__()


        # attention formula : softmax((Q @ K.T)/sqrt(d_k)) @ V
        # In cross attention Q is decoder and K,V is encoder
        # In Q = [n,C1], K = [m,C], V = [m,C] where n is tragate sqeuence length and m is source sequence length
        # Q@K.T = [n,C] @ [C,m] = [n,m]
        # [n,m] @ [m,C] = [n,C]
        # [n,C -> [n,C1]

        self.config = config
        self.qery = nn.Linear(config.num_embeddings_decoder,config.num_embeddings_decoder)
        self.kv = nn.Linear(config.num_embeddings_encoder,config.num_embeddings_decoder*2)

        self.proj_c = nn.Linear(config.num_embeddings_decoder,config.num_embeddings_decoder)

        self.proj_c.NANO_TRANS_D = 1.0
        self.nhead = config.nhead
    def forward(self,x,encoder_output):

        B,T,C = x.size()
        B,T1,C1 = encoder_output.size()

        q = self.qery(x) # B,T,C
        kv = self.kv(encoder_output) # B,T1,2*C
        k,v = torch.split(kv,C,dim = -1)
        q = q.view(B,T,self.nhead,C//self.nhead).transpose(1,2)
        k = k.view(B,T1,self.nhead,C//self.nhead).transpose(1,2)
        v = v.view(B,T1,self.nhead,C//self.nhead).transpose(1,2)
        y = F.scaled_dot_product_attention(q,k,v,is_causal = False) # B,nhead,T,C//nhead
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.proj_c(y)

        return y

class DecoderMLP(nn.Module):

    def  __init__(self,config) -> None:

        super(DecoderMLP,self).__init__()

        self.config = config

        self.mlp = nn.Sequential(
            nn.Linear(config.num_embeddings_decoder,config.dim_feedforward),
            nn.GELU(),
        )
        self.proj_c = nn.Linear(config.dim_feedforward,config.num_embeddings_decoder)
        self.proj_c.NANO_TRANS_D = 1.0

    def forward(self,x):
        y = self.mlp(x)
        y = self.proj_c(y)
        return y

class DecoderBlock(nn.Module):

    def __init__(self,config):

        super(DecoderBlock,self).__init__()

        self.config = config
        self.dec_attention = DecAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(config.num_embeddings_decoder)
        self.layer_norm2 = nn.LayerNorm(config.num_embeddings_decoder)
        self.decoder_mlp = DecoderMLP(self.config)

    def forward(self,x):

        y = self.layer_norm1(x)
        y = self.dec_attention(y)
        y = x + y
        y = self.layer_norm2(y)
        y = x + self.decoder_mlp(y)

        return y

class Decoder_Cross(nn.Module):

    def __init__(self,config):

        super(Decoder_Cross,self).__init__()

        self.config = config

        self.cross_attn = CrossAttention(config)

        self.layer_norm1 = nn.LayerNorm(config.num_embeddings_decoder)
        self.layer_norm2 = nn.LayerNorm(config.num_embeddings_decoder)
        self.layer_norm3 = nn.LayerNorm(config.num_embeddings_encoder)
        self.decoder_mlp = DecoderMLP(config)

    def forward(self,x,y):
        # the X is the source sequence and Y is the target sequence
        y = self.layer_norm3(y)
        x = self.layer_norm1(x)
        z = self.cross_attn(x,y)
        z = x + z
        z = self.layer_norm2(z)
        z = z + self.decoder_mlp(z)
        return z
