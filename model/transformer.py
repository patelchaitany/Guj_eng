import sys
import os

sys.path.append(os.path.abspath("../"))
current_dir = os.path.dirname(os.path.abspath(__file__))  # Path to model/train.py

from model.decoder import *
from model.encode import *
import torch
from torch import nn
from torch.nn import functional as F
import math as maths
from model.config import Config
from model.visionenc import VisionEncoder

class Transformer(nn.Module):

    def __init__(self,config:Config)-> None:

        super(Transformer,self).__init__()

        self.config = config

        # VisionEncoder
        self.visenc = VisionEncoder(config)


        # decoder

        self.wte = nn.Embedding(config.vocab_size,self.config.comman_embedding_dim)
        self.decoder = nn.ModuleDict(dict(
            ln_f0 = nn.LayerNorm(self.config.comman_embedding_dim),
            proj_cm = nn.Linear(self.config.comman_embedding_dim,self.config.num_embeddings_decoder),
            h = nn.ModuleList([DecoderBlock(self.config) if i % 2 == 0 else Decoder_Cross(self.config) for i in range(config.num_decoder_layers * 2)]),
            ln_f = nn.LayerNorm(config.num_embeddings_decoder),
            final_proj = nn.Linear(self.config.num_embeddings_decoder,self.config.comman_embedding_dim),
            ln_f2 = nn.LayerNorm(self.config.comman_embedding_dim)
        ))
        self.lm_head = nn.Linear(config.comman_embedding_dim,config.vocab_size,bias=False)

        # encoder
        self.encoder = nn.ModuleDict(
            dict(
            ln_f0 = nn.LayerNorm(self.config.comman_embedding_dim),
            proj_cm = nn.Linear(self.config.comman_embedding_dim,self.config.num_embeddings_encoder),
            h = nn.ModuleList([EncoderBlock(self.config) for _ in range(self.config.num_encoder_layers)]),
            ln_f = nn.LayerNorm(config.num_embeddings_encoder)
        ))

        self.proj_c = nn.Linear(config.num_embeddings_encoder,config.num_embeddings_decoder)

        self.wte.weight = self.lm_head.weight


        # positional embeddings
        div_term = torch.exp(torch.arange(0,self.config.comman_embedding_dim,2)*(-1*maths.log(10000.0)/self.config.comman_embedding_dim)) # [embeddings//2]
        k = torch.arange(self.config.max_len).unsqueeze(1) # [T,1]
        pos_embd = torch.zeros(self.config.max_len,self.config.comman_embedding_dim) # [T,embeddings]
        pos_embd[:,0::2] = torch.sin(k*div_term) # -> k*div_term : [T,1] * [embeddings//2] = [T,1] * [T,embeddings//2] = [T,embeddings//2]
        pos_embd[:,1::2] = torch.cos(k*div_term)
        pos_embd = pos_embd.unsqueeze(0) # [1,T,embeddings]
        self.register_buffer('pos_embd',pos_embd)

    def forward(self,x,y,is_image = False):
        # X is targate sequence and y is source sequence
        Bx,Tx = x.size()
        if not is_image:
            By,Ty = y.size()
            # encoder
            y = self.wte(y) # [B,T] -> [B,T,comman_embeddings]

            y = y + self.pos_embd[:,:Ty].requires_grad_(False) # [B,T,comman_embeddings]
            y = self.encoder.ln_f0(y) # [B,T,comman_embeddings]
            y = self.encoder.proj_cm(y) # [B,T,comman_embeddings] -> [B,T,encoder_embeddings]
            for block in self.encoder.h:
                y = block(y) # [B,T,encoder_embeddings]

            y = self.encoder.ln_f(y) # [B,T,embeddings]
            y = self.proj_c(y) # [B,T,encoder_embeddings] -> [B,T,decoder_embeddings]

        if is_image:
            By,Cy,Hy,Wx = y.size()
            y = self.visenc(y) # [B,T,Image_encoding]
        # decoder

        x = self.wte(x) # [B,T] -> [B,T,comman_embeddings]
        x = x + self.pos_embd[:,:Tx].requires_grad_(False) # [B,T,comman_embeddings]
        x = self.decoder.ln_f0(x) # [B,T,comman_embeddings]
        x = self.decoder.proj_cm(x) # [B,T,comman_embeddings] -> [B,T,decoder_embeddings]
        for block in self.decoder.h:
            if isinstance(block,Decoder_Cross):
                x = block(x,y) # [B,T,decoder_embeddings]
            else:
                x = block(x) # [B,T,decoder_embeddings]

        x = self.decoder.ln_f(x) # [B,T,decoder_embeddings]
        x = self.decoder.final_proj(x) # [B,T,decoder_embeddings] -> [B,T,comman_embeddings]
        x = self.decoder.ln_f2(x) # [B,T,comman_embeddings]
        x = self.lm_head(x) # [B,T,comman_embeddings] -> [B,T,vocab_size]

        return x

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        num_embeddings_encoder = 64,
        comman_embedding_dim = 20,
    )
    # for the testing perpouse of text

    model = Transformer(config).to(device)
    x = torch.randint(0,100,(4,32)).to(device)
    y = torch.randint(0,100,(4,32)).to(device)
    img = torch.rand((4,3,256,256)).to(device)
    out = model(x,img,is_image = True)

    print(out.shpae)

    # Calculate loss
    criterion = nn.CrossEntropyLoss()
    labels = torch.randint(0, 100, (4, 32)).to(device)  # Create random labels for demonstration
    loss = criterion(out.view(-1, config.vocab_size), labels.view(-1))

    # Perform backward pass
    loss.backward()

    # Print loss value
    print(f"Loss: {loss.item()}")

    # for text only

    out = model(x,y)
    loss = criterion(out.view(-1,config.vocab_size),labels.view(-1))
    loss.backward()

    print(f"Loss text : {loss.item()}")
    # Optionally, we could update weights
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # with torch.autocast(de)
    # optimizer.step()
    # optimizer.zero_grad()
