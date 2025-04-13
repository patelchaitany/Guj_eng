import sys
import os

sys.path.append(os.path.abspath("../"))
current_dir = os.path.dirname(os.path.abspath(__file__))  # Path to model/train.py

import torch
from torch import device, nn
from torch.nn import functional as F
import math as maths
from model.config import Config
from model.STN import STN
from transformers import CLIPVisionModel, CLIPImageProcessor
import torchvision.transforms as T


class VisionEncoder(nn.Module):

    def __init__(self, config):
        super(VisionEncoder, self).__init__()
        self.config = config
        self.clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_encoder.eval()
        for p in self.clip_encoder.parameters():
            p.requires_grad = False
        self.vision_output_dim = self.clip_encoder.config.hidden_size
        self.proj = nn.Linear(self.vision_output_dim,self.config.num_embeddings_encoder)
        self.stn = STN(self.config.control_points,(224,224),3)
        self.normalize = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        self.ly_n = nn.LayerNorm(self.vision_output_dim)
    def forward(self,x):
        y = self.stn(x)
        y = self.normalize(y)
        y = self.clip_encoder(pixel_values = y,output_hidden_states=False)
        y = y.last_hidden_state
        # print(self.vision_output_dim)
        y = self.ly_n(y)
        y = self.proj(y)
        return y

if __name__ == "__main__":
    config = Config(
       vocab_size = 100,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = torch.rand((1, 3, 256, 256)).to(device)
    vis_enc = VisionEncoder(config)
    y = vis_enc(img)
    print(y.shape)
