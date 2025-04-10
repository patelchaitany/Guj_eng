import torch
from torch import nn
from torch.nn import functional as F
import math as maths
from model.config import Config
from STN import STN

class VisionEncoder(nn.Module):

    def __init__(self, config):
        super(VisionEncoder, self).__init__()

    def forward(self,x):
        pass
