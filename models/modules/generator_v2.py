"""
    The Generator Architecture version-2

    > Need to have a proper structure for the creation of generator 
"""
# modules
import torch.nn as nn
from torch.nn.utils import spectral_norm
from utilsblocks import Conv1d, Linear, ConditionalBatchNorm
import torch

class GBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
