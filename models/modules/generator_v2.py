"""
    The Generator Architecture version-2

    > Need to have a proper structure for the creation of generator 
"""
# modules
import torch.nn as nn
from torch.nn.utils import spectral_norm
from utilsblocks import Conv1d, Linear, ConditionalBatchNorm, Upsampling
import torch


class GBlock(nn.Module):
    '''
        Discription : Generation will upsamples the audio aligned features to the wavform
    '''
    def __init__(self):
        super(GBlock, self).__init__()

        # stack 1
        self.bn1 = ConditionalBatchNorm()
        self.stack1 = nn.Sequential(
            nn.ReLU(),
            Upsampling(),
            Conv1d()    
        )

        # stack 2
        self.bn2 = ConditionalBatchNorm()
        self.stack2 = nn.Sequential(
            nn.ReLU(),
            Conv1d()
        )

        # residual stack
        self.residual_stack = nn.Sequential(
            Upsampling(),
            Conv1d()
        )

        # stack 3
        self.bn3 = ConditionalBatchNorm()
        self.stack3 = nn.Sequential(
            nn.ReLU(),
            Conv1d()
        )

        # stack 4
        self.bn4 = ConditionalBatchNorm()
        self.stack4 = nn.Sequential(
            nn.ReLU(),
            Conv1d()
        )

    def forward(self, inputs , ccbn_condition):
        
        # sub_block 1
        x1 = inputs
        x = self.bn1(inputs, ccbn_condition)
        x = self.stack1(x)
        x = self.bn2(x, ccbn_condition)
        x = self.stack2(x)

        # residual block
        res_out = self.residual_stack(x1)
        x = torch.concatenate([res_out, x])
        x2 = x
        
        # sub-block2 
        x = self.bn3(x, ccbn_condition)
        x = self.stack3(x)
        x = self.bn4(x, ccbn_condition)
        x = self.stack4(x)

        out = torch.concatenate([x2, x])
        return out



class Generator(nn.Module):
    '''
        Generator which consits of the GBlocks (For generating the modules)

        Generatie

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self):
        pass




