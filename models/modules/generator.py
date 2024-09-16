'''
    GAN-TTS Generator functions.

    > Consists of 7 Diluted Convolution blocks.
    > Here we used the diluted convolution because need to get the larger receptive field.
'''
# importing modules
import torch.nn as nn
from torch.nn.utils import spectral_norm
from utilsblocks import conv1d, Linear, ConditionalBatchNorm
import torch

class GBlock(nn.Module):

    """
        Generation Block made up of the series of Diluted Convolution Blocks.
        In this block we need to Upsample Layer + then Conv1D 
    """
    def __init__(self, ):
        super().__init__()
        # residualBlock 1
        self.batchnorm1 = ConditionalBatchNorm()
        self.relu1 = nn.ReLU()
        self.upsample1 = nn.ConvTranspose1d()
        self.conv1 = conv1d()
        self.batchnorm2 = ConditionalBatchNorm()
        self.relu2 = nn.ReLU()
        self.conv2 = conv1d()

        # bottleneck
        self.upsample_BN= nn.ConvTranspose1d()
        self.conv_BN = conv1d()
        
        # residualBlock 2
        self.batchnorm3 = ConditionalBatchNorm()
        self.relu3 = nn.ReLU()
        self.conv3 = conv1d()
        self.batchnorm4 = ConditionalBatchNorm()
        self.relu4 = nn.ReLU()
        self.conv4 = conv1d()

    def forward(self, inputs , ccbn_condition):
        # residualBlock1 
        input_features = inputs
        x = self.batchnorm1(inputs, ccbn_condition)
        x = self.relu1(x)
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.batchnorm2(x, ccbn_condition)
        x = self.relu2(x)
        x = self.conv2(x)

        # bottle neck
        feat = self.upsample_BN(input_features)
        feat = self.conv_BN(feat)
        concate_feat = torch.concatenate([x, feat])

        # residual block2
        res_connection = concate_feat
        x = self.batchnorm3(res_connection, ccbn_condition)
        x = self.relu3(x)
        x = self.conv3(x)
        x = self.batchnorm4(x, ccbn_condition)
        x = self.relu4(x)
        x = self.conv4(x)

        output = torch.concatenate([x, res_connection])
        return output
