"""
    Discriminator: 
        > Here we are using the Random Window Discriminators
        > maintaining computationally efficient
        > All 10 discriminator will be in similar architectures.
        > Also Architure is similar to Generator Architecture.


    File Contains:
        > DBlocks ( Discriminators )
        > CDBlocks ( Conditional Discriminators )
"""

# iniitializig modules 
import os
import torch
from torch import nn


# here we are defining the abstract 
class DBlock(nn.Module):
    """
        Discriminator Block:
            Penalizes the quality of the Generated Audio.
    """
    def __init__(self, in_channels, out_channels , kernels, stride, dilation_rate, downsample_factor):
        super().__init__()
        self.downsample = nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels, 
                                    kernel_size=kernels,
                                    padding="same")
        
        self.relu_1 = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernels,
                               padding="same"
                               )
        self.relu_2 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernels,
                               padding="same",
                               dilation=dilation_rate)
        
        # skip connection
        self.residual = nn.Sequential([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernels,
                      padding="same"),
            
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      stride=stride,
                      kernel_size=kernels,
                      padding="same")
            
        ])

    def forward(self, inputs):
        x = inputs
        
        # residual connection
        skip_connect  = self.residual(inputs)
        
        # normal forward-flow
        x = self.downsample(x)
        x = self.relu_1(x)
        x = self.conv1(x)
        x = self.relu_2(x)
        x = self.conv2(x)

        out = x + skip_connect
        return out
    



class CDBlock(nn.Module):
    """
        Conditional Discriminator Block:
            Penalizes the properties of audio were captured or not.
    """
    def __init__(self, in_channels, out_channels , kernels, stride, dilation_rate, downsample_factor):
        super().__init__()
        self.downsample = nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels, 
                                    kernel_size=kernels,
                                    padding="same")
        
        self.relu_1 = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernels,
                               padding="same"
                               )
        self.relu_2 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernels,
                               padding="same",
                               dilation=dilation_rate)
        
        # skip connection
        self.residual1 = nn.Conv1d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernels,
                                   padding="same")


        self.residual2 = nn.Sequential([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernels,
                      padding="same"),
            
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      stride=stride,
                      kernel_size=kernels,
                      padding="same")
            
        ])



    def forward(self, inputs, aligned_feat):
        x = inputs
        
        # residual connection
        skip_connect1 = self.residual1(aligned_feat)
        skip_connect2  = self.residual2(inputs)
        
        
        # normal forward-flow
        x = self.downsample(x)
        x = self.relu_1(x)
        x = self.conv1(x)

        # adopting intermediate features 
        x = x + skip_connect1 

        x = self.relu_2(x)
        x = self.conv2(x)

        out = x + skip_connect2
        return out
    



