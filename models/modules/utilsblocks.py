'''
    Utilities Blocks were used by the Generator Function.

    Generator Blocks were made of CONV1D.
'''
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Conv1d(nn.Module):
    '''
        Convolution 1D layer with spectral normalization, orthogonal Initialization, dialation rate 
        > Orthogonal initialization will be used for vanishing and exploding gradients.
        > Dialation Rate will be useful for to provide the larger receptive field with fewer parameters.
        > So that it will preserve the longer dependencies.

    '''
    def __init__(self, in_channels, out_channels, dialte_rate, kernel_size=3, padding="same"):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              dilation=dialte_rate,
                              padding=padding)
        
        nn.init.orthogonal_(self.conv)
        self.conv = spectral_norm(self.conv)

    def forward(self, inputs):
        a = self.conv(inputs)
        return a
    
class Linear(nn.Module):
    '''
        Designs of Linear Layers:
        > Linear layer initialization with the orthogoal and spectral normalization
        > The Spectral lnormalization is useful for stable training.
    '''
    def __init__(self, in_channels, out_channels) -> None:
        self.linear = nn.Linear(in_features=in_channels, out_features=out_channels)
        nn.init.orthogonal_(self.linear.weight)
        self.linear = spectral_norm(self.linear)

    def forward(self, inputs):
        return self.linear(inputs)
    


class ConditionalBatchNorm(nn.Module):
    """
        conditional Batch Normalization also with spectral normalization.
    """
    def __init__(self, num_features):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features=num_features)
        self.scale = spectral_norm(nn.Linear(in_features=num_features, out_features=num_features))
        self.shift = spectral_norm(nn.Linear(in_features=num_features, out_features=num_features))

    def forward(self, inputs, ccbn_condition):
        scale = self.scale(ccbn_condition)
        shift = self.shift(ccbn_condition)
        normalization = self.batch_norm(inputs)*scale + shift
        return normalization



