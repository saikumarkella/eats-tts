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
    def __init__(self, in_channels, out_channels, dialte_rate=1, kernel_size=3, padding="same"):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              dilation=dialte_rate,
                              padding=padding)
        
        # nn.init.orthogonal_(self.conv)
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
        super().__init__()
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
        self.bn = nn.BatchNorm1d(num_features=num_features)
        self.scale_transformer = spectral_norm(nn.Linear(in_features=num_features, out_features=num_features))
        self.shift_transformer = spectral_norm(nn.Linear(in_features=num_features, out_features=num_features))
        self.extend_layer = spectral_norm(nn.Linear(in_features=256, out_features=num_features))

    def forward(self, inputs, concat_inputs):
        concat_inputs = self.extend_layer(concat_inputs)
        norms_inputs = self.bn(inputs)
        scale = (1 + self.scale_transformer(concat_inputs))[:, :, None] # extending the dims for broadcast
        shift = self.shift_transformer(concat_inputs)[:, :, None] # extending the dims for broadcast
        # norms_inputs = norms_inputs.permute(0, 2, 1)
        cbn = scale * norms_inputs + shift
        return cbn



# customized upsampling layers
class Upsampling(nn.Module):
    """
        upsampling input using the deconvolution or transposed convolution

        Args:
            in_channel (int) : Number of channels have in the input.
            out_channel (int) : Number of desired channels in output.
            kernel_size (int) : Size of the filter or kernel
            stride (int) : Amount of neurons it need to move
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernal_size,
                 stride):
        super(Upsampling, self).__init__()
        # configurations
        in_channel = in_channel
        out_channel = out_channel
        kernal_size = kernal_size
        stride = stride
        padding = (kernal_size - stride)//2   # padding cacluation , results of scaling factor of stride in output spatial dimensions

        upsample = nn.ConvTranspose1d(in_channels=in_channel, 
                                      out_channels=out_channel, 
                                      kernel_size=kernal_size, 
                                      stride=stride, 
                                      padding=padding) # layers inititalization
        nn.init.orthogonal_(upsample.weight) # weight initialization
        self.upsample_norm = spectral_norm(upsample) # normalizing weights to stabilize GAN trainings
                 
    def forward(self, inputs):
        return self.upsample_norm(inputs)
    
