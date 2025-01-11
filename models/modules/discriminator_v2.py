'''
    Discriminator 
     Discriminates between the fake and real audio sample from the synthesizing and generation of the audio.
    
     There are 2 discrimator blocks
       In EATS there is no conditional Discriminators
'''
# importing the modeuls
import torch
from torch  import nn
from pathlib import Path
from utilsblocks import DownSampling, Conv1d, Linear

class Dblock(nn.Module):
    '''
        UnConditional Discriminator Block. It is same as the GAN-TTS discriminators.
    '''
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 downsample_factor):
        
        super(Dblock, self).__init__()
        in_channels = in_channels # N
        out_channels = out_channels # N * m  ( `m` is just factor for increase the channels)
        downsample_factor = downsample_factor # paddin will be estimated in the inthe individual layers

        
        self.stack1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=downsample_factor, stride=downsample_factor),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, dilation=2, padding="same")
        )
        self.residual_stack = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding="same"),
            nn.AvgPool1d(kernel_size=downsample_factor, stride=downsample_factor)
        )


    def forward(self, hidden_rep):
        return self.stack1(hidden_rep) + self.residual_stack(hidden_rep)



class SpectralDiscriminator(nn.Module):
    '''
        Spectral Discriminators, discriminates based on the spectrogram so that it will helps to learn proper speech properties and lingusitic properties.
    '''
    def __init__(self):
        super(SpectralDiscriminator, self).__init__()

    def forward(self, inputs):
        pass



class UnConditionalDiscriminator(nn.Module):
    '''
        Unconditional Discriminator contains 5 DBlocks.

        Args:
            in_channels (int) : Input channels for the layers
            out_channes (int) : Output channels from the each layers
            downsample_factor (int) : Downsampling the input by the factor.

    '''
    def __init__(self,
                 out_channels = (128, 256),
                 downsample_factors = (5,3)):
        super(UnConditionalDiscriminator, self).__init__()
        self.out_channels = out_channels
        self.downsample_factors = downsample_factors

        self.block1 = Dblock(in_channels=2, out_channels=64, downsample_factor=1)
        self.block2 = Dblock(in_channels=64, out_channels=self.out_channels[0], downsample_factor=self.downsample_factors[0])
        self.block3 = Dblock(in_channels=self.out_channels[0], out_channels=self.out_channels[1], downsample_factor=self.downsample_factors[1])
        self.block4 = Dblock(in_channels=self.out_channels[1], out_channels=256, downsample_factor=1)
        self.block5 = Dblock(in_channels=256, out_channels=256, downsample_factor=1)


    def forward(self, inputs):
        x = inputs.view(inputs.shape[0],2,-1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x



# ---- Sanity-checking-codes ----------------
if __name__ == "__main__":
    
    # configuration of the network or inputs
    batch_size = 2
    window_size = 480
    channels = 1

    # size of audio input
    window_input = torch.rand(size=(batch_size, channels, window_size))

    # NC_Disc_block = Dblock(
    #     in_channels=1,
    #     out_channels= 2,
    #     downsample_factor=5
    # )

    # disc_output = NC_Disc_block(window_input)


    # Sanity-checking the whole discriminator 
    UnCond_Disc = UnConditionalDiscriminator()
    unconditional_outpu = UnCond_Disc(window_input)
    print('The shape of the unconditional discriminator output :: ', unconditional_outpu.shape)
