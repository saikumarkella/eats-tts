"""
    Building Blocks for the Aligner Network
"""
import torch
from torch import nn


class SpeakerEmbeddings(nn.Module):
    """  Speaker Embeddings
         Getting the embedding for the every speaker , It will useful for at the time of Multi-speaker Audio Generation.

        Args:
            num_speakers (int) : Total number of speakers
            speaker_dims (int) : Speaker embedding output size.
    
        Returns:
            embeddings (torch.Tensor) : A muti-dimensional speaker embeddings.
    """

    def __init__(self, num_speakers, speaker_dims=128):
        self.embeds = nn.Embedding(num_embeddings= num_speakers, embedding_dim=speaker_dims)
    
    def forward(self, speaker_ids):
        embeddings = self.embeds(speaker_ids)
        return embeddings


# Getting the noise layer 
def generate_noise(batch_size, noise_dims = 128):
    """ Generating Noise for the Aligner. This noise was generating from the Gaussina Distribution.
    """
    noise = torch.normal(mean=0, std=1, size=(batch_size, noise_dims))
    return noise


class ConditionalBatchNorm(nn.Module):
    """
        A class conditional Batch Normalization.
    """
    def __init__(self, seq_len=200):
        super(ConditionalBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features=256)
        self.scale_transformer = nn.utils.spectral_norm(nn.Linear(in_features=256, out_features=seq_len))
        self.shift_transformer = nn.utils.spectral_norm(nn.Linear(in_features=256, out_features=seq_len))

    def forward(self, inputs, concat_inputs):
        norms_inputs = self.bn(inputs)
        scale = (1 + self.scale_transformer(concat_inputs))[:, None, :] # extending the dims for broadcast
        shift = self.shift_transformer(concat_inputs)[:,None, :] # extending the dims for broadcast
        # norms_inputs = norms_inputs.permute(0, 2, 1)
        cbn = scale * norms_inputs + shift
        return cbn


# tokenzier skip connection block
class UnalignedBlock(nn.Module):
    '''
        Token Representation Block for the 
    
    '''
    def __init__(self, out_channels, kernel_size, dilation_rate, seq_len):
        super(UnalignedBlock, self).__init__()

        self.relu_1 = nn.ReLU()
        self.mask_conv1 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation_rate[0], padding="same")
        self.relu_2 = nn.ReLU()
        self.mask_conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation_rate[1], padding="same")
        self.cond_batchnorm = ConditionalBatchNorm(seq_len= seq_len)

    def forward(self, inputs, ccbn_condition):
        inputs = inputs
        skip_input = inputs
        cond_batch = self.cond_batchnorm(inputs, ccbn_condition)
        x = self.relu_1(cond_batch)
        x = self.mask_conv1(x)
        x = self.relu_2(self.cond_batchnorm(x, ccbn_condition))
        x = self.mask_conv2(x)
        x += skip_input
        return x
    

class UnalignedTokenReprentation(nn.Module):
    """ Generating a Token Representation from the phonemes or character token sequences.
        It is a stack of dilated convolutions + batch normalization + ReLU activation.

        Args:
            kernel_sizes (List[int], int) : Size of the kernel Filters.
            seq_length (int) : total input sequence length.
            out_channels (int) : output channels for the 

    """
    def __init__(self, kernel_sizes, seq_length, out_channels, dilation_rates = None):
        super(UnalignedTokenReprentation, self).__init__()
        self.sub_token_blocks = nn.ModuleList([])
        for dil_rate in dilation_rates:
            lay = UnalignedBlock(out_channels=out_channels,
                                kernel_size=kernel_sizes,
                                dilation_rate=dil_rate,
                                seq_len=seq_length)
            self.sub_token_blocks.append(lay)


    def forward(self, inputs, speaker_embeddings, noise):
        ccbn_condition = torch.concatenate([speaker_embeddings, noise], dim=-1)
        x = inputs
        for lay in self.sub_token_blocks:
            x = lay(x, ccbn_condition)
        return x


#-----------------------------------------------------------------------
#                          SANITY CHECKINGS 
#-----------------------------------------------------------------------
# A sanity checking of the unaligned blocks.
if __name__ == "__main__":

    # Simulating the inputs for the unaligned representation network.

    batch = 1
    seq_len = 400
    out_channel = 256
    kernel_size = 3
    dilation_rate = (2,3)

    speaker_embed = torch.rand(size= (batch, 128)) # (batch, 128)
    inputs_embeds =  torch.rand(size = (batch, seq_len, 256)) # (batch , seq , 256)
    noise  = torch.rand(size = (batch, 128)) # (batch, 128)
 
    # making the inputs have the channel first senorie.
    inputs_embeds = torch.permute(inputs_embeds, dims=(0, 2, 1))

    # Concatenating the speaker embeddings and noise embeddings
    ccbn_condition  = torch.concatenate([speaker_embed, noise], dim = -1)
    
    unalign = UnalignedBlock(out_channels = out_channel, kernel_size=kernel_size, dilation_rate=dilation_rate, seq_len=seq_len)
    outs = unalign(inputs_embeds,ccbn_condition)
    print(outs.shape)

    




# # sanity checkings
# if __name__ == "__main__":
#     # checking the class conditional spectral Normalization.
#     cbn = ConditionalBatchNorm()

#     # creating dummy inputs.
#     batch = 4
#     seq_len = 200
#     dim = 256

#     input_ = torch.rand(size=(batch, seq_len, dim))
#     input_ = input_.permute(0, 2, 1)
#     cbn_input = torch.rand(size=(batch, dim))
#     outs = cbn(input_, cbn_input)
#     print(outs.shape)

