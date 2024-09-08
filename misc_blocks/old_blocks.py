import torch
from torch import nn


#---------------------------------------------------------------
#               Aligner Network.
#---------------------------------------------------------------
class Aligner(nn.Module):

    """
        Aligner Network Architecutre and their components.
        
        Components:
            1. Embeddings (a raw characters embeddings / phonemes embeddings).
            2. Base convolution and Hidding representation network.
            3. Token lengths
            4. calculating centers
            5. Getting an interpolated weights
            6. Getting the audio-aligned Representation.
    """
    def __init__(self, n_vocab, embed_dims, num_layers, in_channel, out_channels, kernel_size, dilation_rate):
        super().__init__()
        self.embeds = nn.Embedding(num_embeddings= n_vocab,
                                   embedding_dim= embed_dims
                                   )
        
        self.unaligned_net  = nn.ModuleList([])
        for _ in range(10):
            token_rep_net = UnalignedTokenReprentation(num_layers= num_layers,
                                      in_channle= in_channel,
                                      out_channels= out_channels,
                                      kernel_size= kernel_size,
                                      dilation_rate= dilation_rate)
            self.unaligned_net.append(token_rep_net)
            
        self.token_length = MLPTokenLength()

    def forward(self, inputs):
        tokens = self.embeds(inputs)
        hidden_rep = self.token_rep_net(tokens)
        lengths, centers = self.token_length(hidden_rep)
        interpolate_wghs = self.interpolation_weights(centers, lengths)
        audio_aligns = self.audio_aligned_representation(hidden_rep, interpolate_wghs)
        return audio_aligns


    def interpolation_weights(self, centers, audio_timestamps, temperature=10.0):
        """ 
            Interpolating Token representation into the Audio-Aligned Representation.

            Steps:
                1. Get the centers and audio timestamps.
                2. Need to Compute the distance between centers and audio times.
                3. Compute the softmax to get the weights interpolation.

            Args:
                centers (torch.Tensor) : Token centers for the text/phonemes representations
                audio_timestamps (torch.Tensor) : Audio aligned representations.
        """
        audio_timestamps = audio_timestamps.unqueeze(-1)
        distance_diffs = torch.square(centers - audio_timestamps)
        alings = (-(temperature)**-2) * (distance_diffs)
        weights = torch.softmax(alings, dim=-1)
        return weights


    def audio_aligned_representation(self, hidden_state, interpolate_weights):
        """
            Audio aligned representation:
                It was obtained from the hidden representation and interpolation weights

            Dimension Analysis:
                hiddenstates = Dims( B, )
        """
        alines = torch.matmul(hidden_state, interpolate_weights)
        alinements = torch.sum(alines, dim=-1)
        return alinements
