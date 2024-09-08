"""
    A aligner is kind of a FeedForward Network. It was a audio-aligned representation.

    Functionalities:
        1. input (X) = {x1, x2, x3 ... xn} => These token sequence can be raw characters or phonemes.
        2. Compute the token representation. h = f(x, z, s) 
        3. Predict the length of each input token individually. ln = g(h, z, s)
        4. Compute the predicted token end positions.
        5. Compute the token center position.
        6. Based on the token center position we can interprete from token representation to a audio-aligned representation.
                i. To get this ,compute the interpolation weights for the token representation.
                ii. 

"""

# modules
import torch
from torch import nn
from torch import functional as F
from modules.blocks import SpeakerEmbeddings, generate_noise, UnalignedTokenReprentation, ConditionalBatchNorm


#--------------------------------------------------------
# module2 : Token Length prediction
#--------------------------------------------------------
class TokenLength(nn.Module):
    """
        Token Length Representation of the unaligned represenation
        Mapping to the Predicted Token Lengths.

        Args:
            seq_length (int) : Total sequence length.

    """
    def __init__(self, seq_length):
        super(TokenLength, self).__init__()
        self.condBatchNorm1 = ConditionalBatchNorm(seq_len=seq_length)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)
        self.condBatchNorm2 = ConditionalBatchNorm(seq_len=seq_length)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
        self.relu3 = nn.ReLU()

    def forward(self, inputs, ccbn_condition):
        x = self.relu1(self.condBatchNorm1(inputs, ccbn_condition))
        x = self.conv1(x)
        x = self.relu2(self.condBatchNorm2(x, ccbn_condition))
        x = self.conv2(x)
        token_lengths = self.relu3(x[:,0,:])
        return token_lengths
    



#---------------------------------------------------------------
#                       Aligner Network - Version:2
#---------------------------------------------------------------
class Aligner(nn.Module):
    """
        Aligner Architecture :

            1. Input Embeddings
            2. Unaligned Blocks
            3. Token_length
            4. Center positions


        Args:
            token_vocab_size (int) : Size of the Vocabulary for the input text.
            token_dims (int) : total dims of the token representation.
            lengths (List[int]) : The length of the each sequence.
            num_speakers (int) : The number of speaker ids in entire dataset.
            speaker_dims (int) : A output dimensional for the speakers.
            out_sequence_length (int) : The length of output sequence at 200Hz at time of training , and 400Hz at time of inference.
            temperature (float) : A float value for the softmax.

        Returns:
            alined-features : Audio aligned features that will be fed into the decoder.
            alined-lengths : length of the audio-alined features.
    
    """
    def __init__(self, 
                 token_vocab_size,
                 token_dims,
                 unalign_kernel,
                 seq_length,
                 unalign_out_channels,
                 unalign_dialation_rates,
                 lengths, 
                 out_sequence_length, 
                 temperature):
        super(Aligner, self).__init__()
        self.embedded_tokens = nn.Embedding(token_vocab_size, token_dims)
        self.unalings =  nn.ModuleList([])
        for _ in range(10):
            b = UnalignedTokenReprentation(unalign_kernel, seq_length, unalign_out_channels, unalign_dialation_rates)
            self.unalings.append(b)

        self.tokenLength = TokenLength(seq_length=seq_length)
        

    def forward(self, inputs, speaker_embeddings, noise):

        pass

