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
            5. Getting logits 
            6. Calculating aligned weights (which are interpolation weights)
            7. Calculating audio-aligned features.


        Args:
            token_vocab_size (int) : Size of the Vocabulary for the input text.
            token_dims (int) : total dims of the token representation.
            unalign_kernels (int) : A kernel size for the network which gives the unaligned representation
            seq_length (int) : A sequence length of all the example sequence.
            unalign_out_channels (int) : A output channel for 1D Convolutions
            unalign_dialation_rates (List[tuples]) : 3 sequential dialtion rates for the 1Conv of  unaligned represenations
            lengths (List[int]) : True length od each sequence before padding.
            out_sequence_length (int) : The length of output sequence at 200Hz at time of training , and 400Hz at time of inference.
            temperature (float) : A float value for the softmax.

        Returns:
            alined-features : Audio aligned features that will be fed into the decoder.
            alined-lengths : length of the audio-alined features.

        Dimensional Returns:
            aligned_features : (Batch, out_sequence, hidden_dims) ==> (Batch, 6000, 256)
            aligned_lengths : (Batch, sequence_length) ==> (batch, 600)
    
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
        self.output_seq_length = out_sequence_length
        self.temperature = temperature
        

    def forward(self, inputs, speaker_embeddings, noise):
        x = self.embedded_tokens(inputs)
        x = torch.permute(x, dims=(0,2,1))
        ccbn_conditions = torch.concatenate([speaker_embeddings, noise], dim=-1)

        for unalign_layer in self.unalings:
            x = unalign_layer(x, ccbn_conditions)
        unaligned_features = x

        # calculating the token lengths, token ends token token centers
        token_lengths = self.tokenLength(x, ccbn_conditions)
        token_ends = torch.cumsum(token_lengths, dim=-1)
        token_centers = token_ends - (token_lengths/2)
        aligned_lengths = token_ends

        # Computing the output grid for projecting the unaligned features to the aligned features.
        out_pos = torch.arange(start=0, end=self.output_seq_length, dtype=torch.float32)
        out_pos = torch.Tensor.repeat(out_pos, repeats=(x.shape[0],1))
        # Calculating the logits
        token_centers = token_centers[:,None, :]
        out_pos = out_pos[:,:, None]
        diff = token_centers - out_pos # dims = (N, 6000, 600) after difference
        logits = -(diff**2 / self.temperature)
        # need to calculate the masked logits  @ Actually weights need to find from the masked logits but here we are finding from the logits.
        weights = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)

        # Batch-Wise Matrix Multiplication between @weights and @unaligned_features
        # Changing the shape of tensor using permut function.
        unaligned_features = torch.permute(unaligned_features, dims=(0,2,1))
        aligned_features = torch.bmm(weights, unaligned_features)
        return aligned_features, aligned_lengths


if __name__ == "__main__":
    token_vocab_size = 90
    token_dims = 256
    unalign_kernel = 3
    seq_length = 600
    unalign_out_channels = 256
    unalign_dialation_rates = [(1,2), (4,8), (16, 32)]
    lengths = torch.tensor([])
    out_sequence_length = 6000
    temperature = 10.0
    batch = 4

    # speaker embeddings and noise embeddings 
    num_speakers = 6
    speaker_dims = 128
    noise_dims = 128

    # preparing the input_sequence, Speaker Embeddings and Noise.
    input_tokens = torch.randint(low=0, high=token_vocab_size, size=(batch, seq_length))
    speakers = torch.randint(low=0, high=num_speakers, size=(batch,))
    speakerEmbedClass = SpeakerEmbeddings(num_speakers=num_speakers)
    speaker_embeds = speakerEmbedClass(speakers)
    noise = generate_noise(batch_size=batch, noise_dims=128)

    print("|> Shape of the Speaker Embeddings : ", speaker_embeds.shape)
    print("|> Shape of the Noise : ", noise.shape)
    

    alinger = Aligner(token_vocab_size=token_vocab_size,
                      token_dims=token_dims,
                      unalign_kernel=unalign_kernel,
                      seq_length=seq_length,
                      unalign_out_channels=unalign_out_channels,
                      unalign_dialation_rates=unalign_dialation_rates,
                      lengths=lengths,
                      out_sequence_length=out_sequence_length,
                      temperature=temperature)
    
    aligned_features, aligned_timestamps = alinger(input_tokens, speaker_embeds, noise)

    print("Shape of the Aligned Features are :: ", aligned_features.shape)
    print("shape of the Aligned timestamps :: ", aligned_timestamps.shape)