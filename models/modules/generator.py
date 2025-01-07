'''
    GAN-TTS Generator functions.

    > Consists of 7 Diluted Convolution blocks.
    > Here we used the diluted convolution because need to get the larger receptive field.
'''
# importing modules
import torch.nn as nn
from torch.nn.utils import spectral_norm
from utilsblocks import Conv1d, Linear, ConditionalBatchNorm
import torch



class GBlock(nn.Module):

    """
        Generation Block made up of the series of Diluted Convolution Blocks.
        In this block we mneed to Upsample Layer + then Conv1D 

        Network Architecture:
            - 2 Residual Connections 
            - 1 skip network 

        Args:
            kernel_size (List or Int): Size of the filter
            output_features (List[int]): output features
            input_features (List[int]): input features

    """
    def __init__(self,
                 kernel_size = 3, 
                 output_features = 256, 
                 input_features=256, 
                 dilation_rate = [1,2,4,8],
                 upsample_rate = 1,
                 output_padding=0):
        
        super().__init__()
        # residualBlock 1
        self.batchnorm1 = ConditionalBatchNorm(input_features)
        self.relu1 = nn.ReLU()
        self.upsample1 = nn.ConvTranspose1d(in_channels=input_features, out_channels=input_features, kernel_size=kernel_size, stride=upsample_rate, padding=1, output_padding=output_padding)
        self.conv1 = Conv1d(in_channels=input_features, out_channels=output_features, dialte_rate=dilation_rate[0], kernel_size=kernel_size)
        self.batchnorm2 = ConditionalBatchNorm(output_features)
        self.relu2 = nn.ReLU()
        self.conv2 = Conv1d(in_channels=output_features, out_channels=output_features, dialte_rate=dilation_rate[1])

        # bottleneck
        self.upsample_BN= nn.ConvTranspose1d(in_channels=input_features, out_channels=input_features, kernel_size=kernel_size, stride=upsample_rate, padding=1, output_padding=output_padding)
        self.conv_BN = Conv1d(in_channels=input_features, out_channels=output_features, dialte_rate=1, kernel_size=kernel_size)
        
        # residualBlock 2
        self.batchnorm3 = ConditionalBatchNorm(output_features)
        self.relu3 = nn.ReLU()
        self.conv3 = Conv1d(in_channels=output_features, out_channels=output_features, dialte_rate=dilation_rate[2], kernel_size=kernel_size)
        self.batchnorm4 = ConditionalBatchNorm(output_features)
        self.relu4 = nn.ReLU()
        self.conv4 = Conv1d(in_channels=output_features, out_channels=output_features, dialte_rate=dilation_rate[3], kernel_size=kernel_size)


        # Projecting the input condition by linear layers
        self.linear1 = Linear(in_channels=256, out_channels=input_features)
        self.linear2 = Linear(in_channels=256, out_channels=output_features)
        self.linear3 = Linear(in_channels=256, out_channels=output_features)
        self.linear4 = Linear(in_channels=256, out_channels=output_features)


    def forward(self, inputs , ccbn_condition):
        # residualBlock1 
        input_features = inputs
        print("\t in : ", input_features.shape)
        x = self.batchnorm1(inputs, self.linear1(ccbn_condition))
        x = self.relu1(x)
        x = self.upsample1(x)
        print("\t out : ", x.shape)
        x = self.conv1(x)
        x = self.batchnorm2(x, self.linear2(ccbn_condition))
        x = self.relu2(x)
        x = self.conv2(x)

        # upsampling the skip connections
        feat = self.upsample_BN(input_features)
        feat = self.conv_BN(feat)
        concate_feat = x + feat

        # residual block2
        res_connection = concate_feat
        x = self.batchnorm3(concate_feat, self.linear3(ccbn_condition))
        x = self.relu3(x)
        x = self.conv3(x)
        x = self.batchnorm4(x, self.linear4(ccbn_condition))
        x = self.relu4(x)
        x = self.conv4(x)

        output = x + res_connection

        return output



# -----------------------------------------------------------
#                   BUILDING GENERATOR
#------------------------------------------------------------

class Generator(nn.Module):
    """
        Generator:
            > contains 7-blocks
            > 3-7 blocks : upsample with the rate of (2,2,2,3,5)
            > 3,6,7 blocks need to reduce the channel size by a factor of 2
    """
    def __init__(self, in_channels=256, upsample_rates = [1,1,2,2,2,3,5], output_paddings = [0,0,1,1,1,2,4]):
        super().__init__()
        self.gen_blocks = nn.ModuleList([
            GBlock(input_features=in_channels, output_features=in_channels, upsample_rate=upsample_rates[0], output_padding = output_paddings[0]),
            GBlock(input_features=in_channels, output_features=in_channels, upsample_rate=upsample_rates[1], output_padding = output_paddings[1]),
            GBlock(input_features=in_channels, output_features=in_channels//2, upsample_rate=upsample_rates[2], output_padding = output_paddings[2]),
            GBlock(input_features=in_channels//2, output_features=in_channels//4, upsample_rate=upsample_rates[3], output_padding = output_paddings[3]),
            GBlock(input_features=in_channels//4, output_features=in_channels//8, upsample_rate=upsample_rates[4], output_padding = output_paddings[4]),
            GBlock(input_features=in_channels//8, output_features=in_channels//16, upsample_rate=upsample_rates[5], output_padding = output_paddings[5]),
            GBlock(input_features=in_channels//16, output_features=in_channels//16, upsample_rate=upsample_rates[6], output_padding = output_paddings[6])
        ])
        self.activation = nn.Tanh()
        

    def forward(self, inputs, noise_embeddings, speaker_embeddings):
        ccbn_condition = torch.concatenate([noise_embeddings, speaker_embeddings], dim=-1)
        x = inputs
        for bno, block in enumerate(self.gen_blocks):
            x = block(x, ccbn_condition)
        
        x = self.activation(x)
        return x
    
 



#-----------------------------------------------------------------------------
#                       SANITY-CHECKING FOR GENERATOR BLOCK
#-----------------------------------------------------------------------------
# if __name__ == "__main__":
    
#     # configurations
#     input_features = 256
#     out_seq = 6000
#     ccbn_condition_dims = 256
#     batch_size = 1
    
#     # creatiing a sample inputs
#     alined_features = torch.rand(size=(batch_size, out_seq, input_features))
#     alined_features = torch.permute(alined_features, dims=(0,2,1))
#     ccbn_condition = torch.rand(size=(batch_size, ccbn_condition_dims))

#     # checking the network
#     gblock = GBlock()
#     outputs = gblock(alined_features, ccbn_condition)

#     print("\n\n-----------------------------------------------------") 
#     print("|> shape of the input features : ", alined_features.shape)
#     print("|> shape of the condition input features : ", ccbn_condition.shape)
#     print("|> shape of the output features : ", outputs.shape)




#----------------------------------------------------------------------------------
#                      *** SANITY-CHECKING OF Entire GENERATOR ***
#----------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # configurations
    input_features = 256
    out_seq = 200
    noise_dims = 128
    speaker_dims = 128
    batch_size = 1

    # creating sample inputs
    alined_features = torch.rand(size=(batch_size, out_seq, input_features))
    alined_features = torch.permute(alined_features, dims=(0,2,1))
    noise_embs = torch.rand(size=(batch_size, noise_dims))
    speaker_embs = torch.rand(size=(batch_size, speaker_dims))

    # forwarding to the network
    generator = Generator()
    outputs = generator(alined_features, noise_embs, speaker_embs)

    # View of the inputs and outputs.
    print("|> Shape of the Input Features : ", alined_features.shape)
    print("|> Shape of the Output Features : ", outputs.shape)

