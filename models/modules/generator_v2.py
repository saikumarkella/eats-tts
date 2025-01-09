"""
    The Generator Architecture version-2

    > Need to have a proper structure for the creation of generator 
"""
# modules
import torch.nn as nn
from utilsblocks import Conv1d, ConditionalBatchNorm, Upsampling
import torch
from pathlib import Path

# writing the summary in the tensorboard
from torch.utils.tensorboard import SummaryWriter

class GBlock(nn.Module):
    '''
        Discription : Generation will upsamples the audio aligned features to the wavform
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 upsample_factor):
        super(GBlock, self).__init__()

        # configurations
        self.in_channels = in_channels # 
        self.out_channels = out_channels # Maintain constant out-channels in entire network
        self.upsample_factor = upsample_factor # upsample factor of the network.

        # stack 1
        self.bn1 = ConditionalBatchNorm(num_features=in_channels)
        self.stack1 = nn.Sequential(
            nn.ReLU(),
            Upsampling(in_channel=self.in_channels, out_channel=self.in_channels, stride=self.upsample_factor),
            Conv1d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3)    
        )

        # stack 2
        self.bn2 = ConditionalBatchNorm(num_features=out_channels)
        self.stack2 = nn.Sequential(
            nn.ReLU(),
            Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, dialte_rate=2, kernel_size=3)
        )

        # residual stack
        self.residual_stack = nn.Sequential(
            Upsampling(in_channel=self.in_channels, out_channel=self.in_channels, stride=self.upsample_factor),
            Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        )

        # stack 3
        self.bn3 = ConditionalBatchNorm(num_features=self.out_channels)
        self.stack3 = nn.Sequential(
            nn.ReLU(),
            Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, dialte_rate=4, kernel_size=3)
        )

        # stack 4
        self.bn4 = ConditionalBatchNorm(num_features=self.out_channels)
        self.stack4 = nn.Sequential(
            nn.ReLU(),
            Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, dialte_rate=8, kernel_size=3)
        )

    def forward(self, inputs , ccbn_condition):
        
        # sub_block 1
        x1 = inputs
        x = self.bn1(inputs, ccbn_condition)
        x = self.stack1(x)
        x = self.bn2(x, ccbn_condition)
        x = self.stack2(x)

        # residual block
        res_out = self.residual_stack(x1)
        x = torch.concatenate([res_out, x])
        x2 = x
        
        # sub-block2 
        x = self.bn3(x, ccbn_condition)
        x = self.stack3(x)
        x = self.bn4(x, ccbn_condition)
        x = self.stack4(x)

        out = torch.concatenate([x2, x])
        return out



class Generator(nn.Module):
    '''
        A Generator which will upsample from audio aligned representation to the wavform. 
        It is a sequence of Generation Blocks

        Args:
            output_dims (list) :  A list of outdims of all the blocks
            upsample_factor (list) : A upsample factor of all layers

    '''
    def __init__(self, 
                number_layers = 7,
                output_dims = [256, 256, 128, 64, 32, 16, 8, 1],
                upsample_factors = [1,1,2,2,2,3,5]):
        super(Generator, self).__init__()
        
        # initializing all the layers
        self.number_layers = number_layers
        self.output_dims = output_dims
        self.upsample_factors = upsample_factors
        self.Gblocks = nn.ModuleList([GBlock(in_channels=self.output_dims[i], out_channels=self.output_dims[i+1], upsample_factor=self.upsample_factors[i]) for i in range(number_layers)])

    def forward(self, inputs, ccbn_condition):
        x = inputs
        for block in self.Gblocks:
            x = block(x, ccbn_condition)
        return x

        


# Sanity checking the Generator network for synthesizing audio or speech.
if __name__ == "__main__":
    # configurations
    batch = 1
    out_sequence = 400
    features = 256
    audio_aligned = torch.rand(size=(batch, features, out_sequence)) # input to generator
    noise_embeddings = torch.rand(size=(batch, features//2))
    speaker_emebddings = torch.rand(size=(batch, features//2))
    ccbn_condition = torch.concatenate([noise_embeddings, speaker_emebddings], dim=1)

    
    # initializing block
    in_channels = 256
    out_channels = 128
    upsample_fator = 2
    # gen_block = GBlock(in_channels=in_channels, out_channels=out_channels, upsample_factor=upsample_fator)
    gen_block = Generator()

    output = gen_block(audio_aligned, ccbn_condition)
    print(f"{output.shape = }")

    '''
        # Discriptions of models

        > Mode of the Model ( training or evaluation)
        > Number of layers
        > Number of learnable and non-learnable parameters
        > storing the state_dict.
        > loading the state_dict.

    '''
    print("|> Mode of the model : ", gen_block.training)
    print('|> Layers / modules in the Model : ')
    num_layers = 0
    for i in gen_block.children():
        print(i)
        num_layers+=1
    print('\n|> Number of layers in the modules :: ', num_layers)
    print("\n|> All the parameters in the Main Module :: ")
    total_trainable_parameters = 0
    total_non_trainable_parameters = 0
    for i in gen_block.parameters():
        if(i.requires_grad):
            total_trainable_parameters += i.numel()
        else:
            total_non_trainable_parameters += i.numel()

    print("|> Total number of trainable Parameters :: ", total_trainable_parameters)
    print("|> Total non-trainable parameters :: ", total_non_trainable_parameters)

    # tensorboard configurations
    logs_dir = Path(__file__).parent.parent/"logs"/"project1"


    '''
        TensorBoard Loggings:
        ---------------------

        1. Initializing the tensorboard summary writer
        2. Logging the model architecture.
        3. Logging the scalar or audios or images.
        4. Logging the pytorch profiler
    '''

    # step 1: initialize the tensorboard summary writer
    writer = SummaryWriter(log_dir=logs_dir)

    # step 2: Inspecting the network
    writer.add_graph(model=gen_block, input_to_model=(audio_aligned, ccbn_condition))

    # step 3: Closing the writer 
    writer.close()

