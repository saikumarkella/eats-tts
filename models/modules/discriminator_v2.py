'''
    Discriminator 
     Discriminates between the fake and real audio sample from the synthesizing and generation of the audio.
    
     There are 2 discrimator blocks
        > 1. Conditional Discriminator
        > 2. UnConditinal Discriminator
'''
# importing the modeuls
import torch
from torch  import nn
from pathlib import Path



class Conditional_Dblock(nn.Module):
    '''
        A Conditional Discriminator Block, There was the condition of th audio aligned representation
    '''
    def __init__(self):
        super(Conditional_Dblock, self).__init__()

        self.stack1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(),
            nn.ReLU(),
            nn.Conv1d()
        )

        self.residual_stack = nn.Sequential(
            nn.Conv1d(),
            nn.AdaptiveAvgPool1d()
        )
        self.bottle_neck = nn.Conv1d()

        self.stack2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d()
        )


    def forward(self, hidden_rep, conditional_input):
        x = hidden_rep
        x1 = self.stack1(x)
        x2 = self.bottle_neck(conditional_input)
        y1 = x1 + x2
        
        x3 = self.stack2(y1)
        x4 = self.residual_stack(x)
        y2 = x3 + x4
        return y2



class Dblock(nn.Module):
    '''
        In this Discrimnator Block  there is no condition on this Block
    '''
    def __init__(self):
        super(Dblock, self).__init__()

        self.stack1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(),
            nn.ReLU(),
            nn.Conv1d(),
            nn.ReLU(),
            nn.Conv1d()
        )
        self.residual_stack = nn.Sequential(
            nn.Conv1d(),
            nn.AdaptiveAvgPool1d()
        )

    def forward(self, hidden_rep):
        x1 = self.stack1(hidden_rep)
        x2 = self.residual_stack(hidden_rep)
        y = x1 + x2
        return y



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self):
        pass


