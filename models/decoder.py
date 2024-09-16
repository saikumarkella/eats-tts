'''
    Decoder ( Generator and Discriminator )

    Which will accepts audio aligned represenation and the noise emebddings and speaker emebddinngs.
    These Decode things will inspired from GANTTS
'''
# Here both generator and discriminator modules were presents
import torch
import numpy as np
from modules import generator, discriminator
import sys
import os
import warnings

warnings.filters('ignore')
