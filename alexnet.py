# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 09:53:04 2018

@author: myidi
"""

import torch
import numpy as np

# Check if cuda is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available, training on CPU')
else:
    print('Training on GPU!!!')
    
import torch.nn as nn
import torch.nn.functional as F

# Input image is 80x60x1
class Net(nn.module):
    
    # This network is based on the Nvidia’s Convolutional Neural Network(CNN).
    # Image here -> https://cdn-images-1.medium.com/max/800/1*DvkLcBclo6D7q_vF94OEag.png
    def __init__(self):
        super(Net, self).__init__()
        # Sees an 80x60x1 image
        self.conv1 = nn.Conv2d(1, 96, 11, stride=4) # output 18x13x96 -> 8x6x96
        self.batch_norm1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, 5)  # output 4x2x256 -> 1x1x256
        self.batch_norm2 = nn.BatchNorm2d(2566)
        self.conv3 = nn.Conv2d(256, 384, 3)
        self.conv4 = nn.Conv2d(384, 384, 3)
        self.conv5 = nn.Conv2d(384, 256, 3)
        self.batch_norm3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size = 3, stride=2)
        
        self.fc1 = 
        
    
        
        
            