# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 09:53:04 2018

@author: myidi
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Input image is 80x60x1.
class Net(nn.Module):
    
    # This network is based on the Nvidia’s Convolutional Neural Network(CNN).
    # Image here -> https://cdn-images-1.medium.com/max/800/1*DvkLcBclo6D7q_vF94OEag.png
    def __init__(self):
        super(Net, self).__init__()
        # Sees an 60x80x1 image
        # self.batch_norm = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 3, 5) # output 56x76x3
        self.conv2 = nn.Conv2d(3, 24, 5)  # output 52x72x24
        self.conv3 = nn.Conv2d(24, 36, 5) # output 48x68x36
        self.conv4 = nn.Conv2d(36, 48, 3) # output 46x66x48
        self.conv5 = nn.Conv2d(48, 64, 3) # output 44x64x64
        
        self.fc1 = nn.Linear(64 * 44 * 64, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 3)
        
    def forward(self, x):
#        x = self.batch_norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        # Flatten the image
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        return x
