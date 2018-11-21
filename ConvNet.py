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
    
class Net2(nn.Module):
    
    # This network is based on the Nvidia’s Convolutional Neural Network(CNN).
    # Image here -> https://cdn-images-1.medium.com/max/800/1*DvkLcBclo6D7q_vF94OEag.png
    def __init__(self):
        super(Net2, self).__init__()
        # Sees an 60x80x1 image
        # self.batch_norm = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 3, 5) # 60x80x1 ->  56x76x3 -> 23x38x3
        self.conv2 = nn.Conv2d(3, 24, 5)  # 23x38x3 -> 19x34x24 -> 9x17x24
        self.conv3 = nn.Conv2d(24, 36, 5) # 9x17x24 -> 8x13x36
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(8*13*36, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 3)
        
    def forward(self, x):
#        x = self.batch_norm(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        
#        print(x.shape)
        
        # Flatten the image
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        return x
    
    
class AlexNet(nn.Module):
    
    # Based on AlexNet architecture
    def __init__(self):
        super(AlexNet, self).__init__()
        # input 1x60x80 (Channels x height x width)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2), # 1x60x80 -> 19x46x96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # output 9x22x96
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNet2(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
