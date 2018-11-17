# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 09:53:04 2018

@author: myidi
"""

import torch
import numpy as np

# Check if cuda is available
train_on_gpu = torch.cuda.is_available()

# Some parameters
batch_size = 32
n_epochs = 8
lr = 1e-3
input_width = 66
input_height = 200

if not train_on_gpu:
    print('CUDA is not available, training on CPU')
else:
    print('Training on GPU!!!')
    
import torch.nn as nn
import torch.nn.functional as F

# Input image is 66x200x1 or 66x200x3 channels. Will see later.
class Net(nn.Module):
    
    # This network is based on the Nvidia’s Convolutional Neural Network(CNN).
    # Image here -> https://cdn-images-1.medium.com/max/800/1*DvkLcBclo6D7q_vF94OEag.png
    def __init__(self):
        super(Net, self).__init__()
        # Sees an 66x200x1 image
        self.batch_norm = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 3, 5) # output 62x196x3
        self.conv2 = nn.Conv2d(3, 24, 5)  # output 58x192x24
        self.conv3 = nn.Conv2d(24, 36, 5) # output 54x188x36
        self.conv4 = nn.Conv2d(36, 48, 3) # output 52x186x48
        self.conv5 = nn.Conv2d(48, 64, 3) # output 50x184x64
        
        self.fc1 = nn.Linear(64*50*184, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 3)
        
    def forward(self, x):
        x = self.batch_norm(x)
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
    
model = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# Load the training data
train_data = np.load('train_data_balanced.npy')
train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, input_width, input_height, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, input_width, input_height,1)
test_y = [i[1] for i in test]

# Used while training.
valid_loss_min = np.Inf

for epoch in range(n_epochs):
    train_loss = 0
    valid_loss = 0
    
    model.train()
    
    
# Tjis can convert a 4d numpy array to a pytorch dataloader. Useful for training
import torch.utils.data as utils
my_x = np.array([[np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])], [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])]]) # a list of numpy arrays
my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)

tensor_x = torch.stack([torch.Tensor(i) for i in my_x])
tensor_y = torch.stack([torch.Tensor(i) for i in my_y])

my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader = utils.DataLoader(my_dataset) # create your dataloader

    