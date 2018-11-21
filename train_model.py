# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:42:06 2018

@author: myidi
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ConvNet import Net

# Check if cuda is available
train_on_gpu = torch.cuda.is_available()

# Some parameters
batch_size = 32
n_epochs = 8
lr = 1e-3
input_width = 80
input_height = 60
model_name = "pygta5-car-{}-{}-{}epochs.model".format(lr, 'convnet', n_epochs)

if not train_on_gpu:
    print('CUDA is not available, training on CPU')
else:
    print('Training on GPU!!!')
    
model = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# Load the training data
train_data = np.load('train_data_balanced.npy')
train = train_data[:-9]
test = train_data[-9:]

X = np.array([i[0] for i in train]).reshape(-1, input_height, input_width, 1)
Y = [i[1] for i in train]

# This can convert a 4d numpy array to a pytorch dataloader. Useful for training
# https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader
import torch.utils.data as utils
tensor_X = torch.stack([torch.Tensor(i) for i in X])
tensor_y = torch.stack([torch.Tensor(i) for i in Y])

dataset = utils.TensorDataset(tensor_X,tensor_y) # create your datset
dataloader = utils.DataLoader(dataset) # create your dataloader
# Used while training.
valid_loss_min = np.Inf





