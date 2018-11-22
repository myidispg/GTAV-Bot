# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:42:06 2018

@author: myidi
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

from ConvNet import Net2, AlexNet

# Check if cuda is available
train_on_gpu = torch.cuda.is_available()

# Some parameters
batch_size = 32
n_epochs = 15
lr = 1e-3
input_width = 160
input_height = 120
model_name = "pygta5-car-{}-{}.pt".format(lr, 'convnet2', n_epochs)

if not train_on_gpu:
    print('CUDA is not available, training on CPU')
else:
    print('Training on GPU!!!')

model = AlexNet()    

# Check if a pretrained model exists
if os.path.exists(model_name):
    model.load_state_dict(torch.load(model_name))
    print('loaded a prexisting model')

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)


BASE_DATA_PATH = 'train_data/'
BASE_MODEL_PATH = 'models/'

# Load the training data
train_data = np.load(BASE_DATA_PATH + 'training_data-22-balanced.npy')
train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, 1, input_height, input_width) /255
Y = [i[1] for i in train]

#import cv2
#cv2.imshow('image', X[0])
#cv2.waitKey(0)

test_X = np.array([i[0] for i in test]).reshape(-1, 1, input_height, input_width) / 255
test_Y = [i[1] for i in test]

# This can convert a 4d numpy array to a pytorch dataloader. Useful for training
# https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader
import torch.utils.data as utils
tensor_X = torch.stack([torch.Tensor(i) for i in X])
tensor_y = torch.stack([torch.Tensor(i) for i in Y])

tensor_valid_X = torch.stack([torch.Tensor(i) for i in test_X])
tensor_valid_y = torch.stack([torch.Tensor(i) for i in test_Y])

dataset = utils.TensorDataset(tensor_X,tensor_y.long()) # create your dataset
dataloader = utils.DataLoader(dataset) # create your dataloader

valid_dataset = utils.TensorDataset(tensor_valid_X,tensor_valid_y.long()) # create your validation datset
valid_dataloader = utils.DataLoader(valid_dataset) # create your validation dataloader

del X, Y, train_data, train, test, tensor_X, tensor_y, dataset
del test_X, test_Y, tensor_valid_X, tensor_valid_y, valid_dataset

# Used while training.
valid_loss_min = np.Inf

#if train_on_gpu:
#    model.cuda()

for epoch in range(n_epochs):
    train_loss = 0
    valid_loss = 0
    model.train()
    for data, target in dataloader:
#        print(data.shape)
#        if train_on_gpu:
#            data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        label = torch.max(target, 1)[1]
#        print('expected output- {}    network output- {}'.format(target, output))
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        
        model.eval()
        
    for data, target in valid_dataloader:
#        if train_on_gpu:
#            data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, torch.max(target, 1)[1])
#        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
    
    # Calculate average loss
    train_loss /= len(dataloader.dataset)
    valid_loss /= len(valid_dataloader.dataset)
    
    # print statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, train_loss, valid_loss))
    
     # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), model_name)
        valid_loss_min = valid_loss


