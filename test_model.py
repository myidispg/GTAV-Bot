# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:43:26 2018

@author: myidi
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from ConvNet import Net2, AlexNet2
import os
import torch
from getkeys import key_check

batch_size = 32
n_epochs = 80
lr = 1e-3
input_width = 80
input_height = 60
model_name = "pygta5-car-{}-{}.pt".format(lr, 'convnet2', n_epochs)

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)

def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)

model = Net2()

if os.path.exists(model_name):
    model.load_state_dict(torch.load(model_name))
    print('loaded a prexisting model')
else:
    print('No model found. Go train one!!!')
    
def main():
    last_time = time.time()
    
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    paused = False
    while(True):
        if not paused:
            # 800x600 windowed mode
            screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))
            moves = list(np.around(model.predict([screen.reshape(160,120,1)])[0]))
            if moves == [1,0,0]:
                left()
            elif moves == [0,1,0]:
                straight()
            elif moves == [0,0,1]:
                right()
   
        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)