# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:43:26 2018

@author: myidi
"""

import numpy as np
#from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from ConvNet import Net2, AlexNet
import os
import torch
from getkeys import key_check
from grabscreen import grab_screen

batch_size = 32
n_epochs = 80
lr = 1e-3
input_width = 160
input_height = 120
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

model = AlexNet()

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
#            screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            screen = grab_screen(region=(0, 40, 800, 640))
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))/255
#            moves = list(np.around(model([screen.reshape(1,160,120)])[0]))
#            if moves == [1,0,0]:
#                left()
#            elif moves == [0,1,0]:
#                straight()
#            elif moves == [0,0,1]:
#                right()
            screen = torch.Tensor([screen.reshape(1,160,120)])
            moves = model(screen)[0].detach().numpy()
            moves = np.argmax(moves)
            if moves == 0:
                left()
            elif moves == 1:
                straight()
            elif moves == 2:
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
                

main()
#import cv2
#screen1 = grab_screen(region=(0, 40, 800, 640))
#cv2.imshow('screen', screen)
# cv2.waitKey(0)

#screen = grab_screen(region=(0, 40, 800, 640))
#screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#screen = cv2.resize(screen, (160,120))/255
#screen = torch.Tensor([screen.reshape(1,160,120)])
#out = model(screen)[0].detach().numpy()
#out = np.argmax(out)
#out = list(np.around(model(screen)[0].detach().numpy()))
