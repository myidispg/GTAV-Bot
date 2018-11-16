#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 17:59:48 2018

@author: myidispg
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time
import matplotlib.pyplot as plt
from directkeys import PressKey, ReleaseKey, W, A, S, D
import pyautogui


# Region of interest for finding lanes
def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_img(image):
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500],
                         ], np.int32)
    processed_img = roi(processed_img, [vertices])
    return processed_img

def main():
    
    # Script to simulate key input
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    
    last_time = time.time()
    while True:
        # Get the top left 800x640 screen section
        screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        new_screen = process_img(screen)
#        print('down')
#        PressKey(W)
#        time.sleep(3)
#        print('up')
#        PressKey(S)
        cv2.namedWindow("window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('window', 800,600)
        cv2.imshow('window', new_screen)
        if cv2.waitKey(delay=25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
main()
