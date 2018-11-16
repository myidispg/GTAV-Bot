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

def process_img(image):
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    return processed_img

def main():
#    last_time = time.time()
    while True:
        # Get the top left 800x640 screen section
        screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
#        print('Frame took {} seconds'.format(time.time()-last_time))
#        last_time = time.time()
        new_screen = process_img(screen)
        cv2.imshow('window', new_screen)
        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(delay=25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
main()
