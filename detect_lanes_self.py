# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:33:58 2018

@author: myidi
"""

import cv2
import numpy as np

image = cv2.imread('lanes_image.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('image', image)
cv2.waitKey(0)
"""
Before we detect our edges, we need to make it clear what we are exactly looking for.
Lane lines are always yellow and white. Yellow can be tricky to isolate in RGB space.
So we convert to HSV space(Hue, Saturation, Value). Next, apply a digital mask to 
return the pixels we are interested in.
We can find a target range for yellow values by a Google search.
"""
# Convert to HSV
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Target range of yellow for lanes. Based on a google search
lower_yellow = np.array([20, 100, 100], dtype='uint8')
upper_yellow = np.array([30, 255, 255], dtype='uint8')

mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

cv2.imshow('image', mask_yellow)
cv2.waitKey(0)


