#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 17:59:48 2018

@author: myidispg
"""

import numpy as np
import cv2
import time
# from directkeys import PressKey, ReleaseKey, W, A, S, D
from directkeys import PressKey,ReleaseKey, W, A, S, D
from drawlanes import draw_lanes
from grabscreen import grab_screen


# Region of interest for finding lanes
def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked

def draw_lines(img, lines):
    for line in lines:
        cords = line[0]
        cv2.line(img, (cords[0], cords[1]), (cords[2], cords[3]), [255,255,255], 3)


    
def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500],
                         ], np.int32)
    # smoothen the already existing lines before adding new ones.
    processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)
    processed_img = roi(processed_img, [vertices])
    
    # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    #   edges       rho   theta   thresh         # min length, max gap:
    # Find straight lines
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 100, 5)
    m1 = 0
    m2 = 0
    try:
        l1, l2, m1, m2 = draw_lanes(original_image,lines)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
    except Exception as e:
        print(str(e))
        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
            except Exception as e:
                print(str(e))
    except Exception as e:
        pass

    return processed_img,original_image, m1, m2

# Some driving functions
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)

def right():
    PressKey(D) 
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

def hold_your_horses():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


def main():
    
    # Script to simulate key input
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    
    last_time = time.time()
    while True:
        # Get the top left 800x640 screen section
        screen =  grab_screen(region=(0,40,800,640))
        print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        new_screen,original_image, m1, m2 = process_img(screen)
#        print('down')
#        PressKey(W)
#        time.sleep(3)
#        print('up')
#        PressKey(S)
        # M1 and M2 are the slopes of the left and right lane. 
        # if the bike is towards either left or towards right a bit too much, the 
        # lanes are undetectable. So, the if block reorients the driver.
        if m1 < 0 and m2 < 0:
            right()
        elif m1 > 0  and m2 > 0:
            left()
        else:
            straight()
        cv2.namedWindow("window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('window', 800,600)
        cv2.imshow('window', new_screen)
        if cv2.waitKey(delay=25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
main()
