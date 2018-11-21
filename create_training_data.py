# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:21:43 2018

@author: myidi
"""

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os

'''
Here, we are one-hot encoding the inputs. That means that there will only be one 
keypress at a time. Combination key actions like speed and turn are disregarded for now.
'''

def keys_to_output(keys):
    """
    Convert keys to a ...multi-hot... array.
    [A,W,D] boolean values.
    """
    
    output = [0,0,0]
    if 'A' in keys:
        output[0] = 1
    elif 'W' in keys:
        output[1] = 1
    elif 'D' in keys:
        output[2] = 1
    return output

# Create a training database.

file_name = 'train_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []

# The main function to get screenshots.

def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
        
    while(True):
        # 800x600 windowed mode top left of screen
        screen = grab_screen(region=(0, 40, 800, 640))
        last_time = time.time()
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        # resize to feed into a CNN
        screen = cv2.resize(screen, (80, 60))
        # convert the input keys from user to a one-hot array.
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen, output])
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name,training_data)

main()
            

        
