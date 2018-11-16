# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:40:41 2018

@author: myidi
"""

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import os

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading training data!')
    train_data = np.load('training_data.npy')
else:
    print('File does not exist, go and train some data before bothering me!')

if 'train_data' in locals():
    df = pd.DataFrame(train_data)
    print(df.head())
    print(Counter(df[1].apply(str)))
    
    lefts = []
    rights = []
    forwards = []
    
    shuffle(train_data)
    
    for data in train_data:
        img = data[0]
        action = data[1]
        
        if choice == [1,0,0]:
        lefts.append([img,choice])
    elif choice == [0,1,0]:
        forwards.append([img,choice])
    elif choice == [0,0,1]:
        rights.append([img,choice])
    else:
        print('no matches')
        
    shortest_lengths = min(len(lefts), len(forwards), len(rights))
    
    # get the first indexes. Count depends on minimum of the turns.
    forwards = forwards[:shortest_lengths]
    lefts = lefts[:shortest_lengths]
    rights = rights[:shortest_lengths]
    
final_data = forwards + lefts + rights
shuffle(final_data)

np.save('training_data_balanced.npy', final_data)
    
    