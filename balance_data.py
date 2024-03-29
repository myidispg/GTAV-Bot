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

base_path = 'train_data/train_data.npy'

if os.path.isfile(base_path):
    print('File exists, loading training data!')
    train_data = np.load(base_path)
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
        choice = data[1]
        
        if choice == [1,0,0]:
            lefts.append([img,choice])
        elif choice == [0,1,0]:
            forwards.append([img,choice])
        elif choice == [0,0,1]:
            rights.append([img,choice])
        else:
            print('no matches')
        
#    shortest_lengths = min(len(lefts), len(forwards), len(rights)) * 3
    length = max(len(lefts), len(rights)) * 3
    
    # get the first indexes. Count depends on minimum of the turns.
    forwards = forwards[:length]
    lefts = lefts[:length]
    rights = rights[:length]
    
final_data = forwards + lefts + rights
shuffle(final_data)

np.save('train_data_self_balanced.npy', final_data)
    
    