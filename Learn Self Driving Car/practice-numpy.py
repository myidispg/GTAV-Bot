# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 20:04:41 2018

@author: myidi
"""

import numpy as np


#-----Reshape
x = np.arange(18).reshape(3,2,3)
print(x)

# Slicing multi dimensional arrays
x[1, 1, 1]
x[1, 0:2, 0:3]
x[1]

comparison_operation = x > 5
print(comparison_operation)
x[comparison_operation]
x[x>5]
