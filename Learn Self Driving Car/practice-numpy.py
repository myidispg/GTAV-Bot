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

# Manipulating array shape
ravelled_array = x.ravel() # returns a view of the original array. 
# ^^Modifying this will modify the original array as well
flattened_array = x.flatten() # returns  a new copy.

y = np.arange(9)
y.shape=[3,3]
print(y)

print(y.transpose())

print(np.resize(y, (5,5)))

np.zeros((3,2), dtype=int)

print(np.eye(3)) # Identity matrix.

np.random.rand(4, 4)

# Matrix multiplication
mat_a = np.matrix([0,3,5,5,5,2]).reshape(2,3)
mat_b = np.matrix([3,4,3,-2, 4, -2]).reshape(3,2)
print(np.matmul(mat_a, mat_b))

# Stacking
x = np.arange(4).reshape(2,2)
print(x)
y = np.arange(4, 8).reshape(2,2)
print(y)
z = np.hstack((x,y)) # Stack arrays horizontally.
#Must have the same shape along all but the 2nd axis.
print(z)
q = np.vstack((x,y))
print(q)

print(np.concatenate((x,y), axis=1))