# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:57:47 2018

@author: myidi
"""

import numpy as np

def model(data, weights, biases):
    conv = tf.nn.conv2d(data, weights['w1'], strides=[1,1,1,1])
    hidden = tf.nn.relu(conv + biases['b1'])
    conv = tf.nn.conv2d(data, weights['w2'], strides=[1,1,1,1])
    hidden = tf.nn.relu(conv + biases['b2'])
    conv = tf.nn.conv2d(data, weights['w3'], strides=[1,1,1,1])
    hidden = tf.nn.relu(conv + biases['b3'])
    conv = tf.nn.conv2d(data, weights['w4'], strides=[1,1,1,1])
    hidden = tf.nn.relu(conv + biases['b4'])
    conv = tf.nn.conv2d(data, weights['w5'], strides=[1,1,1,1])
    hidden = tf.nn.relu(conv + biases['b5'])
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    
    fc = tf.nn.relu(tf.matmul(reshape, weights['w6']) + biases['b6'])
    fc = tf.nn.relu(tf.matmul(reshape, weights['w7']) + biases['b7'])
    fc = tf.nn.relu(tf.matmul(reshape, weights['w8']) + biases['b8'])
    fc = tf.matmul(fc, weights['w9'] + biases['b9'])
    
    return fc
    
