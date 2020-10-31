#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 15:38:55 2019

@author: sanjeevkumar
"""

import numpy as np
import matplotlib.pyplot as plt

W = np.nan

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

X = np.array([[1,1,1], [2.5, 1, 1], [1, 4, 1], [-2, 1, 0], [-1, -1, 0], [.5, -1, 0]], dtype =float)

def train(X,alpha=0.001, e=0.0001, max_iter=200):
    X = np.insert(X, 0, 1, axis=1)
    Y = X[:,-1]
    X = X[:,:-1]
    N = X.shape[0]
    global W
    W = np.zeros(X.shape[1])
    E = np.inf
    count = 0
    while E > e and count < max_iter:
        E = 0
        dw = np.zeros(W.shape)
        for j,xj in enumerate(X):
            net = np.dot(W,xj)
            hz = sigmoid(net)
            dw += (hz - Y[j])*xj
            E += - Y[j]* np.log(hz) -(1- Y[j])* np.log(1 - hz)
        dw = dw / N
        E = E / N
        W = W - alpha * dw
        count += 1
        #print('trained {} in {} iterations with Error {}'.format(W, count, E))

def test(sample):
    t = np.array([1, sample[0],sample[1]])
    global W
    net = np.dot(W, t)
    hz = sigmoid(net)
    op = 0 if hz < 0.5 else 1
    return op

train(X)
xs = np.linspace(-5, 5, 100)
ys = np.zeros(len(xs))
for i in range(len(xs)):   
    ys[i] = -W[1]*xs[i]/W[2] + W[0]/W[2]

plt.plot(X[:,0], X[:, 1], 'bo', xs, ys, 'rx')
plt.show()