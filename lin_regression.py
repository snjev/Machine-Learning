#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:18:48 2019

@author: sanjeevkumar
"""

import numpy as np
import matplotlib.pyplot as plt

dataset = np.loadtxt('boston-housing/data2.txt',delimiter = ',')
X = dataset[:,0:2]
Y = dataset[:,-1]

#normalizing DataSet

def normalizingDataset(trainData):
    mean = np.ones(trainData.shape[1])
    std = np.ones(trainData.shape[1])
    
    for i in range(trainData.shape[1]):
       mean[i] = np.mean(trainData.transpose()[i])
       std[i] = np.std(trainData.transpose()[i])
       for j in range(trainData.shape[0]):
           trainData[j][i] = (trainData[j][i] - mean[i]) / std[i]
           #print(trainData[j][i])
       return trainData
   
X = normalizingDataset(X)

def singlePerceptron(x,w):
    net_val = np.dot(x ,w)
    return net_val

def calGradientDescent(X, Y, weight):
    temp_weight = np.zeros(len(weight))
    for i in range(X.shape[0]):
        temp_weight += (singlePerceptron(X[i], weight) - Y[i]) * X[i]
    return temp_weight/X.shape[0]

def calLinRegression(X, Y, alpha,iter_):
    X = np.insert(X,0,1.,axis=1)
    weight = np.zeros(X.shape[1])
    for i in range(iter_):
        weight -= alpha * calGradientDescent(X, Y, weight)
    return weight

alpha = 0.1
iter_ = 700

w = calLinRegression(X, Y, alpha, iter_)
print(w)
xs = np.linspace(-2, 3, 100)

ys = np.zeros(len(xs))
for i in range(len(xs)):   
    ys[i] = w[1]*xs[i] + w[0]

plt.plot(X[:,0], Y,'o',xs,ys,'r*')
plt.show()