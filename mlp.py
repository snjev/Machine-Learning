#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:11:17 2019

@author: sanjeevkumar
"""

import numpy as np
import matplotlib.pyplot as plt

class MLP:
    #initialized all variables
    def __init__(self):
        self.Wji = None
        self.Wkj = None
        self.H = None
        self.n_hidden = None
        self.n_feature = None
        self.n_output = None
        self.O = None
    #calculate activation function
    def sigmoid(self, z):
        return np.reciprocal(1+np.exp(-z))
    #calculate teh feed forward i.e. weight and output of all nodes 
    def fit(self,X, Y, n_hidden = 4, alpha = 1.0, max_iter = 1000, e=0.0001):
        X = np.insert(X, 0, 1, axis=1)
        n_sample, self.n_feature = X.shape
        self.n_output = 2
        self.O = np.zeros((self.n_output,), dtype=float)
        n_hidden+=1
        self.H = np.zeros(shape=(n_hidden,), dtype=float)
        self.H[0] = 1.
        self.Wji = np.random.normal(loc=0.0, scale=1.0, size=(n_hidden*self.n_feature)).reshape((n_hidden, self.n_feature))
        self.Wkj = np.random.normal(loc=0.0, scale=1.0, size=(n_hidden*self.n_output)).reshape((self.n_output, n_hidden))
        iter = 0
        while iter < max_iter:
            for i, xi in enumerate(X):
                # calculate hidden unit
                #print(xi)
                for j, hj in enumerate(self.H):
                    #skip the bias term
                    if j==0: continue
                    #for all other calculate weight and activation function
                    net_hj = np.dot(self.Wji[j], xi)
                    out_hj = self.sigmoid(net_hj)
                    #print(net_hj)
                    self.H[j] = out_hj 
                #calculate output
                for k, yk in enumerate(self.O):
                    net_yk = np.dot(self.Wkj[k], self.H)
                    out_yk = self.sigmoid(net_yk)
                    self.O[k] = out_yk
                
                dWji = np.zeros(self.Wji.shape)
                dWkj = np.zeros(self.Wkj.shape)
                
                propg_term = 0
                
                for k, Ok in enumerate(self.O):
                    dWkj[k] = Ok * (1 - Ok) * (Y[i][k] - Ok)
                    
                    propg_term+=np.dot(dWkj[k],self.Wkj[k])
                #for hidden layer
                for h, Oh in enumerate(self.H):
                    dWji[h] = Oh * (1 - Oh) * propg_term
                    
                self.Wkj = self.Wkj + alpha * dWkj*self.H
                self.Wji = self.Wji + alpha * dWji * xi
            
            iter += 1
    def predict(self, t):
        X = np.insert(t, 0, 1., axis=1)
        #print('Hidden weights', self.Wji)
        #print('Output weights', self.Wkj)
        output = list()
        for i, xi in enumerate(X):
            # calculating hidden units
            for j, hj in enumerate(self.H):
                # skip for bias term
                if j == 0: continue
                # for all other hidden units, calculate net and activation
                net_hj = np.dot(self.Wji[j], xi)
                out_hj = self.sigmoid(net_hj)
                self.H[j] = out_hj
            # calculating output units
            for k, yk in enumerate(self.O):
                net_yk = np.dot(self.Wkj[k], self.H)
                out_yk = self.sigmoid(net_yk)
                self.O[k] = out_yk
            #print(self.O)                    
            output.append(np.array(self.O))
        return output
if __name__ == '__main__':
    X = np.array([[0,1],[1,.8],[1,1.2],[2,3.7],[2,6.0],[3,3],[.5,3]])
    Y = np.array([[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]])
   
        
    mlp = MLP()
    #type(mlp.sigmoid(5))
    
    
    mlp.fit(X,Y)
    mlp.predict(X)
    
    t1 = np.linspace(0,3,100)
    t2 = np.linspace(0,7,100)
    t = np.array([(x,y) for x in t1 for y in t2])
    #print(t)
    
    c = mlp.predict(t)
    c = np.array(c)
    i = c[:,0]>0.5
    g1 = list()
    g0 = list()
    for k in range(len(i)):
        if i[k]:
            g1.append(t[k])
        else:
            g0.append(t[k])
    
    g1 = np.array(g1)
    g0 = np.array(g0)
    
   
    plt.plot(g1[:,0], g1[:,1],'r.',g0[:,0], g0[:,1],'b.',X[:,0],X[:,1],'k*')
    plt.show()