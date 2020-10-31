#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:08:39 2019

@author: sanjeevkumar
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxopt

def linear_kernal(x1, x2):
    return np.dot(x1.T, x2)

def polynomial_kernal(x1, x2, power=3):
    res = (1 + np.dot(x1.T, x2)) ** power
    return res

#def gaussian_kernal(x1, x2, sigma=0.5):
 #   return np.exp( -np.linalg(x1 - x2) ** 2 / 2 * (sigma ** 2))

class SVM:
    def __init__(self, kernal = linear_kernal, C = None):
        #c is none means it is a hard margin SVM
        self.kernal = kernal
        self.C = C
        if self.C is not None: self.C = float(self.C)
    def train(self, X, Y):
        n_sample, n_feature = X.shape
        #(xi, xj) is also called as gram matrix
        k = np.zeros((n_sample,n_sample))
        for i in range(n_sample):
            for j in range(n_sample):
                k[i,j] = self.kernal(X[i], X[j])
        print(k)
        p = cvxopt.matrix(np.outer(Y, Y) * k)
        q = cvxopt.matrix(np.ones(n_sample) * -1)
        A = cvxopt.matrix(Y, (1, n_sample))
        b = cvxopt.matrix(0.0)
        #print(p)
        
        if self.C is None:
            #if hard margin
            G = cvxopt.matrix(np.diag(np.ones(n_sample) * -1))
            h = cvxopt.matrix(np.zeros(n_sample))
        else:
            temp1 = np.diag(np.ones(n_sample) * -1)
            temp2 = np.identity(n_sample)
            G = cvxopt.matrix(np.vstack((temp1, temp2)))
            temp3 = np.zeros(n_sample)
            temp4 = np.ones(n_sample) * self.C
            h = cvxopt.matrix(np.hstack((temp3, temp4)))
        #print(G)
        #print(h)
        #Solve Quadratic Prob
        solution = cvxopt.solvers.qp(p, q, G, h, A, b)
        print(solution)
        #ALL lagrange Multiplier
        a = np.ravel(solution['x'])
        print(a)
        print('***********************')
        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = Y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_sample))
        #intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * k[ind[n],sv])
        self.b /= len(self.a)
        
        #weight
        if self.kernal == linear_kernal:
            self.w = np.zeros(n_feature)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None
    def predict_value(self, X):
        if self.w is not None:
            return (np.dot(X,self.w) + self.b)
        else:
            y_predict = np.zeros(len(X))
            for  i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b
    def test(self, X):
        return np.sign(self.predict_value(X))
    
    def f(self, x, w, b, c=0):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (-w[0] * x - b + c) / w[1]
        
    def plot_margin(self, X1_train, X2_train):
        

        plt.plot(X1_train[:,0], X1_train[:,1], "ro")
        plt.plot(X2_train[:,0], X2_train[:,1], "bo")
        plt.scatter(self.sv[:,0], self.sv[:,1], s=100, c="g")

        # w.x + b = 0
        a0 = -10; a1 = self.f(a0, self.w, self.b)
        b0 = 10; b1 = self.f(b0, self.w, self.b)
        plt.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1
        a0 = -10; a1 = self.f(a0, self.w, self.b, 1)
        b0 = 10; b1 = self.f(b0, self.w, self.b, 1)
        plt.plot([a0,b0], [a1,b1], "k--")

        # w.x + b = -1
        a0 = -10; a1 = self.f(a0, self.w, self.b, -1)
        b0 = 10; b1 = self.f(b0, self.w, self.b, -1)
        plt.plot([a0,b0], [a1,b1], "k--")

        plt.axis("tight")
        plt.show()

        

dataset = np.array([[5,1,1],[6,-1,1],[7,3,1],[1,7,-1],[2,8,-1],[3,8,-1]], dtype = 'double')
X = dataset[:,0:-1]
Y = dataset[:,-1]

svm = SVM()
svm.train(X, Y)

test_data = np.array([[3,4],[4,-4],[1,-2],[6,2],[0,6]], dtype = 'double')

y_predict = svm.test(test_data)
#correct = np.sum(y_predict == y_test)
#print("%d out of %d predictions correct" % (correct, len(y_predict)))
#print(y_predict)
svm.plot_margin(X[Y==1], X[Y==-1]) 