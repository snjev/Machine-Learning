#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:33:39 2019

@author: sanjeevkumar
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def get_mpgdataset():
    dataset = pd.read_csv('dataset_Facebook.csv',delimiter=';')
    del dataset['Page total likes']
    del dataset['Type']
    del dataset['Paid']
    del dataset['Total Interactions']
    del dataset['Category']
    dataset = dataset.dropna(how='any',axis=0)
    return dataset

X = get_mpgdataset()
pca = PCA(n_components = 4)
new_X = pca.fit_transform(X)  


kmeans = KMeans(n_clusters=4, random_state=0).fit(new_X)
centes = kmeans.cluster_centers_
plt.plot(new_X[:, 0], new_X[:, 1], 'bo')
plt.plot(centes[:, 0], centes[:, 1], 'rx')
