#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 07:11:13 2024

@author: Reese Barrett

Code for AMATH 582 Homework 2 assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler

# load in training data
train_filepath = '/Users/Reese/Documents/Research Projects/random/amath582/hw2data/train/'
test_filepath = '/Users/Reese/Documents/Research Projects/random/amath582/hw2data/test/'

#%% task 1: compile all the train samples into a matrix xtrain. Use PCA to
# investigate the dimensionality of xtrain and plot the first 5 PCA modes in
# xyz space.

files = ['jumping_2.npy', 'jumping_3.npy', 'jumping_4.npy', 'jumping_5.npy',
         'running_1.npy', 'running_2.npy', 'running_3.npy', 'running_4.npy',
         'running_5.npy', 'walking_1.npy', 'walking_2.npy', 'walking_3.npy',
         'walking_4.npy', 'walking_5.npy']

# initialize xtrain to store all training data
xtrain = np.zeros([11400,14])
i = 0

# step through all training data files and store in xtrain array
for file in files:
    traindata = np.load(train_filepath + file)
    xtrain[:,i] = traindata.flatten()
    i += 1

# plot data in xtrain: 2-D
fig = plt.figure(figsize=(7,5))
ax = fig.gca()
ax.plot(xtrain)
    
#xyz = np.reshape(traindata[:,:], (38,3,-1))

# scale dataset to have unit variance and zero mean
std_scaler = StandardScaler()
xtrain_scaled = std_scaler.fit_transform(xtrain)

# perform PCA, get first 5 PCA modes
pca = PCA(n_components=5)
principal_components = pca.fit_transform(xtrain_scaled)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
print(pca.components_)

# plot first 5 PCA modes in xyz space
for i in range(0,5):
    mode = np.reshape(principal_components[:,i], (38,3,-1))
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_title('PCA Mode ' + str(i+1))
    for j in range(0,mode.shape[2]):
        ax.scatter(mode[:,0,j],mode[:,1,j],mode[:,2,j]) # this is all time steps on one plot? is this correct?
        
# x, y, and z labels? set axis limits? colors?

#%% task 2: investigate how many PCA modes to keep in order to approximate
# Xtrain up to 70%, 80%, 90%, and 95%. Plot the cumulative energy to justify
# your results.

# initialize array to store cumulative energy at each PCA mode
cum_E = np.zeros((15,))

# calculate cumulative energy for each PCA mode
for i in range (0,15):
    pca = PCA(n_components=i)
    pca.fit(xtrain_scaled)
    cum_E[i] = np.sum(pca.explained_variance_ratio_)
    
# plot cumulative energy
fig = plt.figure(figsize=(7,5))
ax = fig.gca()
ax.plot(cum_E)
ax.set_ylabel('Cumulative Energy ($\\Sigma E_j$)')
ax.set_xlabel('Number of PCA Modes')

# answers to task 2 (from cum_E array)
# 70% and 80% - 1 PCA mode (explains 89% of variance)
# 90% - 2 PCA modes (explains 94% of variance)
# 95% - 3 PCA modes (explains 96% of variance)

#%% task 3: truncate the PCA modes to 2 and 3 modes and plot the projected
# xtrain in truncated PCA space as low dimensional 2D (PC1, PC2 coordinates)
# and 3D (PC2, PC2, PC3 coordinates) trajectories. Use colors for different
# movements and discuss the visualization and your findings.



























