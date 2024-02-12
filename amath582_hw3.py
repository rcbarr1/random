#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:54:40 2024

@author: Reese Barrett

Code for AMATH 582 Homework 3 assignment.

TO-DO:
    - what does "plot the first 16 PC modes as 28 x 28 images mean? like one digit in all 16 modes?"
"""

import numpy as np
import struct
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifierCV

#%% set up filepath
filepath = '/Users/Reese/Documents/Research Projects/random/amath582/hw3data/'

#%% task 1: perform PCA analysis of the digit images
# - reshape each image and stack into Xtrain and Xtext matricies
# - plot the first 16 PC modes as 28 x 28 images

# load in test and training data + associated labels, store in vectors
# training data
with open(filepath + 'train-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    xtrain = data.reshape((size, nrows*ncols))

# training labels
with open(filepath + 'train-labels-idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    ytrain_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))

# test data
with open(filepath + 't10k-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    xtest = data.reshape((size, nrows*ncols))
    
# test labels
with open(filepath + 't10k-labels-idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    ytest_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    
#traindata_imgs =  xtrain.reshape((60000,28,28))    

# perform PCA on training data (keep 16 PCA modes)
pca = PCA(n_components=16)
xtrain_pca = pca.fit_transform(xtrain)
xtrain_xyz = pca.inverse_transform((xtrain_pca))

# plot first 64 training digits in regular space and a
def plot_digits(XX, N, title):
    fig, ax = plt.subplots(N, N, figsize=(8, 8))
    XX = XX.transpose()
    for i in range(N):
      for j in range(N):
        ax[i,j].imshow(XX[:,(N)*i+j].reshape((28, 28)), cmap="Greys")
        ax[i,j].axis("off")
    fig.suptitle(title, fontsize=24)

plot_digits(xtrain, 8, "First 64 Training Images (Unmodified)" )
plot_digits(xtrain_xyz, 8, "First 64 Training Images (16 PC Modes)" )

# calculate first 16 PCA modes of each digit
xtrain_xyz_all16modes = np.zeros((xtrain.shape[0], xtrain.shape[1], 16)) # preallocate array to hold results of all 16 PCA modes
for i in range(0, 16):
    pca = PCA(n_components=i+1) # create PCA basis
    xtrain_pca = pca.fit_transform(xtrain) # transform to PCA space
    xtrain_xyz = pca.inverse_transform((xtrain_pca)) # invert back to xyz space
    xtrain_xyz_all16modes[:, :, i] = xtrain_xyz # store in large array

# plot all 16 PCA modes of one digit
first_digit_all16modes = np.transpose(xtrain_xyz_all16modes[0, :, :])
plot_digits(first_digit_all16modes, 4, "First Digit (1 to 16 PC Modes)")

#%% task 2: inspeact the cumulative energy to determine k (the number of PC
# modes needed to approximate 85% of the energy)

# initialize array to store cumulative energy at each PCA mode
cum_E = np.zeros((xtrain.shape[1],))

# calculate cumulative energy for each PCA mode3
for i in range(0,100):
    pca = PCA(n_components=i)
    pca.fit(xtrain)
    cum_E[i] = np.sum(pca.explained_variance_ratio_)

# plot cumulative energy
fig = plt.figure(figsize=(7,5),dpi=200)
ax = fig.gca()
ax.plot(cum_E,linewidth=2)

ax.set_ylabel('Cumulative Energy ($\\Sigma E_j$)')
ax.set_xlabel('Number of PCA Modes')
ax.set_xlim([0,99])
ax.legend(loc='lower right')

# need 59 PCA modes to get to 85% cumulative energy
k = 59

#%% task 3: write a function that selects a subset of particular digits (all
# samples of them) and returns the subset as new matricies

def digit_subset(digit, xtrain, ytrain_labels, xtest, ytest_labels):
    # digit = the digit you are looking for (i.e. 2)
    # xtrain = dataset of all training data
    # ytrain_labels = labels for all training data
    # xtest = dataset of all test data
    # ytest_labels = labels for all test data
    # xsubtrain = training data associated with the provided digit
    # ysubtrain = labels matching xsubtrain
    # xsubtest = test data associated with the provided digit
    # ysubtest = labels matching xsubtest
    
    digit_idx = np.where(ytrain_labels == digit)
    digit_idx = digit_idx[0]
    xsubtrain = xtrain[digit_idx,:]
    ysubtrain = ytrain_labels[digit_idx]
    
    digit_idx = np.where(ytest_labels == digit)
    digit_idx = digit_idx[0]
    xsubtest = xtest[digit_idx,:]
    ysubtest = ytest_labels[digit_idx]
    return xsubtrain, ysubtrain, xsubtest, ysubtest

#%% task 4: select the digits 1 and 8, project the data onto k-PC modes
# computed in steps 1-2, and apply the Ridge classifier (linear) to distinguish
# between these two digits. Perform cross-validation and testing.

# subset for 1 and 8
xsubtrain1, ysubtrain1, xsubtest1, ysubtest1 = digit_subset(1, xtrain, ytrain_labels, xtest, ytest_labels)
xsubtrain8, ysubtrain8, xsubtest8, ysubtest8 = digit_subset(8, xtrain, ytrain_labels, xtest, ytest_labels)

# stack the 1 and 8 data
xsubtrain = np.vstack((xsubtrain1, xsubtrain8))
ysubtrain = np.hstack((ysubtrain1, ysubtrain8))
xsubtest = np.vstack((xsubtest1, xsubtest8))
ysubtest = np.hstack((ysubtest1, ysubtest8))

# perform PCA with k modes (59)
pca = PCA(n_components=k) # create PCA basis with k modes
xsubtrain_pca = pca.fit_transform(xsubtrain) # fit and transform training data to PCA space
xsubtest_pca = pca.transform(xsubtest) # transform test data to PCA space











