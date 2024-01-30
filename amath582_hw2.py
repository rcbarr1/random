#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 07:11:13 2024

@author: Reese Barrett

Code for AMATH 582 Homework 2 assignment.

QUESTIONS FOR OA
- what does visualizing in xyz space mean? does this look okay? just one plot
  for unmodified, one for 5 modes?
- if I'm using all 14 PCA modes, shouldn't my classifier work 100% correctly?
  It's not, and I'm confused
      - am I computing centroids incorrectly?
      - or distance? those are the two things that could be wrong I guess?
      - or is it because I'm computing the centroid and everything is okay
- do I need to scale the data? It works better if it doesn't but seems like the
  documentation online says I do need to
      - currently using sklearn.preprocessing.StandardScaler to do so
- my accuracy is really bad, is that expected with this method?
      
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

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
xtrain = np.zeros([14, 11400])
i = 0

# step through all training data files and store in xtrain array
for file in files:
    traindata = np.load(train_filepath + file)
    xtrain[i,:] = traindata.flatten()
    i += 1

# scale dataset to have unit variance and zero mean
#std_scaler = StandardScaler()
#xtrain_scaled = std_scaler.fit_transform(xtrain)
xtrain_scaled = xtrain

# perform PCA, get first 5 PCA modes
pca = PCA(n_components=5)
principal_components = pca.fit_transform(xtrain_scaled)

# transform back to xyz space
xtrain_5modes = pca.inverse_transform(principal_components)

# plot walking in xyz space (scatter)

# define the root joint
r = 1000

# define the connections between the joints 
I = np.array([1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14, 15, 16, 17, 18, 19,
              16, 21, 22, 23, 25, 26, 24, 28, 16, 30, 31, 32, 33, 34, 35, 33,
              37]) - 1
J = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
              20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
              37, 38]) - 1

# make plot of jumping unmodified
fig = plt.figure(figsize=(7,5))
ax = plt.axes(projection = '3d')

for train_num in range(0,4):
    xyz = np.reshape(xtrain[train_num,:], (38,3,-1))
    xroot, yroot, zroot = xyz[0,0,0], xyz[0,0,1], xyz[0,0,2] # define scaling of the values
    for t in range(1,xyz.shape[2]):  
        for ij in range(0,I.shape[0]):
            xline = np.array([xyz[I[ij],0,t], xyz[J[ij],0,t]])
            yline = np.array([xyz[I[ij],1,t], xyz[J[ij],1,t]])
            zline = np.array([xyz[I[ij],2,t], xyz[J[ij],2,t]])
            # use plot if you'd like to plot skeleton with lines
            ax.plot(xline,yline,zline,c='steelblue',alpha = 0.05)
        
ax.set_xlim([-r+xroot, r+xroot])
ax.set_zlim([-r+zroot, r+zroot])
ax.set_ylim([-r+yroot, r+yroot])
ax.set_title('Jumping (Unmodified)')
        
# plot of jumping with 5 PCA modes
fig = plt.figure(figsize=(7,5))
ax = plt.axes(projection = '3d')

for train_num in range(0,4):
    xyz_5modes = np.reshape(xtrain_5modes[train_num,:], (38,3,-1))
    xroot, yroot, zroot = xyz_5modes[0,0,0], xyz_5modes[0,0,1], xyz_5modes[0,0,2] # define scaling of the values
    for t in range(1,xyz.shape[2]):  
        for ij in range(0,I.shape[0]):
            xline = np.array([xyz_5modes[I[ij],0,t], xyz_5modes[J[ij],0,t]])
            yline = np.array([xyz_5modes[I[ij],1,t], xyz_5modes[J[ij],1,t]])
            zline = np.array([xyz_5modes[I[ij],2,t], xyz_5modes[J[ij],2,t]])
            # use plot if you'd like to plot skeleton with lines
            ax.plot(xline,yline,zline,c='steelblue',alpha = 0.05)
        
ax.set_xlim([-r+xroot, r+xroot])
ax.set_zlim([-r+zroot, r+zroot])
ax.set_ylim([-r+yroot, r+yroot])
ax.set_title('Jumping (5 PCA Modes)')

# make plot of running unmodified
fig = plt.figure(figsize=(7,5))
ax = plt.axes(projection = '3d')

for train_num in range(4,9):
    xyz = np.reshape(xtrain[train_num,:], (38,3,-1))
    xroot, yroot, zroot = xyz[0,0,0], xyz[0,0,1], xyz[0,0,2] # define scaling of the values
    for t in range(1,xyz.shape[2]):  
        for ij in range(0,I.shape[0]):
            xline = np.array([xyz[I[ij],0,t], xyz[J[ij],0,t]])
            yline = np.array([xyz[I[ij],1,t], xyz[J[ij],1,t]])
            zline = np.array([xyz[I[ij],2,t], xyz[J[ij],2,t]])
            # use plot if you'd like to plot skeleton with lines
            ax.plot(xline,yline,zline,c='steelblue',alpha = 0.05)
        
ax.set_xlim([-r+xroot, r+xroot])
ax.set_zlim([-r+zroot, r+zroot])
ax.set_ylim([-r+yroot, r+yroot])
ax.set_title('Running (Unmodified)')
        
      
# plot of running with 5 PCA modes
fig = plt.figure(figsize=(7,5))
ax = plt.axes(projection = '3d')

for train_num in range(4,9):
    xyz_5modes = np.reshape(xtrain_5modes[train_num,:], (38,3,-1))
    xroot, yroot, zroot = xyz_5modes[0,0,0], xyz_5modes[0,0,1], xyz_5modes[0,0,2] # define scaling of the values
    for t in range(1,xyz.shape[2]):  
        for ij in range(0,I.shape[0]):
            xline = np.array([xyz_5modes[I[ij],0,t], xyz_5modes[J[ij],0,t]])
            yline = np.array([xyz_5modes[I[ij],1,t], xyz_5modes[J[ij],1,t]])
            zline = np.array([xyz_5modes[I[ij],2,t], xyz_5modes[J[ij],2,t]])
            # use plot if you'd like to plot skeleton with lines
            ax.plot(xline,yline,zline,c='steelblue',alpha = 0.05)
        
ax.set_xlim([-r+xroot, r+xroot])
ax.set_zlim([-r+zroot, r+zroot])
ax.set_ylim([-r+yroot, r+yroot])
ax.set_title('Running (5 PCA Modes)')

# make plot of walking unmodified
fig = plt.figure(figsize=(7,5))
ax = plt.axes(projection = '3d')

for train_num in range(9,14):
    xyz = np.reshape(xtrain[train_num,:], (38,3,-1))
    xroot, yroot, zroot = xyz[0,0,0], xyz[0,0,1], xyz[0,0,2] # define scaling of the values
    for t in range(1,xyz.shape[2]):  
        for ij in range(0,I.shape[0]):
            xline = np.array([xyz[I[ij],0,t], xyz[J[ij],0,t]])
            yline = np.array([xyz[I[ij],1,t], xyz[J[ij],1,t]])
            zline = np.array([xyz[I[ij],2,t], xyz[J[ij],2,t]])
            # use plot if you'd like to plot skeleton with lines
            ax.plot(xline,yline,zline,c='steelblue',alpha = 0.05)
        
ax.set_xlim([-r+xroot, r+xroot])
ax.set_zlim([-r+zroot, r+zroot])
ax.set_ylim([-r+yroot, r+yroot])
ax.set_title('Walking (Unmodified)')
        
      
# plot of running with 5 PCA modes
fig = plt.figure(figsize=(7,5))
ax = plt.axes(projection = '3d')

for train_num in range(9,14):
    xyz_5modes = np.reshape(xtrain_5modes[train_num,:], (38,3,-1))
    xroot, yroot, zroot = xyz_5modes[0,0,0], xyz_5modes[0,0,1], xyz_5modes[0,0,2] # define scaling of the values
    for t in range(1,xyz.shape[2]):  
        for ij in range(0,I.shape[0]):
            xline = np.array([xyz_5modes[I[ij],0,t], xyz_5modes[J[ij],0,t]])
            yline = np.array([xyz_5modes[I[ij],1,t], xyz_5modes[J[ij],1,t]])
            zline = np.array([xyz_5modes[I[ij],2,t], xyz_5modes[J[ij],2,t]])
            # use plot if you'd like to plot skeleton with lines
            ax.plot(xline,yline,zline,c='steelblue',alpha = 0.05)
        
ax.set_xlim([-r+xroot, r+xroot])
ax.set_zlim([-r+zroot, r+zroot])
ax.set_ylim([-r+yroot, r+yroot])
ax.set_title('Walking (5 PCA Modes)')

#%% task 2: investigate how many PCA modes to keep in order to approximate
# Xtrain up to 70%, 80%, 90%, and 95%. Plot the cumulative energy to justify
# your results.

# initialize array to store cumulative energy at each PCA mode
cum_E = np.zeros((xtrain.shape[0],))

# calculate cumulative energy for each PCA mode
for i in range (0,xtrain.shape[0]):
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
# 70% - 4 PCA modes (explains 73.5% of variance)
# 80% - 5 PCA modes (explains 81.2% of variance)
# 90% - 8 PCA modes (explains 92.5% of variance)
# 95% - 10 PCA modes (explains 97.1% of variance)

#%% task 3: truncate the PCA modes to 2 and 3 modes and plot the projected
# xtrain in truncated PCA space as low dimensional 2D (PC1, PC2 coordinates)
# and 3D (PC2, PC2, PC3 coordinates) trajectories. Use colors for different
# movements and discuss the visualization and your findings.

# 2D PCA space
pca = PCA(n_components=2)
principal_components = pca.fit_transform(xtrain_scaled)

# plot 2D PCA
fig = plt.figure(figsize=(7,5))
ax = fig.gca()
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.scatter(principal_components[0:4, 0], principal_components[0:4, 1], label='Jumping')
ax.scatter(principal_components[4:9, 0], principal_components[4:9, 1], label='Running')
ax.scatter(principal_components[9:14, 0], principal_components[9:14, 1], label='Walking')
ax.legend()

# 3D PCA space
pca = PCA(n_components=3)
principal_components = pca.fit_transform(xtrain_scaled)

# plot 3D PCA
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection = '3d')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.scatter3D(principal_components[0:4, 0], principal_components[0:4, 1], principal_components[0:4, 2], label='Jumping')
ax.scatter3D(principal_components[4:9, 0], principal_components[4:9, 1], principal_components[4:9, 2], label='Running')
ax.scatter3D(principal_components[9:14, 0], principal_components[9:14, 1], principal_components[9:14, 2], label='Walking')
ax.legend()

#%% task 4: create a ground truth table with an integer per class and assign an
# appropriate label to each sample in xtrain. Then, for each movement compute
# its centroid in k-modes PCA space.

# 0 = jumping
# 1 = running
# 2 = walking
train_truth = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

# calculate centroid in k-modes PCA space
# The three rows in this array represent the three types of movements (see key
# above). Each column represents the centroid of that mode PCA space. For
# example, the first three columns would be considered the "centroid" when
# looking at 3-modes PCA space

# initialize array to hold all centroids
# array is three rows (0 = jumping, 1 = running, 2 = walking)
centroids = np.zeros((3,xtrain.shape[0]))

# do PCA
pca = PCA(n_components=xtrain.shape[0])
principal_components = pca.fit_transform(xtrain_scaled)

for i in range(0, principal_components.shape[1]):
    centroids[0,i] = np.mean(principal_components[0:4, i])
    centroids[1,i] = np.mean(principal_components[4:9, i])
    centroids[2,i] = np.mean(principal_components[9:14, i])

#%% task 5: create another vector of trained labels. To assign these labels,
# for each sample in xtrain compute the distance between the projected point in
# k-modes PCA space and each of the centroids. The minimal distance will
# determine to which class the sample belongs. Assign the label of the class of
# the centroid with minimal distance in the trained labels vector (5.1).
# Compute the trained labels for various k values of k-PCA truncation and
# report the accuracy of the trained classifier (the percentage of samples for
# which the ground truth and the trained labels match) You can use the
# accuracy_score function in sklearn for this purpose (5.2). Discuss your
# results in terms of optimal k for accuracy.

# 5.1: For each training dataset, compute the distance between the projected
# point in k-modes PCA space and each of the centroids. Choose the minimum
# distance and assign the corresponding label.

# do PCA, principal_components contains k-modes information for each training dataset point
pca = PCA(n_components=xtrain.shape[0])
principal_components = pca.fit_transform(xtrain_scaled)

# for each mode, compute distance between the point and the centroid
# initialize array to store distance between point and centroid
classified = np.zeros((xtrain.shape[0],xtrain.shape[0]))

for j in range(0, xtrain.shape[0]):
    for i in range(0, xtrain.shape[0]):
        dist0 = distance.euclidean(principal_components[i,0:j+1], centroids[0,0:j+1])
        dist1 = distance.euclidean(principal_components[i,0:j+1], centroids[1,0:j+1]) 
        dist2 = distance.euclidean(principal_components[i,0:j+1], centroids[2,0:j+1])
        
        # choose minimum distance for each row, assign label
        dists = [dist0, dist1, dist2]
        classified[i,j] = dists.index(min(dists))
        
# 5.2: report accuracy of classifier
scores = np.zeros(xtrain.shape[0])
for i in range(0,xtrain.shape[0]):
    scores[i] = accuracy_score(train_truth, classified[:,i])

#%% task 6: load the given test samples and for each sample assign the ground
# truth label. By projecting into k-PCA space and computing the centroids,
# predict the test labels. Report the accuracy.

files = ['jumping_1t.npy', 'running_1t.npy', 'walking_1t.npy']
test_truth = [0, 1, 2]

# initialize xtest to store all training data
xtest = np.zeros([3, 11400])
i = 0

# step through all test data files and store in xtest array
for file in files:
    testdata = np.load(test_filepath + file)
    xtest[i,:] = testdata.flatten()
    i += 1

# scale dataset to have unit variance and zero mean
#xtest_scaled = std_scaler.fit_transform(xtest)
xtest_scaled = xtest

# project test data into PCA space
test_principal_components = pca.transform(xtest_scaled)

# for each mode, compute distance between the point and the centroid
# initialize array to store distance between point and centroid
test_classified = np.zeros((xtest.shape[0],xtrain.shape[0]))

for j in range(0, xtrain.shape[0]):
    for i in range(0, xtest.shape[0]):
        dist0 = distance.euclidean(test_principal_components[i,0:j+1], centroids[0,0:j+1])
        dist1 = distance.euclidean(test_principal_components[i,0:j+1], centroids[1,0:j+1]) 
        dist2 = distance.euclidean(test_principal_components[i,0:j+1], centroids[2,0:j+1])
        
        # choose minimum distance for each row, assign label
        dists = [dist0, dist1, dist2]
        test_classified[i,j] = dists.index(min(dists))
        
# report accuracy of classifier
test_scores = np.zeros(xtest.shape[0])
for i in range(0,xtest.shape[0]):
    test_scores[i] = accuracy_score(test_truth, test_classified[:,i])
















