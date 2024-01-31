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
- my accuracy is really bad, is that expected with this method?
      
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

# load in training data
train_filepath = '/Users/Reese/Documents/Research Projects/random/amath582/hw2data/train/'
test_filepath = '/Users/Reese/Documents/Research Projects/random/amath582/hw2data/test/'

#%% task 1: compile all the train samples into a matrix xtrain. Use PCA to
# investigate the dimensionality of xtrain and plot the first 5 PCA modes in
# xyz space.

files = ['jumping_1.npy', 'jumping_2.npy', 'jumping_3.npy', 'jumping_4.npy',
         'jumping_5.npy', 'running_1.npy', 'running_2.npy', 'running_3.npy',
         'running_4.npy', 'running_5.npy', 'walking_1.npy', 'walking_2.npy',
         'walking_3.npy', 'walking_4.npy', 'walking_5.npy']

# initialize xtrain to store all training data
xtrain = np.zeros([114, 1500])
i = 0

# step through all training data files and store in xtrain array
for file in files:
    traindata = np.load(train_filepath + file)
    xtrain[:,i:i+100] = traindata
    i += 100

xtrain = xtrain.T

# perform PCA, get first 5 PCA modes
pca = PCA(n_components=5)
xtrain_proj = pca.fit_transform(xtrain)

# transform back to xyz space
xtrain_5modes = pca.inverse_transform(xtrain_proj)

# prepare for plotting
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

for train_num in range(0,5):
    xyz = np.reshape(xtrain[train_num:train_num+100,:].T, (38,3,-1))
    xroot, yroot, zroot = xyz[0,0,0], xyz[0,0,1], xyz[0,0,2] # define scaling of the values
    for t in range(1,xyz.shape[2]):  
        for ij in range(0,I.shape[0]):
            xline = np.array([xyz[I[ij],0,t], xyz[J[ij],0,t]])
            yline = np.array([xyz[I[ij],1,t], xyz[J[ij],1,t]])
            zline = np.array([xyz[I[ij],2,t], xyz[J[ij],2,t]])
            # plot skeleton with lines
            ax.plot(xline,yline,zline,c='steelblue',alpha = 0.05)
        
ax.set_xlim([-r+xroot, r+xroot])
ax.set_zlim([-r+zroot, r+zroot])
ax.set_ylim([-r+yroot, r+yroot])
ax.set_title('Jumping (Unmodified)')

# plot of jumping with 5 PCA modes
fig = plt.figure(figsize=(7,5))
ax = plt.axes(projection = '3d')

for train_num in range(0,5):
    xyz_5modes = np.reshape(xtrain_5modes[train_num:train_num+100,:].T, (38,3,-1))
    xroot, yroot, zroot = xyz_5modes[0,0,0], xyz_5modes[0,0,1], xyz_5modes[0,0,2] # define scaling of the values
    for t in range(1,xyz.shape[2]):  
        for ij in range(0,I.shape[0]):
            xline = np.array([xyz_5modes[I[ij],0,t], xyz_5modes[J[ij],0,t]])
            yline = np.array([xyz_5modes[I[ij],1,t], xyz_5modes[J[ij],1,t]])
            zline = np.array([xyz_5modes[I[ij],2,t], xyz_5modes[J[ij],2,t]])
            # plot skeleton with lines
            ax.plot(xline,yline,zline,c='steelblue',alpha = 0.05)
        
ax.set_xlim([-r+xroot, r+xroot])
ax.set_zlim([-r+zroot, r+zroot])
ax.set_ylim([-r+yroot, r+yroot])
ax.set_title('Jumping (5 PCA Modes)')

# make plot of running unmodified
fig = plt.figure(figsize=(7,5))
ax = plt.axes(projection = '3d')

for train_num in range(5,10):
    xyz = np.reshape(xtrain[train_num:train_num+100,:].T, (38,3,-1))
    xroot, yroot, zroot = xyz[0,0,0], xyz[0,0,1], xyz[0,0,2] # define scaling of the values
    for t in range(1,xyz.shape[2]):  
        for ij in range(0,I.shape[0]):
            xline = np.array([xyz[I[ij],0,t], xyz[J[ij],0,t]])
            yline = np.array([xyz[I[ij],1,t], xyz[J[ij],1,t]])
            zline = np.array([xyz[I[ij],2,t], xyz[J[ij],2,t]])
            # plot skeleton with lines
            ax.plot(xline,yline,zline,c='steelblue',alpha = 0.05)
        
ax.set_xlim([-r+xroot, r+xroot])
ax.set_zlim([-r+zroot, r+zroot])
ax.set_ylim([-r+yroot, r+yroot])
ax.set_title('Running (Unmodified)')
        
      
# plot of running with 5 PCA modes
fig = plt.figure(figsize=(7,5))
ax = plt.axes(projection = '3d')

for train_num in range(5,10):
    xyz_5modes = np.reshape(xtrain_5modes[train_num:train_num+100,:].T, (38,3,-1))
    xroot, yroot, zroot = xyz_5modes[0,0,0], xyz_5modes[0,0,1], xyz_5modes[0,0,2] # define scaling of the values
    for t in range(1,xyz.shape[2]):  
        for ij in range(0,I.shape[0]):
            xline = np.array([xyz_5modes[I[ij],0,t], xyz_5modes[J[ij],0,t]])
            yline = np.array([xyz_5modes[I[ij],1,t], xyz_5modes[J[ij],1,t]])
            zline = np.array([xyz_5modes[I[ij],2,t], xyz_5modes[J[ij],2,t]])
            # plot skeleton with lines
            ax.plot(xline,yline,zline,c='steelblue',alpha = 0.05)
        
ax.set_xlim([-r+xroot, r+xroot])
ax.set_zlim([-r+zroot, r+zroot])
ax.set_ylim([-r+yroot, r+yroot])
ax.set_title('Running (5 PCA Modes)')

# make plot of walking unmodified
fig = plt.figure(figsize=(7,5))
ax = plt.axes(projection = '3d')

for train_num in range(10,15):
    xyz = np.reshape(xtrain[train_num:train_num+100,:].T, (38,3,-1))
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

for train_num in range(10,15):
    xyz_5modes = np.reshape(xtrain_5modes[train_num:train_num+100,:].T, (38,3,-1))
    xroot, yroot, zroot = xyz_5modes[0,0,0], xyz_5modes[0,0,1], xyz_5modes[0,0,2] # define scaling of the values
    for t in range(1,xyz.shape[2]):  
        for ij in range(0,I.shape[0]):
            xline = np.array([xyz_5modes[I[ij],0,t], xyz_5modes[J[ij],0,t]])
            yline = np.array([xyz_5modes[I[ij],1,t], xyz_5modes[J[ij],1,t]])
            zline = np.array([xyz_5modes[I[ij],2,t], xyz_5modes[J[ij],2,t]])
            # plot skeleton with lines
            ax.plot(xline,yline,zline,c='steelblue',alpha = 0.05)
        
ax.set_xlim([-r+xroot, r+xroot])
ax.set_zlim([-r+zroot, r+zroot])
ax.set_ylim([-r+yroot, r+yroot])
ax.set_title('Walking (5 PCA Modes)')

#%% task 2: investigate how many PCA modes to keep in order to approximate
# Xtrain up to 70%, 80%, 90%, and 95%. Plot the cumulative energy to justify
# your results.

# initialize array to store cumulative energy at each PCA mode
cum_E = np.zeros((xtrain.shape[1],))

# calculate cumulative energy for each PCA mode
for i in range (0,xtrain.shape[1]):
    pca = PCA(n_components=i)
    pca.fit(xtrain)
    cum_E[i] = np.sum(pca.explained_variance_ratio_)
    
# plot cumulative energy
fig = plt.figure(figsize=(7,5))
ax = fig.gca()
ax.plot(cum_E)
ax.set_ylabel('Cumulative Energy ($\\Sigma E_j$)')
ax.set_xlabel('Number of PCA Modes')

# answers to task 2 (from cum_E array)
# 70% - 2 PCA modes (explains 72.7% of variance)
# 80% - 3 PCA modes (explains 82.9% of variance)
# 90% - 5 PCA modes (explains 91.2% of variance)
# 95% - 7 PCA modes (explains 95.4% of variance)

#%% task 3: truncate the PCA modes to 2 and 3 modes and plot the projected
# xtrain in truncated PCA space as low dimensional 2D (PC1, PC2 coordinates)
# and 3D (PC2, PC2, PC3 coordinates) trajectories. Use colors for different
# movements and discuss the visualization and your findings.

# not plotting actual principal components, I need to plot the projection of the actual data onto the principal components
# should see a point for each time sample
# should see trajectories

# 2D PCA space
pca = PCA(n_components=2)
xtrain_proj = pca.fit_transform(xtrain)

# plot 2D PCA
fig = plt.figure(figsize=(7,5))
ax = fig.gca()
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.scatter(xtrain_proj[0:500, 0], xtrain_proj[0:500, 1], label='Jumping', alpha = 0.5)
ax.scatter(xtrain_proj[500:1000, 0], xtrain_proj[500:1000, 1], label='Running', alpha = 0.5)
ax.scatter(xtrain_proj[1000:1500, 0], xtrain_proj[1000:1500, 1], label='Walking', alpha = 0.5)
ax.legend()

# 3D PCA space
pca = PCA(n_components=3)
xtrain_proj = pca.fit_transform(xtrain)

# plot 3D PCA
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection = '3d')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.scatter3D(xtrain_proj[0:500, 0], xtrain_proj[0:500, 1], xtrain_proj[0:500, 2], label='Jumping', alpha = 0.5)
ax.scatter3D(xtrain_proj[500:1000, 0], xtrain_proj[500:1000, 1], xtrain_proj[500:1000, 2], label='Running', alpha = 0.5)
ax.scatter3D(xtrain_proj[1000:1500, 0], xtrain_proj[1000:1500, 1], xtrain_proj[1000:1500, 2], label='Walking', alpha = 0.5)
ax.legend()

#%% task 4: create a ground truth table with an integer per class and assign an
# appropriate label to each sample in xtrain. Then, for each movement compute
# its centroid in k-modes PCA space.

# 0 = jumping
# 1 = running
# 2 = walking
train_truth = [0] * 500 + [1] * 500 + [2] * 500

# calculate centroid in k-modes PCA space
# The three rows in this array represent the three types of movements (see key
# above). Each column represents the centroid of that mode PCA space. For
# example, the first three columns would be considered the "centroid" when
# looking at 3-modes PCA space

# initialize array to hold all centroids
# array is three rows (0 = jumping, 1 = running, 2 = walking)
centroids = np.zeros((3,xtrain.shape[1]))

# do PCA
pca = PCA(n_components=xtrain.shape[1])
xtrain_proj = pca.fit_transform(xtrain)

for i in range(0, xtrain_proj.shape[1]):
    centroids[0,i] = np.mean(xtrain_proj[0:500, i])
    centroids[1,i] = np.mean(xtrain_proj[500:1000, i])
    centroids[2,i] = np.mean(xtrain_proj[1000:1500, i])

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

# do PCA, xtrain_proj contains k-modes information for each training dataset point
pca = PCA(n_components=xtrain.shape[1])
xtrain_proj = pca.fit_transform(xtrain)

# for each mode, compute distance between the point and the centroid
# initialize array to store distance between point and centroid
# shape = (1500, 114) --> rows are for each sample, columns are for the classification at that k-mode PCA
classified = np.zeros((xtrain.shape[0],xtrain.shape[1]))

# should be computing the distance between the projection onto the principal components and the centroid, not the distance between the principal components and the centroid

for j in range(0, xtrain.shape[1]):
    for i in range(0, xtrain.shape[0]):
        dist0 = distance.euclidean(xtrain_proj[i,0:j+1], centroids[0,0:j+1])
        dist1 = distance.euclidean(xtrain_proj[i,0:j+1], centroids[1,0:j+1]) 
        dist2 = distance.euclidean(xtrain_proj[i,0:j+1], centroids[2,0:j+1])
        
        # choose minimum distance for each row, assign label
        dists = [dist0, dist1, dist2]
        classified[i,j] = dists.index(min(dists))
       
# 5.2: report accuracy of classifier
scores = np.zeros(xtrain.shape[1])
for i in range(0,xtrain.shape[1]):
    scores[i] = accuracy_score(train_truth, classified[:,i])

#%% task 6: load the given test samples and for each sample assign the ground
# truth label. By projecting into k-PCA space and computing the centroids,
# predict the test labels. Report the accuracy.

files = ['jumping_1t.npy', 'running_1t.npy', 'walking_1t.npy']
test_truth = [0] * 100 + [1] * 100 + [2] * 100

# initialize xtest to store all training data
xtest = np.zeros([114, 300])
i = 0

# step through all test data files and store in xtest array
for file in files:
    testdata = np.load(test_filepath + file)
    xtest[:,i:i+100] = testdata
    i += 100

xtest = xtest.T

# project test data into PCA space
xtest_proj = pca.transform(xtest)

# for each mode, compute distance between the point and the centroid
# initialize array to store distance between point and centroid
test_classified = np.zeros((xtest.shape[0],xtest.shape[1]))

for j in range(0, xtest.shape[1]):
    for i in range(0, xtest.shape[0]):
        dist0 = distance.euclidean(xtest_proj[i,0:j+1], centroids[0,0:j+1])
        dist1 = distance.euclidean(xtest_proj[i,0:j+1], centroids[1,0:j+1]) 
        dist2 = distance.euclidean(xtest_proj[i,0:j+1], centroids[2,0:j+1])
        
        # choose minimum distance for each row, assign label
        dists = [dist0, dist1, dist2]
        test_classified[i,j] = dists.index(min(dists))
        
# report accuracy of classifier
test_scores = np.zeros(xtest.shape[1])
for i in range(0,xtest.shape[1]):
    test_scores[i] = accuracy_score(test_truth, test_classified[:,i])
















