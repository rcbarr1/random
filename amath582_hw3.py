#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:54:40 2024

@author: Reese Barrett

Code for AMATH 582 Homework 3 assignment.
"""

import numpy as np
import struct
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
    
#%%
# perform PCA on training data (keep 16 PCA modes)
pca = PCA(n_components=16)
xtrain_pca = pca.fit_transform(xtrain)
xtrain_xyz = pca.inverse_transform((xtrain_pca))

# plot first 64 training digits in regular space
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

# plot 16 PCA modes
modes_xyz = pca.inverse_transform(np.transpose(pca.components_))
plot_digits(pca.components_, 4, "First 16 PC Modes")

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

# calculate cumulative energy for each PCA mode
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

# need 59 PCA modes to get to 85% cumulative energy
k = 59

# plot first 64 with k modes
pca = PCA(n_components=k)
xtrain_pca = pca.fit_transform(xtrain)
xtrain_xyz = pca.inverse_transform((xtrain_pca))
plot_digits(xtrain_xyz, 8, "First 64 Training Images (59 PC Modes)" )

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
# computed in steps 1-2, and apply the ridge classifier (linear) to distinguish
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

# apply the ridge classifier
RidgeCL = RidgeClassifierCV()
RidgeCL.fit(xsubtrain_pca, ysubtrain)

print("Training Score: {}".format(RidgeCL.score(xsubtrain_pca, ysubtrain)))
print("Testing Score: {}".format(RidgeCL.score(xsubtest_pca, ysubtest)))

# do cross validation
scores = cross_val_score(RidgeCL, xsubtrain_pca, ysubtrain, cv=10)
print("%0.5f accuracy with a standard deviation of %0.5f" % (scores.mean(), scores.std()))

# evaluate results by plotting confusion matrix
ysubpred = RidgeCL.predict(xsubtest_pca)

fig, ax = plt.subplots(figsize=(10, 5), dpi = 200)
disp = ConfusionMatrixDisplay.from_predictions(ysubtest, ysubpred, ax=ax)
disp.im_.set_clim(0, 1000)
ax.set_title("Confusion Matrix for RidgeClassifierCV Digit Classification (1, 8)")
ax.legend()

#%% task 5: repeat the same classification procedure for pairs of digits 3, 8
# and 2, 7. Report your results and compare them with the results in step 4,
# explaining the differences.

# for 3 and 8
# subset for 3 and 8
xsubtrain3, ysubtrain3, xsubtest3, ysubtest3 = digit_subset(3, xtrain, ytrain_labels, xtest, ytest_labels)
xsubtrain8, ysubtrain8, xsubtest8, ysubtest8 = digit_subset(8, xtrain, ytrain_labels, xtest, ytest_labels)

# stack the 3 and 8 data
xsubtrain = np.vstack((xsubtrain3, xsubtrain8))
ysubtrain = np.hstack((ysubtrain3, ysubtrain8))
xsubtest = np.vstack((xsubtest3, xsubtest8))
ysubtest = np.hstack((ysubtest3, ysubtest8))

# perform PCA with k modes (59)
pca = PCA(n_components=k) # create PCA basis with k modes
xsubtrain_pca = pca.fit_transform(xsubtrain) # fit and transform training data to PCA space
xsubtest_pca = pca.transform(xsubtest) # transform test data to PCA space

# apply the ridge classifier
RidgeCL = RidgeClassifierCV()
RidgeCL.fit(xsubtrain_pca, ysubtrain)

print("Training Score: {}".format(RidgeCL.score(xsubtrain_pca, ysubtrain)))
print("Testing Score: {}".format(RidgeCL.score(xsubtest_pca, ysubtest)))

# do cross validation
scores = cross_val_score(RidgeCL, xsubtrain_pca, ysubtrain, cv=10)
print("%0.5f accuracy with a standard deviation of %0.5f" % (scores.mean(), scores.std()))

# evaluate results by plotting confusion matrix
ysubpred = RidgeCL.predict(xsubtest_pca)

fig, ax = plt.subplots(figsize=(10, 5), dpi = 200)
disp = ConfusionMatrixDisplay.from_predictions(ysubtest, ysubpred, ax=ax)
disp.im_.set_clim(0, 1000)
ax.set_title("Confusion Matrix for RidgeClassifierCV Digit Classification (3, 8)")

# for 2 and 7
# subset for 2 and 7
xsubtrain2, ysubtrain2, xsubtest2, ysubtest2 = digit_subset(2, xtrain, ytrain_labels, xtest, ytest_labels)
xsubtrain7, ysubtrain7, xsubtest7, ysubtest7 = digit_subset(7, xtrain, ytrain_labels, xtest, ytest_labels)

# stack the 2 and 7 data
xsubtrain = np.vstack((xsubtrain2, xsubtrain7))
ysubtrain = np.hstack((ysubtrain2, ysubtrain7))
xsubtest = np.vstack((xsubtest2, xsubtest7))
ysubtest = np.hstack((ysubtest2, ysubtest7))

# perform PCA with k modes (59)
pca = PCA(n_components=k) # create PCA basis with k modes
xsubtrain_pca = pca.fit_transform(xsubtrain) # fit and transform training data to PCA space
xsubtest_pca = pca.transform(xsubtest) # transform test data to PCA space

# apply the ridge classifier
RidgeCL = RidgeClassifierCV()
RidgeCL.fit(xsubtrain_pca, ysubtrain)

print("Training Score: {}".format(RidgeCL.score(xsubtrain_pca, ysubtrain)))
print("Testing Score: {}".format(RidgeCL.score(xsubtest_pca, ysubtest)))

# do cross validation
scores = cross_val_score(RidgeCL, xsubtrain_pca, ysubtrain, cv=10)
print("%0.5f accuracy with a standard deviation of %0.5f" % (scores.mean(), scores.std()))

# evaluate results by plotting confusion matrix
ysubpred = RidgeCL.predict(xsubtest_pca)

fig, ax = plt.subplots(figsize=(10, 5), dpi = 200)
disp = ConfusionMatrixDisplay.from_predictions(ysubtest, ysubpred, ax=ax)
disp.im_.set_clim(0, 1000)
ax.set_title("Confusion Matrix for RidgeClassifierCV Digit Classification (2, 7)")

#%% step 6: use all the digits and perform multiclass classification with
# ridge, KNN, and LDA classifiers. Report your results and discuss how they
# compare between the methods. Which method performs the best?

# perform PCA with k modes (59)
pca = PCA(n_components=k) # create PCA basis with k modes
xtrain_pca = pca.fit_transform(xtrain) # fit and transform training data to PCA space
xtest_pca = pca.transform(xtest) # transform test data to PCA space

# ridge classification
# apply the ridge classifier
RidgeCL = RidgeClassifierCV()
RidgeCL.fit(xtrain_pca, ytrain_labels)

print("Training Score: {}".format(RidgeCL.score(xtrain_pca, ytrain_labels)))
print("Testing Score: {}".format(RidgeCL.score(xtest_pca, ytest_labels)))

# do cross validation
scores = cross_val_score(RidgeCL, xtrain_pca, ytrain_labels, cv=10)
print("%0.5f accuracy with a standard deviation of %0.5f" % (scores.mean(), scores.std()))

# evaluate results by plotting confusion matrix
ytest_pred = RidgeCL.predict(xtest_pca)

fig, ax = plt.subplots(figsize=(10, 5), dpi = 200)
disp = ConfusionMatrixDisplay.from_predictions(ytest_labels, ytest_pred, ax=ax)
disp.im_.set_clim(0, 1000)
ax.set_title("Confusion Matrix for RidgeClassifierCV Digit Classification")

#%% KNN classification
# apply the KNN classifier
KNNCL = KNeighborsClassifier(n_neighbors=3)
KNNCL.fit(xtrain_pca,ytrain_labels)
    
training_score = KNNCL.score(xtrain_pca, ytrain_labels)
testing_score = KNNCL.score(xtest_pca, ytest_labels)
print("Training Score: {}".format(training_score))
print("Testing Score: {}".format(testing_score))

# do cross validation
scores = cross_val_score(KNNCL, xtrain_pca, ytrain_labels, cv=10)
cv_mean = scores.mean()
cv_std = scores.std()
print("%0.5f accuracy with a standard deviation of %0.5f" % (cv_mean, cv_std))
 
ytest_pred = KNNCL.predict(xtest_pca)
 
fig, ax = plt.subplots(figsize=(10, 5), dpi = 200)
disp = ConfusionMatrixDisplay.from_predictions(ytest_labels, ytest_pred, ax=ax)
disp.im_.set_clim(0, 1000)
ax.set_title("Confusion Matrix for KNeighborsClassifier Digit Classification (k = 3)")

#%% test different k for kNN classifier
def knn_classifier(k, xtrain_pca, ytrain_labels, xtest_pca, ytest_labels):
    KNNCL = KNeighborsClassifier(n_neighbors=k)
    KNNCL.fit(xtrain_pca,ytrain_labels)
        
    training_score = KNNCL.score(xtrain_pca, ytrain_labels)
    testing_score = KNNCL.score(xtest_pca, ytest_labels)
    print("Training Score: {}".format(training_score))
    print("Testing Score: {}".format(testing_score))
    
    # do cross validation
    scores = cross_val_score(KNNCL, xtrain_pca, ytrain_labels, cv=10)
    cv_mean = scores.mean()
    cv_std = scores.std()
    print("%0.2f accuracy with a standard deviation of %0.2f" % (cv_mean, cv_std))
    
    return training_score, testing_score, cv_mean, cv_std
    
# preallocate to store data
ks = range(1, 6)
training_scores = np.zeros(len(ks))
testing_scores = np.zeros(len(ks))
cv_means = np.zeros(len(ks))
cv_stds = np.zeros(len(ks))

# run knn for various k
i = 0
for k in ks:
    training_scores[i], testing_scores[i], cv_means[i], cv_stds[i] = knn_classifier(k, xtrain_pca, ytrain_labels, xtest_pca, ytest_labels)
    i += 1

# plot results of various k
fig = plt.figure(figsize=(10, 5), dpi = 200)
ax = fig.gca()
ax.plot(list(ks), training_scores, label='Train Data Score')
ax.plot(list(ks), testing_scores, label='Test Data Score')
ax.errorbar(list(ks), cv_means, yerr=cv_stds, label='Cross-Validation Mean')
ax.set_title('Evaluation of k Nearest Neighbors')
ax.set_xlabel('Number of Neighbors (k)')
ax.set_ylabel('Accuracy Score')
ax.legend()

#%% LDA classification
# apply the LDA classifier
LDACL = LinearDiscriminantAnalysis()
LDACL.fit(xtrain_pca, ytrain_labels)

print("Training Score: {}".format(LDACL.score(xtrain_pca, ytrain_labels)))
print("Testing Score: {}".format(LDACL.score(xtest_pca, ytest_labels)))

# do cross validation
scores = cross_val_score(LDACL, xtrain_pca, ytrain_labels, cv=10)
print("%0.5f accuracy with a standard deviation of %0.5f" % (scores.mean(), scores.std()))

# evaluate results by plotting confusion matrix
ytest_pred = LDACL.predict(xtest_pca)

fig, ax = plt.subplots(figsize=(10, 5), dpi = 200)
disp = ConfusionMatrixDisplay.from_predictions(ytest_labels, ytest_pred, ax=ax)
disp.im_.set_clim(0, 1000)
ax.set_title("Confusion Matrix for LinearDiscriminantAnalysis Digit Classification")






