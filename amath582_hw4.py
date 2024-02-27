#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:53:09 2024

@author: Reese Barrett

Code for AMATH 582 Homework 4 assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import tqdm
import seaborn as sns
import time
import scipy.io

#%% load and set up data
# download and normalize fashionMNIST dataset
datapath = '/Users/Reese/Documents/Research Projects/random/amath582/hw4data/'

xtrain = torchvision.datasets.FashionMNIST(datapath, train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize((0.1307,),(0.3081,))]))

xtest = torchvision.datasets.FashionMNIST(datapath, train=False, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize((0.1307,),(0.3081,))]))

# create a validation set of 10%
train_indicies, val_indicies, _, _ = train_test_split(range(len(xtrain)),
                                                      xtrain.targets,
                                                      stratify=xtrain.targets,
                                                      test_size=0.1)
train_split = Subset(xtrain,train_indicies)
val_split = Subset(xtrain, val_indicies)

# set batch sizes (# of samples to propagate through before updating model)
train_batch_size = 512
test_batch_size = 256 # can be larger than train batch size

# define dataloader objects to iterate over batches and samples
train_batches = DataLoader(train_split, batch_size=train_batch_size,
                           shuffle=True)
val_batches = DataLoader(val_split, batch_size=train_batch_size, shuffle=True)
test_batches = DataLoader(xtest, batch_size=test_batch_size, shuffle=True)

#print(len(train_batches)) # number of training batches
#print(len(val_batches)) # number of validation batches
#print(len(test_batches)) # number of testing batches

#%% visualize the first sample in first 16 batches
batch_num = 0
for train_features, train_labels in train_batches:
    if batch_num == 16:
        break
    
    batch_num += 1
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap='Greys')
    plt.show()
    print(f"Label: {label}")

#%% plot N^2 impages from the dataset
def plot_images(XX, N, title):
    fig, ax = plt.subplots(N, N, figsize=(8,8))
    
    for i in range(N):
        for j in range(N):
            ax[i,j].imshow(XX[(N)*i+j], cmap='Greys')
            ax[i,j].axis('off')
            
    fig.suptitle(title,fontsize=24)
    
plot_images(xtrain.data[:64], 8, 'First 64 Training Images')

#%% task 1: design your FCN model to have an adjustable number of hidden layers
# and neurons in each layer with relu activiation.
# - input layer: dimension 784 (pixels per image)
# - output layer: dimension 10 (10 possible classes)
# - cross entropy loss
# - SGD optimizer, learning rate as adjustable parameter
# - number of epochs as adjustable parameter
# - calculate training loss and accuracy of validation set at each epoch, plot
# - perform testing
# - TOTAL 5 HYPERPARAMETERS: # training batches, # layers, # neurons in each
#   layer, learning rate, # epochs

# define your (as cool as it gets) fully connected neural network
class fashionFCN(torch.nn.Module):
    
    # initialize model layers, add additional arguments to adjust
    def __init__(self, num_layers, layer_dim):
        # num_layers = integer number of layers in model (including input and
        #              output layer)
        # layer_dim = array of length num_layers + 1 that defines the number of
        #             input and output connections of the corresponding layers
        
        # define network layer(s)
        super(fashionFCN,self).__init__()
        
        self.num_layers = num_layers
        for i in range(num_layers):
            setattr(self, f'layer{i+1}', torch.nn.Linear(layer_dim[i], layer_dim[i+1]))
    
    def forward(self, input):
       
        # define activation function(s) and how model propagates through
        # network for each output layer
        
        # models with more than one layer
        if self.num_layers > 1:
            # get output of layer 1
            output = torch.nn.functional.relu(self.layer1(input))
            
            # propagate through layers 2 - num_layers - 1
            for i in range(2, self.num_layers):
                current_layer = getattr(self, f'layer{i}')
                output = torch.nn.functional.relu(current_layer(output))
                
            # propagate through final layer, do not do activation function here
            current_layer = getattr(self, f'layer{i+1}')
            output = current_layer(output)
            
        # special case: models with only one layer
        elif self.num_layers == 1:
            output = self.layer1(input) 
            
        return output
        
# initialize neural network model with input, output, and hidden layer dimensions
model = fashionFCN(num_layers = 4, layer_dim = [784, 50, 50, 20, 10])
# define the learning rate and number of epochs
learning_rate = 0.1
epochs = 50

# initialize arrays to store loss and accuracy
train_loss_list = np.zeros((epochs,))
validation_accuracy_list = np.zeros((epochs,))
# define loss function and optimizer
loss_func = torch.nn.CrossEntropyLoss() # cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # SGD optimizer

# iterate over epochs, batches with progress bar and train/validate fashionFCN
tic = time.time()
for epoch in tqdm.trange(epochs):
    
    # fashionFCN TRAINING
    for train_features, train_labels in train_batches: 
        model.train() # set model in training mode
        train_features = train_features.reshape(-1, 28*28) # reshape images into a vector
        optimizer.zero_grad() # reset gradients
        train_outputs = model(train_features) # call model
        loss = loss_func(train_outputs, train_labels) # calculate training loss on model
        
        # perform optimization, back propagation
        loss.backward()
        optimizer.step()
        
    # record loss for the epoch
    train_loss_list[epoch] = loss.item()
    
    # VALIDATION
    # initialize array to store accuracy for all batches in each epoch
    all_val_accuracies = np.zeros(len(val_batches))
    i = 0
    for val_features, val_labels in val_batches:
        
        # tell pytorch to not pass inputs to the network for training purposes
        with torch.no_grad():
            model.eval()
            val_features = val_features.reshape(-1, 28*28) # reshape validation images into a vector
            
            # compute validation targets and accuracy
            val_outputs = model(val_features)
            correct = (torch.argmax(val_outputs, dim=1) == val_labels).type(torch.FloatTensor)
            all_val_accuracies[i] = correct.mean()
            i+=1
            
    # record accuracy for the epoch
    validation_accuracy_list[epoch] = all_val_accuracies.mean()
    
    # print training loss, validation accuracy
    print("\nEpoch: "+ str(epoch) +"; Training Loss: " + str(train_loss_list[epoch]*100) + '%')
    print("Epoch: "+ str(epoch) +"; Validation Accuracy: " + str(validation_accuracy_list[epoch]*100) + '%\n')

print('Time Elapsed: ' + str(time.time() - tic))

#%%
# TESTING
# initialize array to store accuracy for all batches in each epoch
all_test_accuracies = np.zeros(len(test_batches))
i = 0
# tell pytorch to not pass inputs to network for training purposes
with torch.no_grad():
    for test_features, test_labels in test_batches:
        model.eval()
        test_features = test_features.reshape(-1, 28*28) # reshape test images into a vector

        # compute test targets and accuracy
        test_outputs = model(test_features)
        correct = (torch.argmax(test_outputs, dim=1) == test_labels).type(torch.FloatTensor)
        all_test_accuracies[i] = correct.mean()
        i+=1

# compute total (mean) accuracy and standard deviation
test_accuracy = all_test_accuracies.mean()
test_std = all_test_accuracies.std()

print('Test Data Accuracy: ' + str(test_accuracy*100) + ' ± ' + str(test_std*100) + '%\n')

#%%    
# plot training loss and validation accuracy throughout the training epochs
plt.figure(figsize = (8, 5), dpi=200)

plt.subplot(2, 1, 1)
plt.plot(train_loss_list*100, linewidth = 3)
plt.ylabel("Training Loss (%)")
plt.xlim([0,50])

plt.subplot(2, 1, 2)
plt.plot(validation_accuracy_list *100, linewidth = 3, color = 'gold')
plt.ylabel("Validation Accuracy (%)")
plt.xlabel("Epochs")
plt.xlim([0,50])
sns.despine()

#%% task 2: find a configuration of the FCN that trains in reasonable time and
# results in a reasonable training loss curve and testing accuracy (at least
# above 85%)

# configuration is run above in task 1
# number of training batches = 106
# number of layers = 4
# number of neurons in each layer = [784, 50, 40, 20, 10]
# learning rate = 0.1
# epochs = 50
# time elapsed: approximately 7 minutes


#%% task 3: perform hyperparameter tuning from the baseline, trying the
# following configurations
# a: consider different optimizers (RMSProp Adam, SGD) with different learning rates
#    - log training loss, validation, and test accuracy
#    - compare these optimizers, observe which is most suitable, try to explain
# b: analyze the overfitting/underfitting situation of your model
#    - include dropout regularization, discuss if this improves performance
# c: consider different initializations (random normal, xavier normal, kaiming
#    he uniform), discuss how they affect training process and accuracy
# d: include normalization such as batch normalization, discuss how it affects
#    training process and accuracy


# PART 3.a ####################################################################
# initialize neural network model with input, output, and hidden layer dimensions
model = fashionFCN(num_layers = 4, layer_dim = [784, 50, 50, 20, 10])
# define the learning rate and number of epochs
learning_rate = [0.005, 0.0025, 0.005]
#learning_rate = [0.05, 0.10, 0.20]
epochs = 50

# initialize arrays to store loss and accuracy
train_loss_list = np.zeros((epochs,len(learning_rate)))
validation_accuracy_list = np.zeros((epochs,len(learning_rate)))
test_accuracy = np.zeros(len(learning_rate))
test_std = np.zeros(len(learning_rate))

for j in range(len(learning_rate)):
    # iterate over epochs, batches with progress bar and train/validate fashionFCN
    # define loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss() # cross entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[j]) # select optimizer (SGD, RMSprop, Adam)
    tic = time.time()
    for epoch in tqdm.trange(epochs):
            
        # fashionFCN TRAINING
        for train_features, train_labels in train_batches: 
            model.train() # set model in training mode
            train_features = train_features.reshape(-1, 28*28) # reshape images into a vector
            optimizer.zero_grad() # reset gradients
            train_outputs = model(train_features) # call model
            loss = loss_func(train_outputs, train_labels) # calculate training loss on model
            
            # perform optimization, back propagation
            loss.backward()
            optimizer.step()
            
        # record loss for the epoch
        train_loss_list[epoch, j] = loss.item()
        
        # VALIDATION
        # initialize array to store accuracy for all batches in each epoch
        all_val_accuracies = np.zeros(len(val_batches))
        i = 0
        for val_features, val_labels in val_batches:
            
            # tell pytorch to not pass inputs to the network for training purposes
            with torch.no_grad():
                model.eval()
                val_features = val_features.reshape(-1, 28*28) # reshape validation images into a vector
                
                # compute validation targets and accuracy
                val_outputs = model(val_features)
                correct = (torch.argmax(val_outputs, dim=1) == val_labels).type(torch.FloatTensor)
                all_val_accuracies[i] = correct.mean()
                i+=1
                
        # record accuracy for the epoch
        validation_accuracy_list[epoch, j] = all_val_accuracies.mean()
        
        # print training loss, validation accuracy
        print("\nEpoch: "+ str(epoch) +"; Training Loss: " + str(train_loss_list[epoch]*100) + '%')
        print("Epoch: "+ str(epoch) +"; Validation Accuracy: " + str(validation_accuracy_list[epoch]*100) + '%\n')
    
    print('Time Elapsed: ' + str(time.time() - tic))
    
    # TESTING
    # initialize array to store accuracy for all batches in each epoch
    all_test_accuracies = np.zeros(len(test_batches))
    i = 0
    # tell pytorch to not pass inputs to network for training purposes
    with torch.no_grad():
        for test_features, test_labels in test_batches:
            model.eval()
            test_features = test_features.reshape(-1, 28*28) # reshape test images into a vector

            # compute test targets and accuracy
            test_outputs = model(test_features)
            correct = (torch.argmax(test_outputs, dim=1) == test_labels).type(torch.FloatTensor)
            all_test_accuracies[i] = correct.mean()
            i+=1

    # compute total (mean) accuracy and standard deviation
    test_accuracy[j] = all_test_accuracies.mean()
    test_std[j] = all_test_accuracies.std()

    print('Test Data Accuracy: ' + str(test_accuracy*100) + ' ± ' + str(test_std*100) + '%\n')

#%% plot results
# plot training loss and validation accuracy throughout the training epochs
plt.figure(figsize = (5, 5), dpi=200)
labels = ['0.0010', '0.0025', '0.0050']
#labels = ['0.05', '0.10', '0.20']

plt.subplot(2, 1, 1)
plt.plot(train_loss_list*100, linewidth = 3)
plt.ylabel("Training Loss (%)")
plt.xlim([0,50])
#plt.ylim([0,100])
plt.legend(labels=labels)

plt.subplot(2, 1, 2)
plt.plot(validation_accuracy_list*100, linewidth = 3)
plt.ylabel("Validation Accuracy (%)")
plt.xlabel("Epochs")
plt.xlim([0,50])
#plt.ylim([0,100])
sns.despine()





