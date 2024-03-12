#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:25:02 2024

@author: Reese Barrett

Code for AMATH 582 Homework 5 assignment.

BIG QUESTION: can I take ctd pressure, salinity, temperature, month, day,
nitrate, and oxygen and accurately classify the location of a profile taken at
a time series measurement?
 - how many of these predictors are necessary? (how to optimize ocean
   measurements)
 - can be used to determine which ocean environments are more similar and which
   are more different (also quality control?)
 - does a classifier or FCN work better? (maybe, depending on time)
 
 # columns to keep for X: date, time, latitude, longitude, pressure/depth,
 #                        temperature, salinity, oxygen
 # columns to keep for y: time series site
 #
 # variable names, units (source: Table1_ProductVariables_v2.csv, site above)
 # TimeSeriesSite:     Time-Series Station unique identifier
 # DATE:               Date in yyyymmdd
 # TIME:               Time in hhmm (utc)
 # LATITUDE:           Latitude (degN)
 # LONGITUDE:          Longitude (degE)
 # CTDPRES:            Depth of sample (dbar)
 # CTDTEMP:            Temperature of sample (degC, ITS-90)
 # CTDSAL:             Sensor salinity (PSS-78)
 # CTDSAL_FLAG_W:      WOCE sensor quality flag (2 = acceptable)
 # OXYGEN:             Bottle oxygen (µmol/kg)
 # OXYGEN_FLAG_W:      WOCE bottle quality flag (2 = acceptable)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

#%% step 1a: read in SPOTS data, make it prettier
# data source: https://www.bco-dmo.org/dataset/896862
datapath = '/Users/Reese/Documents/Research Projects/random/amath582/data/'
spots_data = pd.read_csv(datapath + 'spots.csv', na_values='-999')

# get rid of year from 'DATE' column, turn into 'DAYOFYEAR'
dates = pd.to_datetime(spots_data['DATE'], format='%Y%m%d')
day_of_year = dates.dt.dayofyear
spots_data.insert(6, "day_of_year", day_of_year, True)

# keep only rows with all variables available
spots = spots_data[['TimeSeriesSite', 'day_of_year', 'TIME', 'LATITUDE',
                    'LONGITUDE', 'CTDPRS', 'CTDTMP', 'CTDSAL',
                    'CTDSAL_FLAG_W', 'OXYGEN', 'OXYGEN_FLAG_W']]
spots = spots.dropna() # get rid of rows with any nans

# do quality control (keep only rows with WOCE flags = 2
spots = spots[spots['CTDSAL_FLAG_W'] == 2]
spots = spots[spots['OXYGEN_FLAG_W'] == 2]

# drop cariaco data (pulling that in later)
spots = spots[spots['TimeSeriesSite'] != 'CARIACO']

# drop columns not included in larger set
spots = spots[['TimeSeriesSite', 'day_of_year', 'LATITUDE', 'LONGITUDE',
               'CTDPRS', 'CTDTMP', 'CTDSAL', 'OXYGEN']]

# rename columns
spots = spots.rename(columns={'LATITUDE':'Latitude', 'LONGITUDE':'Longitude',
                              'CTDPRS':'Pressure', 'CTDTMP':'Temperature',
                              'CTDSAL':'Salinity', 'OXYGEN':'Oxygen'})

#%% step 1b: read in BATS data, make it prettier
# data source: https://bats.bios.asu.edu/bats-data/
bats_datapath = '/Users/Reese/Documents/Research Projects/random/amath582/data/bats/'
bats_files = os.listdir(bats_datapath)
bats_files = [f for f in bats_files if f[-3:] == 'xls']
bats_data = pd.DataFrame()

# loop through all files, concatenate into one
for f in bats_files:
    bats = pd.read_excel(bats_datapath + f, na_values=-999, header=None,
                         names=['cruise number', 'decimal year',
                                'Latitude (N)', 'Longitude (W)',
                                'Pressure (dbar)', 'Depth (m)',
                                'Temperature (ITS-90, C)',
                                'Conductivity (S/m)', 'Salinity (PSS-78)',
                                'Dissolved Oxygen (umol/kg)',
                                'Beam Attenuation Coefficient (1/m)',
                                'Fluorescence (relative fluorescence units)',
                                'PAR(uE/m^2/s)'])
    
    bats_data = pd.concat([bats_data, bats], ignore_index=True)

# get rid of year from 'DATE' column, turn into 'DAYOFYEAR'
bats_data = bats_data[bats_data['decimal year']<1970000] # keep only decimal year data
bats_data = bats_data[bats_data['Dissolved Oxygen (umol/kg)']>=0] # keep only positive DO data

# convert to day of year
bats_data.loc[:,'decimal year'] %= 1
bats_data.loc[:,'decimal year'] = np.round(bats_data.loc[:,'decimal year']*365.25)
bats_data = bats_data.rename(columns={'decimal year':'day_of_year'})

# keep only rows with all variables available
bats = bats_data[['day_of_year', 'Latitude (N)', 'Longitude (W)',
             'Pressure (dbar)', 'Temperature (ITS-90, C)', 'Salinity (PSS-78)',
             'Dissolved Oxygen (umol/kg)']]
bats = bats.dropna() # get rid of rows with any nans

# rename columns, add TimeSeriesSite column
bats = bats.rename(columns={'Latitude (N)':'Latitude',
                            'Longitude (W)':'Longitude',
                            'Pressure (dbar)':'Pressure',
                            'Temperature (ITS-90, C)':'Temperature',
                            'Salinity (PSS-78)':'Salinity',
                            'Dissolved Oxygen (umol/kg)':'Oxygen'})


# convert longitude to degE (easy conversion because only in Bermuda)
bats.loc[:,'Longitude'] *= -1

# cut out some bats data (too much to process, keep every hundredth row)
bats = bats.iloc[::100, :]

#%% step 1c: read in CARIACO data
# data source: https://www.bco-dmo.org/dataset/3092#data-files
datapath = '/Users/Reese/Documents/Research Projects/random/amath582/data/'
cariaco_data = pd.read_csv(datapath + 'ctd.csv', na_values='nd')

# get rid of year from 'DATE' column, turn into 'DAYOFYEAR'
dates = pd.to_datetime(cariaco_data['Date'], format='%Y-%m-%d')
day_of_year = dates.dt.dayofyear
cariaco_data.insert(6, "day_of_year", day_of_year, True)

# keep only rows with all variables available
cariaco = cariaco_data[['day_of_year', 'Latitude', 'Longitude', 'press',
                        'temp', 'sal', 'O2_ml_L']]
cariaco = cariaco.dropna() # get rid of rows with any nans

# rename columns
cariaco = cariaco.rename(columns={'press':'Pressure', 'temp':'Temperature',
                                  'sal':'Salinity', 'O2_ml_L':'Oxygen'})

# convert oxygen units to umol/L
# source: https://www.nodc.noaa.gov/OC5/WOD/wod18-notes.html#:~:text=1%20ml%2Fl%20of%20O,of%201025%20kg%2Fm3.
cariaco.loc[:,'Oxygen'] *= 43.570

# cut out some cariaco data (too much to process, keep every tenth row)
cariaco = cariaco.iloc[::10, :]

#%% step 1d: turn data into inputs and labels arrays

# see how much data in each label category
# number of points with good OXYGEN data: BATS = 26153, ALOHA = 21930, CARIACO = 3384, CVOO = 524, K2 = 596
# number of points with only S, T, P: BATS = 26153, ALOHA = 86651, CARIACO = 4103, CVOO = 1166, K2 = 609

# LABEL KEY
# BATS      = 0 (bermuda atlantic time series)
# ALOHA     = 1 (hawaii ocean time series)
# CARIACO   = 2 (cariaco basin time series)
# CVOO      = 3 (cape verde ocean observatory)
# K2        = 4 (time series K2)

# create labels for bats
bats['Label'] = 0

# create labels for cariaco
cariaco['Label'] = 2

# convert TimeSeriesSite from string to integer label
spots['Label'] = 0
spots.loc[(spots['TimeSeriesSite'] == 'ALOHA'),'Label'] = 1
spots.loc[(spots['TimeSeriesSite'] == 'CVOO'),'Label'] = 3
spots.loc[(spots['TimeSeriesSite'] == 'K2'),'Label'] = 4

# set up data and labels arrays
Xdata_bats = bats[['day_of_year', 'Pressure', 'Temperature', 'Salinity', 'Oxygen']]
Xdata_cariaco = cariaco[['day_of_year', 'Pressure', 'Temperature', 'Salinity', 'Oxygen']]
Xdata_spots = spots[['day_of_year', 'Pressure', 'Temperature', 'Salinity', 'Oxygen']]

Xdata = pd.concat([Xdata_bats, Xdata_cariaco, Xdata_spots], ignore_index=True)
ylabels = pd.concat([bats['Label'], cariaco['Label'], spots['Label']], ignore_index=True)

# get indicies of each label
bats_idx = ylabels.index[ylabels == 0].tolist()
hot_idx = ylabels.index[ylabels == 1].tolist()
cariaco_idx = ylabels.index[ylabels == 2].tolist()
cvoo_idx = ylabels.index[ylabels == 3].tolist()
k2_idx = ylabels.index[ylabels == 4].tolist()

#%% step 2a: show what the data looks like as map of where stations are located

# set up map
fig = plt.figure(figsize=(7,6), dpi=200)
ax = plt.axes(projection=ccrs.PlateCarree())
#ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180)) # paciifc-centered view
#ax = plt.axes(projection=ccrs.Orthographic(0,90)) # arctic-centered view (turn off "extent" variable)
#ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0)
g1.top_labels = False
g1.right_labels = False
ax.add_feature(cfeature.LAND,color='silver')
extent = [-180, 180, -90, 90]
ax.set_extent(extent)

# pull latitude and longitudes (degN, degE)
# BATS      (31.83, -64.16)
# HOT     (22.75, -158.00)
# CARIACO   (10.50, -64.67)
# CVOO      (17.60, -24.30)
# K2        (47.00, 160.00)

lat = [31.83, 22.75, 10.50, 17.60, 47.00]
lon = [-64.16, -158.00, -64.67, -24.30, 160.00]

ax.scatter(lon,lat,transform=ccrs.PlateCarree(),marker='*',edgecolors='none',s=150,color='firebrick')

ax.set_title('Time Series Stations with Available\nTemperature, Pressure, Salinity, and Oxygen Data')
fig.text(0.175, 0.515, 'HOT', fontsize=12)
fig.text(0.385, 0.59, 'BATS', fontsize=12)
fig.text(0.86, 0.58, 'K2', fontsize=12)
fig.text(0.265, 0.487, 'CARIACO', fontsize=12)
fig.text(0.465, 0.565, 'CVOO', fontsize=12)

#%% step 2b: plots of each variable with a separate line for each station over time

def plot_var(bats, spots, var_name, ax):
    
    # BATS
    bats = bats.sort_values(by=['day_of_year'])
    x = bats.loc[:,'day_of_year']
    y = bats.loc[:,var_name]
    ax.scatter(x, y, label='BATS', alpha = 0.05)
    
    # HOT
    spots = spots.sort_values(by=['day_of_year'])
    x = spots.loc[(spots['TimeSeriesSite'] == 'ALOHA'),'day_of_year']
    y = spots.loc[(spots['TimeSeriesSite'] == 'ALOHA'), var_name]
    ax.scatter(x, y, label='HOT', alpha = 0.01)
    
    # CARIACO
    x = spots.loc[(spots['TimeSeriesSite'] == 'CARIACO'),'day_of_year']
    y = spots.loc[(spots['TimeSeriesSite'] == 'CARIACO'), var_name]
    ax.scatter(x, y, label='CARIACO', alpha = 0.2)
    
    # CVOO
    x = spots.loc[(spots['TimeSeriesSite'] == 'CVOO'),'day_of_year']
    y = spots.loc[(spots['TimeSeriesSite'] == 'CVOO'), var_name]
    ax.scatter(x, y, label='CVOO', alpha = 0.2)
    
    # K2
    x = spots.loc[(spots['TimeSeriesSite'] == 'K2'),'day_of_year']
    y = spots.loc[(spots['TimeSeriesSite'] == 'K2'), var_name]
    ax.scatter(x, y, label='K2', alpha = 0.2)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7,5), dpi=500, sharex=True, sharey=False)
fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.tight_layout(w_pad=2.5)
plt.xlabel('Day of Year')

plot_var(bats, spots, 'Temperature', axs[0,0])
axs[0,0].set_xlim([0, 365])
axs[0,0].set_ylim([0, 35])
axs[0,0].set_ylabel('Temperature (ºC)')

plot_var(bats, spots, 'Pressure', axs[0,1])
axs[0,1].set_ylabel('Pressure (dbar)')
#axs[0,1].set_ylim([32, 40])

plot_var(bats, spots, 'Salinity', axs[1,0])
axs[1,0].set_ylabel('Salinity (PSU)')

plot_var(bats, spots, 'Oxygen', axs[1,1])
axs[1,1].set_ylabel('Dissolved Oxygen ($µmol$ $kg^{-1})$')
axs[1,1].legend(bbox_to_anchor = (-1.2, -0.25), loc='upper left', ncols=5)

plt.title('Patterns in Time Series Temperature, Pressure, Salinity, and Oxygen', pad=10)

#%% step 3: perform PCA
# - see which parameters are most necessary to describe data

# calculate and plot cumulative energy
# initialize array to store cumulative energy at each PCA mode
cum_E = np.zeros((Xdata.shape[1],))

# calculate cumulative energy for each PCA mode3
for i in range (0,Xdata.shape[1]):
    pca = PCA(n_components=i+1)
    pca.fit(Xdata)
    cum_E[i] = np.sum(pca.explained_variance_ratio_)
    
# plot cumulative energy
fig = plt.figure(figsize=(5,3),dpi=200)
ax = fig.gca()
ax.plot([1, 2, 3, 4, 5], cum_E*100,linewidth=2)

ax.set_ylabel('$\\Sigma E_j$ (%)')
ax.set_xlabel('Number of PCA Modes')
ax.set_title('Cumulative Energy ($\\Sigma E_j$) Achieved by PCA')
ax.set_xlim([0,5])
ax.set_xticks([0, 1, 2, 3, 4, 5])

# plot PCA
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6,3), dpi=500, sharex=True, sharey=False)

# plot 2D PCA
pca = PCA(n_components=2)
xtrain_pca = pca.fit_transform(Xdata)
axs[0].set_xlabel('PC 1')
axs[0].set_ylabel('PC 2')
axs[0].scatter(xtrain_pca[bats_idx, 0], xtrain_pca[bats_idx, 1], label='BATS', alpha =0.2)
axs[0].scatter(xtrain_pca[hot_idx, 0], xtrain_pca[hot_idx, 1], label='HOT', alpha = 0.2)
axs[0].scatter(xtrain_pca[cariaco_idx, 0], xtrain_pca[cariaco_idx, 1], label='CARIACO', alpha = 0.2)
axs[0].scatter(xtrain_pca[cvoo_idx, 0], xtrain_pca[cvoo_idx, 1], label='CVOO', alpha = 0.2)
axs[0].scatter(xtrain_pca[k2_idx, 0], xtrain_pca[k2_idx, 1], label='K2', alpha = 0.2)

# plot 3D PCA
pca = PCA(n_components=3)
xtrain_pca = pca.fit_transform(Xdata)
axs[1].remove()
axs[1] = fig.add_subplot(1,2,2,projection='3d')
axs[1].set_xlabel('PC 1')
axs[1].set_ylabel('PC 2', labelpad=9)
axs[1].set_zlabel('PC 3', labelpad=9)
axs[1].scatter3D(xtrain_pca[bats_idx, 0], xtrain_pca[bats_idx, 1], xtrain_pca[bats_idx, 2], label='BATS', alpha = 0.05)
axs[1].scatter3D(xtrain_pca[hot_idx, 0], xtrain_pca[hot_idx, 1], xtrain_pca[hot_idx, 2], label='HOT', alpha = 0.05)
axs[1].scatter3D(xtrain_pca[cariaco_idx, 0], xtrain_pca[cariaco_idx, 1], xtrain_pca[cariaco_idx, 2], label='CARIACO', alpha = 0.6)
axs[1].scatter3D(xtrain_pca[cvoo_idx, 0], xtrain_pca[cvoo_idx, 1], xtrain_pca[cvoo_idx, 2], label='CVOO', alpha = 0.6)
axs[1].scatter3D(xtrain_pca[k2_idx, 0], xtrain_pca[k2_idx, 1], xtrain_pca[k2_idx, 2], label='K2', alpha = 0.6)
axs[1].legend(loc='upper center', bbox_to_anchor=(-0.1, -0.25), ncol=5)

plt.title('     Two- and Three-Dimensional Prinicipal Component Analysis            ', x = -0.05, y = 1.25)

#%% step 4a: build classifier
# - test with varying predictors and classifiers
# - different PCA modes?
# - binary classification?
# - NN?

# ridge classification function
def ridge_classification(Xdata, ylabels, test_size, ax):
    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(Xdata, ylabels, test_size=test_size, random_state=42)
    
    # apply the ridge classifier
    RidgeCL = RidgeClassifierCV()
    RidgeCL.fit(X_train, y_train)
    
    print("Training Score: {}".format(RidgeCL.score(X_train, y_train)))
    print("Testing Score: {}".format(RidgeCL.score(X_test, y_test)))
    
    # do cross validation
    scores = cross_val_score(RidgeCL, X_train, y_train, cv=10)
    print("%0.5f accuracy with a standard deviation of %0.5f" % (scores.mean(), scores.std()))
    
    # evaluate results by plotting confusion matrix
    ysubpred = RidgeCL.predict(X_test)
    
    disp = ConfusionMatrixDisplay.from_predictions(y_test, ysubpred, ax=ax)
    disp.im_.set_clim(0, 3500)
    disp.im_.colorbar.remove()
    ax.set_title("Ridge")
    
# kNN classification function
def kNN_classification(Xdata, ylabels, test_size, k, ax, title):
    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(Xdata, ylabels, test_size=test_size, random_state=42)
    
    # apply the LDA classifier
    KNNCL = KNeighborsClassifier(n_neighbors=k)
    KNNCL.fit(X_train,y_train)
    
    # score training and testing sets
    train_score = KNNCL.score(X_train, y_train)
    test_score = KNNCL.score(X_test, y_test)
    print("Training Score: {}".format(train_score))
    print("Testing Score: {}".format(test_score))
    
    # do cross validation
    scores = cross_val_score(KNNCL, X_train, y_train, cv=10)
    cross_val_score_mean = scores.mean()
    cross_val_score_std = scores.std()
    print("%0.5f accuracy with a standard deviation of %0.5f" % (cross_val_score_mean, cross_val_score_std))
    
    # evaluate results by plotting confusion matrix
    ysubpred = KNNCL.predict(X_test)
    
    disp = ConfusionMatrixDisplay.from_predictions(y_test, ysubpred, ax=ax)
    disp.im_.set_clim(0, 3500)
    disp.im_.colorbar.remove()
    ax.set_title(title)
    
    compare_labels = pd.DataFrame(data={'Test' : y_test.reset_index(drop=True), 'Predicted' : ysubpred})
    bats_correct = len(compare_labels[(compare_labels['Test'] == compare_labels['Predicted']) & (compare_labels['Test'] == 0)]) / len(compare_labels[(compare_labels['Test'] == 0)])
    print('BATS correct: ' + str(bats_correct*100) + '\n')
    
    hot_correct = len(compare_labels[(compare_labels['Test'] == compare_labels['Predicted']) & (compare_labels['Test'] == 1)]) / len(compare_labels[(compare_labels['Test'] == 1)])
    print('HOT correct: ' + str(hot_correct*100) + '\n')
    
    cariaco_correct = len(compare_labels[(compare_labels['Test'] == compare_labels['Predicted']) & (compare_labels['Test'] == 2)]) / len(compare_labels[(compare_labels['Test'] == 2)])
    print('CARIACO correct: ' + str(cariaco_correct*100) + '\n')
    
    cvoo_correct = len(compare_labels[(compare_labels['Test'] == compare_labels['Predicted']) & (compare_labels['Test'] == 3)]) / len(compare_labels[(compare_labels['Test'] == 3)])
    print('CVOO correct: ' + str(cvoo_correct*100) + '\n')
    
    k2_correct = len(compare_labels[(compare_labels['Test'] == compare_labels['Predicted']) & (compare_labels['Test'] == 4)]) / len(compare_labels[(compare_labels['Test'] == 4)])
    print('K2 correct: ' + str(k2_correct*100) + '\n')
    
    return train_score, test_score, cross_val_score_mean, cross_val_score_std
    
    
# LDA classification function
def LDA_classification(Xdata, ylabels, test_size, ax):
    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(Xdata, ylabels, test_size=test_size, random_state=42)
    
    # apply the LDA classifier
    LDACL = LinearDiscriminantAnalysis()
    LDACL.fit(X_train, y_train)
    
    print("Training Score: {}".format(LDACL.score(X_train, y_train)))
    print("Testing Score: {}".format(LDACL.score(X_test, y_test)))
    
    # do cross validation
    scores = cross_val_score(LDACL, X_train, y_train, cv=10)
    print("%0.5f accuracy with a standard deviation of %0.5f" % (scores.mean(), scores.std()))
    
    # evaluate results by plotting confusion matrix
    ysubpred = LDACL.predict(X_test)
    
    disp = ConfusionMatrixDisplay.from_predictions(y_test, ysubpred, ax=ax)
    disp.im_.set_clim(0, 3500)
    disp.im_.colorbar.remove()
    ax.set_title("LDA")

#%% step 4b: determine which k is best
ks = range(1,20)
train_scores = np.zeros(len(ks))
test_scores = np.zeros(len(ks))
cross_val_score_means = np.zeros(len(ks))
cross_val_score_stds = np.zeros(len(ks))

# loop through ks
for i in ks:
    fig, ax = plt.subplots(figsize=(10, 5), dpi = 200)
    train_scores[i-1], test_scores[i-1], cross_val_score_means[i-1], cross_val_score_stds[i-1] = kNN_classification(Xdata, ylabels, 0.15, i, ax, '')

# plot scores for each k
fig = plt.figure(figsize=(7, 4), dpi = 200)
ax = fig.gca()
ax.plot(list(ks), train_scores, label='Train Data Score')
ax.plot(list(ks), test_scores, label='Test Data Score')
ax.errorbar(list(ks), cross_val_score_means, yerr=cross_val_score_stds, label='Cross-Validation Mean')
ax.set_title('Evaluation of k Nearest Neighbors')
ax.set_xlabel('Number of Neighbors (k)')
ax.set_ylabel('Accuracy Score')
ax.legend()
ax.set_xticks(list(ks))

#%% step 4c: do classification with unaltered data

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8,5), dpi=500, sharex=True, sharey=True)

ridge_classification(Xdata, ylabels, 0.15, axs[0])
kNN_classification(Xdata, ylabels, 0.15, 1, axs[1], "kNN")
LDA_classification(Xdata, ylabels, 0.15, axs[2])

#%% step 4d: do classification with PCA
pca = PCA(n_components=3)
Xdata_pca = pca.fit_transform(Xdata)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8,5), dpi=500, sharex=True, sharey=True)

ridge_classification(Xdata_pca, ylabels, 0.15, axs[0])
kNN_classification(Xdata_pca, ylabels, 0.15, 1, axs[1], "kNN")
LDA_classification(Xdata_pca, ylabels, 0.15, axs[2])

# plot classifier scores with PC modes
pc_modes = range(0,Xdata.shape[1])

train_scores = np.zeros((len(pc_modes), 3))
test_scores = np.zeros((len(pc_modes), 3))
cross_val_score_means = np.zeros((len(pc_modes), 3))
cross_val_score_stds = np.zeros((len(pc_modes), 3))

for i in pc_modes:
    pca = PCA(n_components=i+1)
    Xdata_pca = pca.fit_transform(Xdata)
    X_train, X_test, y_train, y_test = train_test_split(Xdata_pca, ylabels, test_size=0.15, random_state=42)

    
    # do ridge
    RidgeCL = RidgeClassifierCV()
    RidgeCL.fit(X_train, y_train)
    train_scores[i, 0] = RidgeCL.score(X_train, y_train)
    test_scores[i, 0] = RidgeCL.score(X_test, y_test)
    scores = cross_val_score(RidgeCL, X_train, y_train, cv=10)
    cross_val_score_means[i, 0] = scores.mean()
    cross_val_score_stds[i, 0] = scores.std()
    
    # do kNN
    KNNCL = KNeighborsClassifier(n_neighbors=1)
    KNNCL.fit(X_train, y_train)
    train_scores[i, 1] = KNNCL.score(X_train, y_train)
    test_scores[i, 1] = KNNCL.score(X_test, y_test)
    scores = cross_val_score(KNNCL, X_train, y_train, cv=10)
    cross_val_score_means[i, 1] = scores.mean()
    cross_val_score_stds[i, 1] = scores.std()
    
    
    # do LDA
    LDACL = LinearDiscriminantAnalysis()
    LDACL.fit(X_train, y_train)
    train_scores[i, 2] = LDACL.score(X_train, y_train)
    test_scores[i, 2] = LDACL.score(X_test, y_test)
    scores = cross_val_score(LDACL, X_train, y_train, cv=10)
    cross_val_score_means[i, 2] = scores.mean()
    cross_val_score_stds[i, 2] = scores.std()
    

# plot scores for each k
fig = plt.figure(figsize=(7, 3), dpi = 200)
ax = fig.gca()

ax.plot(list(range(1,6)), train_scores[:,0], c='mediumseagreen', ls='-', label='Ridge Train')
ax.plot(list(range(1,6)), test_scores[:,0], c='mediumseagreen', ls='--', label='Ridge Test')
ax.errorbar(list(range(1,6)), cross_val_score_means[:,0], yerr=cross_val_score_stds[:,0], c='mediumseagreen', ls=':', label='Ridge CV')

ax.plot(list(range(1,6)), train_scores[:,1], c='darksalmon', ls='-', label='kNN Train')
ax.plot(list(range(1,6)), test_scores[:,1], c='darksalmon', ls='--', label='kNN Test')
ax.errorbar(list(range(1,6)), cross_val_score_means[:,1], yerr=cross_val_score_stds[:,1], c='darksalmon', ls=':', label='kNN CV')

ax.plot(list(range(1,6)), train_scores[:,2], c='steelblue', ls='-', label='LDA Train')
ax.plot(list(range(1,6)), test_scores[:,2], c='steelblue', ls='--', label='LDA Test')
ax.errorbar(list(range(1,6)), cross_val_score_means[:,2], yerr=cross_val_score_stds[:,2], c='steelblue', ls=':', label='LDA CV')

ax.set_title('Classifier Accuracy with Varying Principal Component Modes')
ax.set_xlabel('Principal Component Modes')
ax.set_ylabel('Accuracy Score')

handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1,6,2,3,7,4,5,8]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right', ncols=3, bbox_to_anchor=(0.86, -0.2))

ax.set_xticks(list(range(1,6)))
#%% step 4e: do different combinations of variables for classification:
# variable options: day of year, pressure, temperature, oxygen, salinity

fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(12,6), dpi=500, sharex=True, sharey=True)

# combination 1: day of year, pressure, temperature
Xdata_sub = Xdata[['day_of_year', 'Pressure', 'Temperature']]
kNN_classification(Xdata_sub, ylabels, 0.15, 1, axs[0,0], 'Day, P, T')

# combination 2: day of year, pressure, salinity
Xdata_sub = Xdata[['day_of_year', 'Pressure', 'Salinity']]
kNN_classification(Xdata_sub, ylabels, 0.15, 1, axs[0,1], 'Day, P, S')

# combination 3: day of year, pressure, oxygen
Xdata_sub = Xdata[['day_of_year', 'Pressure', 'Oxygen']]
kNN_classification(Xdata_sub, ylabels, 0.15, 1, axs[0,2], 'Day, P, O')

# combination 4: day of year, temperature, salinity
Xdata_sub = Xdata[['day_of_year', 'Temperature', 'Salinity']]
kNN_classification(Xdata_sub, ylabels, 0.15, 1, axs[0,3], 'Day, T, S')

# combination 5: day of year, temperature, oxygen
Xdata_sub = Xdata[['day_of_year', 'Temperature', 'Oxygen']]
kNN_classification(Xdata_sub, ylabels, 0.15, 1, axs[0,4], 'Day, T, 0')

# combination 6: day of year, salinity, oxygen
Xdata_sub = Xdata[['day_of_year', 'Salinity', 'Oxygen']]
kNN_classification(Xdata_sub, ylabels, 0.15, 1, axs[1,0], 'Day, S, O')

# combination 7: pressure, temperature, salinity
Xdata_sub = Xdata[['Pressure', 'Temperature', 'Salinity']]
kNN_classification(Xdata_sub, ylabels, 0.15, 1, axs[1,1], 'P, T, S')

# combination 8: pressure, temperature, oxygen
Xdata_sub = Xdata[['Pressure', 'Temperature', 'Oxygen']]
kNN_classification(Xdata_sub, ylabels, 0.15, 1, axs[1,2], 'P, T, O')

# combination 9: pressure, salinity, oxygen
Xdata_sub = Xdata[['Pressure', 'Salinity', 'Oxygen']]
kNN_classification(Xdata_sub, ylabels, 0.15, 1, axs[1,3], 'P, S, O')

# combination 10: temperature, salinity, oxygen
Xdata_sub = Xdata[['Temperature', 'Salinity', 'Oxygen']]
kNN_classification(Xdata_sub, ylabels, 0.15, 1, axs[1,4], 'T, S, O')





