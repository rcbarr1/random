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

#%% step 1c: turn data into inputs and labels arrays

# see how much data in each label category
# number of points with good OXYGEN data: BATS = 26153, ALOHA = 21930, CARIACO = 3384, CVOO = 524, K2 = 596
# number of points with only S, T, P: BATS = 26153, ALOHA = 86651, CARIACO = 4103, CVOO = 1166, K2 = 609

# LABEL KEY
# BATS      = 0 (bermuda atlantic time series)
# ALOHA     = 1 (hawaii ocean time series)
# CARIACO   = 2 (caricao basin time series)
# CVOO      = 3 (cape verde ocean observatory)
# K2        = 4 (time series K2)

# create labels for bats
bats['Label'] = 0

# convert TimeSeriesSite from string to integer label
spots['Label'] = 0
spots.loc[(spots['TimeSeriesSite'] == 'ALOHA'),'Label'] = 1
spots.loc[(spots['TimeSeriesSite'] == 'CARIACO'),'Label'] = 2
spots.loc[(spots['TimeSeriesSite'] == 'CVOO'),'Label'] = 3
spots.loc[(spots['TimeSeriesSite'] == 'K2'),'Label'] = 4

# set up data and labels arrays
Xdata_bats = bats[['day_of_year', 'Pressure', 'Temperature', 'Salinity', 'Oxygen']]
Xdata_spots = spots[['day_of_year', 'Pressure', 'Temperature', 'Salinity', 'Oxygen']]

Xdata = pd.concat([Xdata_bats, Xdata_spots], ignore_index=True)
ylabels = pd.concat([bats['Label'], spots['Label']], ignore_index=True)

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
    ax.scatter(x, y, label='BATS', alpha = 0.1)
    
    # HOT
    spots = spots.sort_values(by=['day_of_year'])
    x = spots.loc[(spots['TimeSeriesSite'] == 'ALOHA'),'day_of_year']
    y = spots.loc[(spots['TimeSeriesSite'] == 'ALOHA'), var_name]
    ax.scatter(x, y, label='HOT', alpha = 0.05)
    
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

#%% step 3: build classifier
# - test with varying predictors and classifiers




#%% step 4: build FCN (maybe?)
