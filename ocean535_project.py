#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:20:28 2024

Calculates annual means of sea surface temperature for each of the regions
defined in the dictionary below from 2003 to 2020.

@author: Reese Barrett
"""

import xarray as xr
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# open up dataset
# source: https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/
filepath = '/Users/Reese/Documents/Research Projects/random/ocean535/'
data = xr.open_dataset(filepath + 'sst.oisst.mon.mean.1982.nc')

# define nine regions - pulled out a map and did my best to make boxes lol
# eurasian arctic: longitude = 0 to 92.06, latitude = 80.22 to 84.86
# AND for eurasian arctic: longitude = 346.60 to 359, latitude = 80.22 to 84.86
# amerasian arctic: longitude = 110.22 to 258.40, latitude = 17.02 to 84.99
# sea of okhotsk: longitude = 134.48 to 160.32, latitude = 45.78 to 61.97
# bering sea: longitude = 164.76 to 201.33, latitude = 52.16 to 66.10
# barents sea: longitude = 22.63 to 97.16, latitude = 67.79 to 81.97
# greenland sea: longitude = 0 to 16.84, latitude = 69.92 to 82.58
# AND for greenland sea: longitude = 339.49 to 359, latitude = 69.92 to 82.58
# hudson bay: longitude = 265.17 to 288.02, latitude = 53.80 to 70.50
# baffin bay/labrador sea: longitude = 283.80 to 316.78, latitude = 53.36 to 76.85
# north atlantic: longitude = 307.38 to 346.76, latitude = 47.07 to 68.92

# make dictionary of closest latitudes and longitudes to loop through xarray with
regions = {
    'eurasian' : np.array([[0, 92, 80.5, 84.5], [347, 359, 80.5, 84.5]]),
    'amerasian' : np.array([[110, 258, 17.5, 84.5]]),
    'okhotsk' : np.array([[134, 160, 45.5, 61.5]]),
    'bering' : np.array([[165, 201, 52.5, 66.5]]),
    'barents' : np.array([[23, 97, 67.5, 81.5]]),
    'greenland' : np.array([[0, 17, 69.5, 82.5],[339, 359, 69.5, 82.5]]),
    'hudson' : np.array([[265, 288, 53.5, 70.5]]),
    'baffin' : np.array([[284, 318, 53.5, 76.5]]),
    'atlantic' : np.array([[307, 347, 47.5, 68.5]])
    }

# define time slices for each year from 2003 to 2020
year_starts = np.array(['2003-01-01', '2004-01-01', '2005-01-01', '2006-01-01',
                        '2007-01-01', '2008-01-01', '2009-01-01', '2010-01-01',
                        '2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01',
                        '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01',
                        '2019-01-01', '2020-01-01'],dtype='datetime64')
year_ends = np.array(['2003-12-01', '2004-12-01', '2005-12-01', '2006-12-01',
                      '2007-12-01', '2008-12-01', '2009-12-01', '2010-12-01',
                      '2011-12-01', '2012-12-01', '2013-12-01', '2014-12-01',
                      '2015-12-01', '2016-12-01', '2017-12-01', '2018-12-01',
                      '2019-12-01', '2020-12-01'],dtype='datetime64')

# preallocate array to store annual means for each region
annual_means = np.zeros((len(regions),len(year_starts)))

# loop through each region
i = 0
for key in regions:
    
    # loop through each year to get an annual average
    for j in range(0,len(year_starts)):
        if regions[key].shape[0] == 1: # if the region is only in one box
            slice1 = regions[key][0]
            temp_slice = data.sel(time=slice(year_starts[j],year_ends[j]), lat=slice(slice1[2], slice1[3]),
                                  lon=slice(slice1[0], slice1[1])).sst.values
        
        elif regions[key].shape[0] == 2: # if I had to break the region into two boxes because of the longitude
            
            slice1 = regions[key][0]
            temp_slice1 = data.sel(time=slice(year_starts[j],year_ends[j]), lat=slice(slice1[2], slice1[3]),
                                  lon=slice(slice1[0], slice1[1])).sst.values
        
            slice2 = regions[key][1]
            temp_slice2 = data.sel(time=slice(year_starts[j],year_ends[j]), lat=slice(slice2[2], slice2[3]),
                                  lon=slice(slice2[0], slice2[1])).sst.values
            
            temp_slice = np.append(temp_slice1.flatten(), temp_slice2.flatten())
        else: # should never hit this
            break
            
        # do annual mean over all of the data
        annual_means[i,j] = np.nanmean(temp_slice)
    i += 1
     
# plot trends in sst over time for each region
fig = plt.figure(figsize=(7,5), dpi=200)
ax = fig.gca()
ax.plot(year_starts, annual_means[0,:],label='Eurasian Arctic')
ax.plot(year_starts, annual_means[1,:],label='Amerasian Arctic')
ax.plot(year_starts, annual_means[2,:],label='Sea of Okhotsk')
ax.plot(year_starts, annual_means[3,:],label='Bering Sea')
ax.plot(year_starts, annual_means[4,:],label='Barents Sea')
ax.plot(year_starts, annual_means[5,:],label='Greenland Sea')
ax.plot(year_starts, annual_means[6,:],label='Hudson Bay')
ax.plot(year_starts, annual_means[7,:],label='Baffin Bay/Labrador Sea')
ax.plot(year_starts, annual_means[8,:],label='North Atlantic')
ax.set_ylabel('Temperature (ºC)')
ax.set_title('Sea Surface Temperature in Arctic Regions (2003 - 2020)')
plt.legend(bbox_to_anchor=(0.5, -.30), loc='lower center', ncols=3)

#%% next steps: change from plot to scatter, do linear regressions for each and find slopes
# figure out some statistical test to compare trends

# calculate slopes and intercepts for each region
# preallocate array to store annual means for each region
regress_data = np.zeros((len(regions)+1,3))

# create array of integer years
years = np.array([2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
         2014, 2015, 2016, 2017, 2018, 2019, 2020])

# loop through each region
i = 0
for key in regions:
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(years, annual_means[i,:], alternative='two-sided')
    regress_data[i, 0] = slope
    regress_data[i, 1] = intercept
    regress_data[i, 2] = pvalue
    i += 1

# do for average
slope, intercept, rvalue, pvalue, stderr = stats.linregress(years, np.mean(annual_means,axis=0), alternative='two-sided')
regress_data[9, 0] = slope
regress_data[9, 1] = intercept
regress_data[9, 2] = pvalue


#%%
# plot all subplots
fig, axs = plt.subplots(5,2, figsize=(10,7), layout='constrained', sharex=True, sharey=True, dpi=200)
axs[0,0].scatter(years, annual_means[0,:],label='Eurasian Arctic')
axs[0,0].set_title('Eurasian Arctic', loc='left')
axs[0,0].plot(years, regress_data[0,1] + regress_data[0,0] * years, color="firebrick", lw=1, ls='--')
axs[0,0].set_ylim([-5, 25])
axs[0,0].text(0.42, 0.8, '$y={:.3f}x+{:.2f}$, p-value={:.3f}'.format(regress_data[0,0],regress_data[0,1],regress_data[0,2]), transform=axs[0,0].transAxes)

axs[0,1].scatter(years, annual_means[1,:],label='Amerasian Arctic')
axs[0,1].set_title('Amerasian Arctic', loc='left')
axs[0,1].plot(years, regress_data[1,1] + regress_data[1,0] * years, color="firebrick", lw=1, ls='--')
axs[0,1].text(0.42, 0.8, '$y={:.3f}x+{:.2f}$, p-value={:.3f}'.format(regress_data[1,0],regress_data[1,1],regress_data[1,2]), transform=axs[0,1].transAxes)

axs[1,0].scatter(years, annual_means[2,:],label='Sea of Okhotsk')
axs[1,0].set_title('Sea of Okhotsk', loc='left')
axs[1,0].plot(years, regress_data[2,1] + regress_data[2,0] * years, color="firebrick", lw=1, ls='--')
axs[1,0].text(0.42, 0.8, '$y={:.3f}x+{:.2f}$, p-value={:.3f}'.format(regress_data[2,0],regress_data[2,1],regress_data[2,2]), transform=axs[1,0].transAxes)

axs[1,1].scatter(years, annual_means[3,:],label='Bering Sea')
axs[1,1].set_title('Bering Sea', loc='left')
axs[1,1].plot(years, regress_data[3,1] + regress_data[3,0] * years, color="firebrick", lw=1, ls='--')
axs[1,1].text(0.42, 0.8, '$y={:.3f}x+{:.2f}$, p-value={:.3f}'.format(regress_data[3,0],regress_data[3,1],regress_data[3,2]), transform=axs[1,1].transAxes)

axs[2,0].scatter(years, annual_means[4,:],label='Barents Sea')
axs[2,0].set_title('Barents Sea', loc='left')
axs[2,0].plot(years, regress_data[4,1] + regress_data[4,0] * years, color="firebrick", lw=1, ls='--')
axs[2,0].text(0.42, 0.8, '$y={:.3f}x+{:.2f}$, p-value={:.3f}'.format(regress_data[4,0],regress_data[4,1],regress_data[4,2]), transform=axs[2,0].transAxes)

axs[2,1].scatter(years, annual_means[5,:],label='Greenland Sea')
axs[2,1].set_title('Greenland Sea', loc='left')
axs[2,1].plot(years, regress_data[5,1] + regress_data[5,0] * years, color="firebrick", lw=1, ls='--')
axs[2,1].text(0.42, 0.8, '$y={:.3f}x+{:.2f}$, p-value={:.3f}'.format(regress_data[5,0],regress_data[5,1],regress_data[5,2]), transform=axs[2,1].transAxes)

axs[3,0].scatter(years, annual_means[6,:],label='Hudson Bay')
axs[3,0].set_title('Hudson Bay', loc='left')
axs[3,0].plot(years, regress_data[6,1] + regress_data[6,0] * years, color="firebrick", lw=1, ls='--')
axs[3,0].text(0.42, 0.8, '$y={:.3f}x+{:.2f}$, p-value={:.3f}'.format(regress_data[6,0],regress_data[6,1],regress_data[6,2]), transform=axs[3,0].transAxes)

axs[3,1].scatter(years, annual_means[7,:],label='Baffin Bay/Labrador Sea')
axs[3,1].set_title('Baffin Bay/Labrador Sea', loc='left')
axs[3,1].plot(years, regress_data[7,1] + regress_data[7,0] * years, color="firebrick", lw=1, ls='--')
axs[3,1].text(0.42, 0.8, '$y={:.3f}x+{:.2f}$, p-value={:.3f}'.format(regress_data[7,0],regress_data[7,1],regress_data[7,2]), transform=axs[3,1].transAxes)

axs[4,0].scatter(years, annual_means[8,:],label='North Atlantic')
axs[4,0].set_title('North Atlantic', loc='left')
axs[4,0].plot(years, regress_data[8,1] + regress_data[8,0] * years, color="firebrick", lw=1, ls='--')
axs[4,0].text(0.42, 0.8, '$y={:.3f}x+{:.2f}$, p-value={:.3f}'.format(regress_data[8,0],regress_data[8,1],regress_data[8,2]), transform=axs[4,0].transAxes)

axs[4,1].scatter(years, np.mean(annual_means,axis=0),label='Average of Nine Regions')
axs[4,1].set_title('Average of Nine Regions', loc='left')
axs[4,1].plot(years, regress_data[9,1] + regress_data[9,0] * years, color="firebrick", lw=1, ls='--')
axs[4,1].text(0.42, 0.8, '$y={:.3f}x+{:.2f}$, p-value={:.3f}'.format(regress_data[9,0],regress_data[9,1],regress_data[9,2]), transform=axs[4,1].transAxes)

plt.xticks(years[0::2])
fig.supxlabel('Year')
fig.supylabel('Temperature (ºC)')
