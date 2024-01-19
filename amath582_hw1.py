#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:00:58 2024

@author: Reese Barrett

Code for AMATH 582 Homework 1 assignment.
"""

import numpy as np
#import plotly
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objs as go
pio.renderers.default='browser'

# load in submarine data
filepath = '/Users/Reese/Documents/Research Projects/random/amath582/'
subdata = np.load(filepath + 'subdata.npy')
  
#%% task 1: through averaging of the Fourier transform and visual inspection,
# determine the frequency signature (center frequency, in kx, ky, kz)
# determined by the submarine

# define spatial domain for x, y, and z dimensions
L = 10 # spatial domain length (cube has sides L = 2*10)
N_grid = 64 # grid is 64 x 64 x 64
xx = np.linspace(-L, L, N_grid+1) # create linspace along a dimension
x_grid = xx[0:N_grid] # spatial grid in x-direction
y_grid = xx[0:N_grid] # spatial grid in y-direction
z_grid = xx[0:N_grid] # spatial grid in z-direction

# define frequency domain for x, y, and z dimensions - CHECK SCALING ON THIS
Kx_grid = (2*np.pi / (2/L)) * np.linspace(-N_grid/2, N_grid/2 - 1, N_grid) # frequency grid in x-direction
Ky_grid = (2*np.pi / (2/L)) * np.linspace(-N_grid/2, N_grid/2 - 1, N_grid) # frequency grid in y-direction
Kz_grid = (2*np.pi / (2/L)) * np.linspace(-N_grid/2, N_grid/2 - 1, N_grid) # frequency grid in z-direction

#Kx_grid = np.linspace(-N_grid/2, N_grid/2 - 1, N_grid) # frequency grid in x-direction
#Ky_grid = np.linspace(-N_grid/2, N_grid/2 - 1, N_grid) # frequency grid in y-direction
#Kz_grid = np.linspace(-N_grid/2, N_grid/2 - 1, N_grid) # frequency grid in z-direction

# preallocate array to store Fourier results from each time step
all_fhat_ss = np.zeros_like(subdata)

# loop through time steps
for i in range (0, subdata.shape[1]):
    # make 3D array of acoustic signal at the time step
    time_step = np.reshape(subdata[:, i], (N_grid, N_grid, N_grid))
    
    # do 3D Fourier transform (plus FFT shift)
    fhat = np.fft.fftn(time_step, axes=(0,1,2)) # use FFTN to do Fast Fourier Transform on 3D array
    fhat_s = np.fft.fftshift(fhat) # do fftshfit to ensure peaks appear at the correct frequency
    fhat_ss = (1/N_grid)*fhat_s # scale amplitudes of the coefficients based on grid
    flat_fhat_ss = fhat_ss.flatten() # flatten 3D grid for storage 
    all_fhat_ss[:,i] = flat_fhat_ss # store flattened grid in array that will hold all time steps
    
# average across all 50 arrays
all_fhat_ss_abs = np.abs(all_fhat_ss)
avg_fhat_ss_abs = all_fhat_ss_abs.mean(axis=1)

# reshape back to cube
avg_fhat_ss_abs = np.reshape(avg_fhat_ss_abs, (N_grid, N_grid, N_grid))

# center frequency is the maximum of the fourier cube averaged across all time steps
center_freq_idx = np.where(avg_fhat_ss_abs == avg_fhat_ss_abs.max()) # find index of center frequency
center_freq = [Kx_grid[center_freq_idx[0][0]], Ky_grid[center_freq_idx[1][0]], Kz_grid[center_freq_idx[2][0]]] # convert to kx, ky, kz grid

# visual inspection: plot averaged fourier domain
xv, yv, zv = np.meshgrid(Kx_grid, Ky_grid, Kz_grid) # create meshgrid in 3D
flat_avg_fhat_ss_abs = avg_fhat_ss_abs.flatten() # flatten avg_fhat_ss_abs
flat_norm_avg_fhat_ss_abs = flat_avg_fhat_ss_abs / flat_avg_fhat_ss_abs.max() # normallize flat_avg_fhat_ss_abs
fig_data = go.Isosurface(x = xv.flatten(), y = yv.flatten(), z = zv.flatten(),
                         value = flat_norm_avg_fhat_ss_abs, isomin=0.7, isomax=0.7)
fig = go.Figure(data = fig_data)
fig.show()

#%% task 2: design and implement a filter to extract this frequency signature
# in order to denoise the data and determine a more robust path of the
# submarine
# plot the 3D path of the submarine

# create Gaussian filter centered around the center frequency
# Gaussian filter (Kutz 13.2.1): F(k) = exp(-T(k - k0)^2)
# expand to 3D: F([kx, ky, kz]) = exp[-T((kx - kx0)^2 + (ky - ky0)^2 + (kz - kz0)^2)]
# source: https://mathworld.wolfram.com/GaussianFunction.html
# T = bandwidth of the filter
# k0 = center frequency (kx0, ky0, kz0)
#T = 0.007 # tune this manually (this works for [-32, 31])
T = 0.000007 # tune this manually (this works for [-1005.31, 973.89])

# create Gaussian filter
# preallocate array for storage
Gaussian_filter = np.zeros_like(np.abs(fhat))

# extract center frequencies
kx0 = center_freq[0]
ky0 = center_freq[1]
kz0 = center_freq[2]

# for each entry in fhat, pull kx, ky, and kz and calculate F
for i in range(0, len(Kx_grid)): # loop through x-dimension
    for j in range(0, len(Ky_grid)): # loop through y-dimension
        for k in range(0, len(Kz_grid)): # loop through z-dimension
            kx = Kx_grid[i]
            ky = Ky_grid[j]
            kz = Kz_grid[k]
            
            Gaussian_filter[i, j, k] = np.exp(-T * ((kx - kx0)**2 + (ky - ky0)**2 + (kz - kz0)**2))

# apply Gaussian filter to noisy signal at each time step
# preallocate array to store IFFT results and sub location from each time step 
all_f_clean = np.zeros_like(np.abs(subdata)) # IFFT results
all_location_idx = np.zeros([subdata.shape[1], np.ndim(Gaussian_filter)]) # submarine location at each time step

# flatten Gaussian filter
flat_Gaussian_filter = Gaussian_filter.flatten()

for i in range (0, subdata.shape[1]):
    # apply flattened Gaussian filter to flattened fhat
    filtered_fhat = all_fhat_ss[:,i] * flat_Gaussian_filter
    
    # convert filtered_vals back to 3D
    filtered_fhat_3D = np.reshape(filtered_fhat, (N_grid, N_grid, N_grid))

    # apply IFFT Shift and IFFT
    f_clean = np.fft.ifftn(np.fft.ifftshift(filtered_fhat_3D), axes=(0,1,2))
    f_clean_real = np.real(f_clean) # keep only real numbers

    # find index of maximum value for each time step, store data
    location_idx = np.where(f_clean_real == f_clean_real.max()) # find index of sub location at that time step
    all_location_idx[i,0] = location_idx[0][0]
    all_location_idx[i,1] = location_idx[1][0]
    all_location_idx[i,2] = location_idx[2][0]

    # store IFFT data
    flat_f_clean_real = f_clean_real.flatten() # flatten 3D grid for storage
    all_f_clean[:,i] = flat_f_clean_real # store flattened grid in array that will hold all time steps
    
# convert location indicies to actual coordinates
all_location = np.zeros_like(all_location_idx) # preallocate array
for i in range(0,len(all_location)):
    all_location[i,0] = x_grid[int(all_location_idx[i,0])]
    all_location[i,1] = y_grid[int(all_location_idx[i,1])]
    all_location[i,2] = z_grid[int(all_location_idx[i,2])]

#%% plot actual coordinates in 3D

# set up figure
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

# set up location and time data
x_coords = all_location[:,0]
y_coords = all_location[:,1]
z_coords = all_location[:,2]
time = np.flip(np.linspace(0, 24, num = 49))

ax.plot3D(y_coords, x_coords, z_coords, color='black') # swapped x and y here to resemble submarine.gif
scpl = ax.scatter3D(y_coords, x_coords, z_coords, c=time, alpha=0.5) # swapped x and y here to resemble submarine.gif

c = fig.colorbar(scpl, ax=ax, pad=0.15)
c.set_label('Elapsed Time (hours)')

ax.axes.set_xlim3d(left=-10, right=10) 
ax.axes.set_ylim3d(bottom=-10, top=10) 
ax.axes.set_zlim3d(bottom=-10, top=10) 

ax.set_title("Submarine Trajectory (3D View)")
ax.set_xlabel('y') # swapped x and y here to resemble submarine.gif
ax.set_ylabel('x') # swapped x and y here to resemble submarine.gif
ax.set_zlabel('z')
ax.invert_yaxis() # invert to resemble submarine.gif

#%% task 3: determine and plot the x, y coordinates of the submarine during the
# 24 hour period

# set up figure
fig = plt.figure()
ax = fig.gca()

# set up location and time data
x_coords = all_location[:,0]
y_coords = all_location[:,1]
time = np.flip(np.linspace(0, 24, num = 49))

plt.plot(x_coords ,y_coords, color='black', linewidth=2)
scpl = plt.scatter(x_coords ,y_coords, c=time, alpha=0.5)

c = fig.colorbar(scpl, ax=ax)
c.set_label ('Elapsed Time (hours)')

plt.grid(visible=True)

ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_xlabel('x')
ax.set_ylabel('y',rotation=0)
ax.set_title("Submarine Trajectory (Top-Down View)")

#%% bonus: implement an alternative technique for noise filtering,
# substantially different from task 2, and visualize and compare the results
# for this, basic assumption is changing: I will no longer assume the
# submarine's frequency is constant in time

# implement wavelet transformation (slide window, larger window at low
# frequencies and smaller window at high frequencies)




















