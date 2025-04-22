#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures for outreach slides

Created on Thu Apr 17 13:19:43 2025

@author: Reese Barrett
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patheffects
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter
from matplotlib import colormaps as cmaps
import pandas as pd
import cmocean

# set global font
rcParams['font.family'] = 'Verdana'

#%% make better version of atmospheric CO2, seawater pCO2, pH figure

# load atmospheric CO2 data (Mauna Loa)
# https://gml.noaa.gov/ccgg/trends/data.html
# Dr. Xin Lan, NOAA/GML (gml.noaa.gov/ccgg/trends/) and Dr. Ralph Keeling, Scripps Institution of Oceanography (scrippsco2.ucsd.edu/).

mauna_loa = '/Users/Reese_1/Documents/Research Projects/random/outreach_data/co2_mm_mlo.csv'
mauna_loa_data = pd.read_csv(mauna_loa, delimiter=',', skiprows=40) 
    
# load ocean data (station ALOHA)
# https://hahana.soest.hawaii.edu/hot/hotco2/hotco2.html
# Dore, J.E., R. Lukas, D.W. Sadler, M.J. Church, and D.M. Karl. 2009. Physical and
# biogeochemical modulation of ocean acidification in the central North Pacific. Proc Natl Acad
# Sci USA 106:12235-12240.

aloha = '/Users/Reese_1/Documents/Research Projects/random/outreach_data/HOT_surface_CO2.txt'
aloha_data = pd.read_csv(aloha, delimiter='\t', skiprows=8, na_values=-999) 

# convert aloha data to decimal year to match mauna loa data
aloha_data['date'] = pd.to_datetime(aloha_data['date'], format='%d-%b-%y')
def to_decimal_year(dates):
    year = dates.dt.year
    start_of_year = pd.to_datetime(year.astype(str) + '-01-01')
    end_of_year = pd.to_datetime((year + 1).astype(str) + '-01-01')
    return year + (dates - start_of_year) / (end_of_year - start_of_year)

aloha_data['date'] = to_decimal_year(aloha_data['date'])

# plot data!!
fig = plt.figure(figsize=(6,4), dpi=200)
ax1 = fig.gca()
ax2 = ax1.twinx()

# plot mauna loa CO2
ax1.plot(mauna_loa_data['decimal date'], mauna_loa_data['average'], linestyle='-', linewidth=0.5, c='#DA9497')
ax1.scatter(mauna_loa_data['decimal date'], mauna_loa_data['average'], s=3, c='#DA9497')

# plot aloha pCO2
ax1.plot(aloha_data['date'], aloha_data['pCO2calc_insitu'], linestyle='-', linewidth=0.5, c='#A9769C')
ax1.scatter(aloha_data['date'], aloha_data['pCO2calc_insitu'], s=3, c='#A9769C')

# plot aloha pH
ax2.plot(aloha_data['date'], aloha_data['pHmeas_insitu'], linestyle='-', linewidth=0.5, c='#255F85')
ax2.scatter(aloha_data['date'], aloha_data['pHmeas_insitu'], s=3, c='#255F85')

# set axis limits
ax1.set_xlim([mauna_loa_data['decimal date'][0]-2, mauna_loa_data['decimal date'][804]+2])
ax1.set_ylim([250, 440])
ax2.set_xlim([mauna_loa_data['decimal date'][0]-2, mauna_loa_data['decimal date'][804]+2])
ax2.set_ylim([8.00, 8.50])

# set axis labels & title
ax1.set_ylabel('CO$_{2}$ Concentration (µatm)') # assuming ideal gas to keep same units for CO2
ax2.set_ylabel('pH')
ax1.set_xlabel('Year')
ax1.set_title('CO$_{2}$ Time Series in North Pacific')

# make legend
legend_elements = [
    Line2D([0], [0], color='#DA9497', marker='o', label='Mauna Loa Atmospheric CO$_{2}$'),
    Line2D([0], [0], color='#A9769C', marker='o', label='Station ALOHA Seawater $p$CO$_{2}$'),
    Line2D([0], [0], color='#255F85', marker='o', label='Station ALOHA Seawater pH'),
]
ax1.legend(handles=legend_elements, loc='upper left')

plt.show()

#%% make pH only figure
fig = plt.figure(figsize=(5.5,1.6), dpi=200)
ax1 = fig.gca()

# plot aloha pH
ax1.plot(aloha_data['date'], aloha_data['pHmeas_insitu'], linestyle='-', linewidth=0.5, c='#255F85')
ax1.scatter(aloha_data['date'], aloha_data['pHmeas_insitu'], s=3, c='#255F85')
ax1.set_xlim([aloha_data['date'][0]-2, aloha_data['date'][347]+2])
ax1.set_ylim([7.95, 8.3])

# set axis labels
ax1.set_ylabel('pH')
ax1.set_xlabel('Year')

# make legend
# make legend
legend_elements = [
    Line2D([0], [0], color='#255F85', marker='o', label='Station ALOHA Seawater pH'),
]
ax1.legend(handles=legend_elements, loc='upper left')

# plot horizontal dashed lines 0.1 pH unit apart
ax1.axhline(y=8.025, c='#595959', linestyle='--')
ax1.axhline(y=8.125, c='#595959', linestyle='--')

plt.show()

#%% make better version of pH scale figure

# choose colormap
cmap = cmaps['RdBu']

# parameters
num_colors = 15
box_width = 0.9 # slightly less than 1 to leave space
spacing = 0.1 # gap between boxes
total_width = num_colors * (box_width + spacing)

# generate evenly spaced colors
colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

# create figure
fig, ax = plt.subplots(figsize=(12, 2), dpi=200)

# plot colored boxes and add numbers
for i, color in enumerate(colors):
    x = i * (box_width + spacing)
    rect = plt.Rectangle((x, 0), box_width, 1, color=color)
    ax.add_patch(rect)
    
    # add number inside the box
    num_text = ax.text(x + box_width / 2, 0.5, str(i), color='white', ha='center', va='center', fontsize=30)

    # add a dark gray outline around the number
    num_text.set_path_effects([
        patheffects.withStroke(linewidth=2, foreground='#595959')
    ])

# formatting
ax.set_xlim(0, total_width)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('auto')
plt.tight_layout()

# remove black box
for spine in ax.spines.values():
    spine.set_visible(False)
    
plt.show()

#%% figure showing pH on x-axis and H+ concentration on y-axis

fig = plt.figure(figsize=(4.5, 3.5), dpi=200)
ax1 = fig.gca()

x = np.linspace(7.5, 9.5, num=999) # pH
y = 10**(-1*x) # [H+] = 10^(-pH)

ax1.plot(x,y, c='#255F85')

# add dashed vertical lines
#ax1.plot([8.1, 8.1], [y[-1], y[300]], c='#595959' , linestyle='--')

ax1.plot([8.0, 8.0], [y[-1], y[250]], c='#595959' , linestyle='--')
ax1.plot([8.1, 8.1], [y[-1], y[300]], c='#b5b0b0' , linestyle='--')

# add dashed horizontal lines
#ax1.plot([7, 8.1], [y[300], y[300]], c='#595959' , linestyle='--')

ax1.plot([7, 8.0], [y[250], y[250]], c='#595959' , linestyle='--')
ax1.plot([7, 8.1], [y[300], y[300]], c='#b5b0b0' , linestyle='--')

# formatting
ax1.set_xlim([7.5, 9.5])
ax1.set_ylim([y[-1], 2e-8])
ax1.set_xticks([7.5, 8.0, 8.5, 9])
ax1.set_yticks([0.5e-8, 1e-8, 1.5e-8, 2e-8])
#ax1.set_yticks([0.5e-8, 1e-8, y[300], 1.5e-8, 2e-8])

# set scientific notation directly on y-axis ticks
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))  # force scientific notation for all values
ax1.yaxis.set_major_formatter(formatter)

# labels
ax1.set_xlabel('pH')
ax1.set_ylabel('Concentration of\nHydrogen Ions (mol/L)')

plt.show()





