# Script for visualizing the value of the objective function for the variant identification objective

import fastaparser
import os
import pandas as pd
import numpy as np
from itertools import combinations 
import seaborn as sns
import matplotlib.ticker as ticker
import sys
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns
import random

plt.rcParams['font.size'] = 20
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams["legend.frameon"] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.top'] = False
# ax.spines['right'].set_color('red')
# ax.spines['left'].set_color('red')
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['font.family'] = 'Helvetica'

cmap = plt.cm.get_cmap('viridis')
gcmap = plt.cm.get_cmap('gray')
base_col = "#bbbbbb"
adapt_col =  "#555555" 
evolutionary_col = "#55ad70"
wgan_col =  (.35, .09, .35) 


sns.set_theme(font="Helvetica", style='ticks')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

from matplotlib import font_manager
font_manager.fontManager.addfont('/home/ubuntu/Helvetica.ttf')


# Function to compute the value of the fitness given the minimum activity on on-target set and the maximum activity on the off-target set
def act_value(min_t1, max_t2):
    return (1/(1 + a*np.exp(k*(min_t1 - o)))) + r*-(1/(1 + a*np.exp(k*(max_t2 - o))))


#evolutionary parameters
a = 5.897292
k = -2.857755
o = -2.510856
r = 1.736507


zz = np.arange(-4, -.4, .1)
ranged = pd.DataFrame(columns = zz, index = zz)
for i, min_t1 in enumerate(zz):
    for j, max_t2 in enumerate(zz):
        ranged.iloc[i, j] = act_value(min_t1, max_t2)

fig, ax = plt.subplots(nrows =1, ncols = 1, figsize = (6, 6))
ax = sns.heatmap(ranged.astype(float), square = True, cbar_kws = {'shrink' : 0.7})
ax.set(ylabel="Minimum Activity of Guide on $T_1$", xlabel="Maximum Activity of Guide on $T_2$")
# format text labels
fmt = '{:0.2f}'
xticklabels = []
for item in ax.get_xticklabels():
    item.set_text(fmt.format(float(item.get_text())))
    xticklabels += [item]
yticklabels = []
for item in ax.get_yticklabels():
    item.set_text(fmt.format(float(item.get_text())))
    yticklabels += [item]

ax.set_xticklabels(xticklabels)
ax.set_yticklabels(yticklabels)
ax.collections[0].colorbar.set_label("Objective Function Value, $f_D(g|T_1, T_2)$")
plt.locator_params(axis='both', nbins=10)
fig.savefig('./computational/disc_obj_fn_evolutionary.pdf', dpi = 500)



#wgan parameters 
a = 3.769183
k = -3.833902
o = -2.134395
r = 2.973

zz = np.arange(-4, -.4, .1)
ranged = pd.DataFrame(columns = zz, index = zz)
for i, min_t1 in enumerate(zz):
    for j, max_t2 in enumerate(zz):
        ranged.iloc[i, j] = act_value(min_t1, max_t2)
         
        
fig, ax = plt.subplots(nrows =1, ncols = 1, figsize = (6, 6))
ax = sns.heatmap(ranged.astype(float), square = True, cbar_kws = {'shrink' : 0.7})
ax.set(ylabel="Minimum Activity of Guide on $T_1$", xlabel="Maximum Activity of Guide on $T_2$")
# format text labels
fmt = '{:0.2f}'
xticklabels = []
for item in ax.get_xticklabels():
    item.set_text(fmt.format(float(item.get_text())))
    xticklabels += [item]
yticklabels = []
for item in ax.get_yticklabels():
    item.set_text(fmt.format(float(item.get_text())))
    yticklabels += [item]

ax.set_xticklabels(xticklabels)
ax.set_yticklabels(yticklabels)
ax.collections[0].colorbar.set_label("Objective Function Value, $f_D(g|T_1, T_2)$")
plt.locator_params(axis='both', nbins=10)
fig.savefig('./computational/disc_obj_fn_wgan.pdf', dpi = 500)


