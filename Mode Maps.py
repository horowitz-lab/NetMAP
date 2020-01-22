# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 19:38:08 2020

@author: vhorowit
"""

import pandas as pd 
import os 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
from myheatmap import myheatmap

import matplotlib
print(matplotlib.__version__) # version 3 required for cmap='twilight_shifted'

sns.set_context('poster') # makes text larger

add_circle = True

############### Mac vs Windows ##############
#folder = '/Volumes/Aleman-Lab/Group/Projects/GrapheneCoupledResonators/Data/Sample2/2019/2019/08/09/'
folder = r'\\cas-fs1.uoregon.edu\Material-Science-Institute\Aleman-Lab\Group\Projects\GrapheneCoupledResonators\Data\Sample2\2019\2019\08\09'


file = 'PosGU01Array01.00r0.5p3Array01.00r0.5p3Device1.2Drum_15.16MHz_1104325' 
path = os.path.join(folder,file)
df = pd.read_csv(os.path.join(path, 'Demod1.csv'), skiprows = 2) 
#df.r = df.r/df.r.max() 
#print(df) 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.23,5))

#plt.suptitle(file + '\n\n');

plt.sca(ax1)
df['rlog'] = np.log10(df.r) 
df['X um']=df['Green X']
df['Y um']= df['Green Y']
amp1 = df.pivot_table(index = 'Y um', columns = 'X um', values = 'r').sort_index(axis = 0, ascending = False) 
myheatmap(amp1, 'r', cmap = 'viridis');
plt.xlabel(u'$x$ [μm]')
plt.ylabel(u'$y$ [μm]')
ax1.axis('equal')


plt.sca(ax2)
df['rlog'] = np.log10(df.r) 
df['X um']=df['Green X']
df['Y um']= df['Green Y']
phase1 = df.pivot_table(index = 'Y um', columns = 'X um', values = 'phase').sort_index(axis = 0, ascending = False) 
myheatmap(phase1, 'phase', cmap='PiYG');
ax2.axis('equal');
plt.xlabel(u'$x$ [μm]')
plt.ylabel(u'$y$ [μm]')

if add_circle:
    circle1 = plt.Circle((-6.8, 7.6), 1, color='w', alpha = .3)
    circle1b = plt.Circle((-7.0829,7.27412), 1, color='w', alpha = .3)
    circle2 = plt.Circle((-2.9,9.9), 1, color='w', alpha = .3)
    circle2b = plt.Circle((-2.61785,9.06889), 1, color='w', alpha = .3)
    circle2c = plt.Circle((-2.70398,9.617), 1, color='w', alpha = .3)    
    circle3 = plt.Circle((-0.92905,5.01833), 1, color='w', alpha = .3)
    circle5 = plt.Circle((-0.385325,4.73517), 1, color='w', alpha = .3)
#    circle6 = plt.Circle((-8.96522,11.2139), 1, color='w', alpha = .3)
    circle6 = plt.Circle((-8.57326,11.2718), 1, color='w', alpha = .3)
    circle7 = plt.Circle((-7.69574,14.5408), 1, color='w', alpha = .3)
    circle8 = plt.Circle((0.442485,8.64886), 1, color='w', alpha = .3)
    circle9 = plt.Circle((-5.73944,10.3789), 1, color='w', alpha = .3)
#    circle10 = plt.Circle((-4.23007,13.9537), 1, color='w', alpha = .3)
    circle10 = plt.Circle((-4.36885,13.8841), 1, color='w', alpha = .3)
    circle11 = plt.Circle((-8.04321,4.34145), 1, color='w', alpha = .3)
#    circle12 = plt.Circle((-5.10392,2.59376), 1, color='w', alpha = .3)
    circle12 = plt.Circle((-4.83885,2.82488), 1, color='w', alpha = .3)
    circle13 = plt.Circle((-2.08518,1.8788), 1, color='w', alpha = .3)



    ax1.add_artist(circle1)
     #  ax1.add_artist(circle1b)
    ax1.add_artist(circle2c)
  #  ax1.add_artist(circle2b)
    ax1.add_artist(circle3)
    #ax1.add_artist(circle5)
    ax1.add_artist(circle6)
    ax1.add_artist(circle7)
    ax1.add_artist(circle8)
    ax1.add_artist(circle9)
    ax1.add_artist(circle10)
    ax1.add_artist(circle11)
    ax1.add_artist(circle12)
    ax1.add_artist(circle13)

plt.tight_layout();


file = 'PosGU01Array01.00r0.5p3Array01.00r0.5p3Device1.2Drum1_15.5MHz105133' 
path = os.path.join(folder,file)
df = pd.read_csv(os.path.join(path, 'Demod1.csv'), skiprows = 2) 
#df.r = df.r/df.r.max() 
#print(df) 
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(13.23,5))

plt.sca(ax3)
df['rlog'] = np.log10(df.r) 
df['X um']=df['Green X']
df['Y um']= df['Green Y']
amp2 = df.pivot_table(index = 'Y um', columns = 'X um', values = 'r').sort_index(axis = 0, ascending = False) 
myheatmap(amp2, 'r', cmap = 'viridis');
plt.xlabel(u'$x$ [μm]')
plt.ylabel(u'$y$ [μm]')
ax3.axis('equal');

if add_circle:
    circle1 = plt.Circle((-6.8, 7.6), 1, color='w', alpha = .3)
    circle1b = plt.Circle((-7.0829,7.27412), 1, color='w', alpha = .3)
    circle2 = plt.Circle((-2.9,9.9), 1, color='w', alpha = .3)
    circle2b = plt.Circle((-2.61785,9.06889), 1, color='w', alpha = .3)
    circle2c = plt.Circle((-2.70398,9.617), 1, color='w', alpha = .3)    
    circle3 = plt.Circle((-0.92905,5.01833), 1, color='w', alpha = .3)
    circle5 = plt.Circle((-0.385325,4.73517), 1, color='w', alpha = .3)
#    circle6 = plt.Circle((-8.96522,11.2139), 1, color='w', alpha = .3)
    circle6 = plt.Circle((-8.57326,11.2718), 1, color='w', alpha = .3)
    circle7 = plt.Circle((-7.69574,14.5408), 1, color='w', alpha = .3)
    circle8 = plt.Circle((0.442485,8.64886), 1, color='w', alpha = .3)
    circle9 = plt.Circle((-5.73944,10.3789), 1, color='w', alpha = .3)
#    circle10 = plt.Circle((-4.23007,13.9537), 1, color='w', alpha = .3)
    circle10 = plt.Circle((-4.36885,13.8841), 1, color='w', alpha = .3)
    circle11 = plt.Circle((-8.04321,4.34145), 1, color='w', alpha = .3)
#    circle12 = plt.Circle((-5.10392,2.59376), 1, color='w', alpha = .3)
    circle12 = plt.Circle((-4.83885,2.82488), 1, color='w', alpha = .3)
    circle13 = plt.Circle((-2.08518,1.8788), 1, color='w', alpha = .3)


    
    ax3.add_artist(circle1)
     #  ax1.add_artist(circle1b)
    ax3.add_artist(circle2c)
  #  ax1.add_artist(circle2b)
    ax3.add_artist(circle3)
    #ax1.add_artist(circle5)
    ax3.add_artist(circle6)
    ax3.add_artist(circle7)
    ax3.add_artist(circle8)
    ax3.add_artist(circle9)
    ax3.add_artist(circle10)
    ax3.add_artist(circle11)
    ax3.add_artist(circle12)
    ax3.add_artist(circle13)

plt.tight_layout();

plt.sca(ax4)
df['rlog'] = np.log10(df.r) 
df['X um']=df['Green X']
df['Y um']= df['Green Y']
phase2 = df.pivot_table(index = 'Y um', columns = 'X um', values = 'phase').sort_index(axis = 0, ascending = False) 
myheatmap(phase2, 'phase', cmap='PiYG');
plt.xlabel(u'$x$ [μm]')
plt.ylabel(u'$y$ [μm]')
ax4.axis('equal');
plt.tight_layout();



plt.show()