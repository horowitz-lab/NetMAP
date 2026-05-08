#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:35:48 2024

@author: lydiabullock
"""
from comparing_curvefit_types import complex_noise, get_parameters_NetMAP, find_avg_e, automate_guess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Trimer_simulator import realamp1, realamp2, realamp3, imamp1, imamp2, imamp3, curve1, theta1, curve2, theta2, curve3, theta3
import sys
import os
myheatmap = os.path.abspath('..')
sys.path.append(myheatmap)
from myheatmap import myheatmap
import matplotlib.colors as mcolors


def plot_data(frequencies, params_guess, params_correct, e, force_all):
    #Original Data
    X1 = realamp1(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    Y1 = imamp1(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) 
    
    X2 = realamp2(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    Y2 = imamp2(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) 
    
    X3 = realamp3(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    Y3 = imamp3(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
        
    Amp1 = curve1(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    Phase1 = theta1(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) \
        + 2 * np.pi
    Amp2 = curve2(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    Phase2 = theta2(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) \
        + 2 * np.pi
    Amp3 = curve3(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    Phase3 = theta3(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) \
        + 2 * np.pi

    ## Begin graphing - Re vs Im
    fig = plt.figure(figsize=(10,6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1,1,1], hspace=0.25, wspace=0.05)
    
    ax1 = fig.add_subplot(gs[0, 0], aspect='equal')
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1, aspect='equal')
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1, aspect='equal')
    
    #Original Data
    ax1.plot(X1,Y1,'ro', alpha=0.5, markersize=5.5, label = 'Data')
    ax2.plot(X2,Y2,'bo', alpha=0.5, markersize=5.5, label = 'Data')
    ax3.plot(X3,Y3,'go', alpha=0.5, markersize=5.5, label = 'Data')
    
    fig.suptitle('Trimer Resonator: Real and Imaginary', fontsize=16)
    ax1.set_title('Resonator 1', fontsize=14)
    ax2.set_title('Resonator 2', fontsize=14)
    ax3.set_title('Resonator 3', fontsize=14)
    ax1.set_ylabel('Im(Z) (m)')
    ax1.set_xlabel('Re(Z) (m)')
    ax2.set_xlabel('Re(Z) (m)')
    ax3.set_xlabel('Re(Z) (m)')
    ax1.label_outer()
    ax2.label_outer()
    ax3.label_outer()
    plt.show()
    
    ## Begin graphing - Amp and Phase
    fig = plt.figure(figsize=(16,8))
    gs = fig.add_gridspec(2, 3, hspace=0.1, wspace=0.1)
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = gs.subplots(sharex=True, sharey='row')
    
    #original data
    ax1.plot(frequencies, Amp1,'ro', alpha=0.5, markersize=5.5, label = 'Data')
    ax2.plot(frequencies, Amp2,'bo', alpha=0.5, markersize=5.5, label = 'Data')
    ax3.plot(frequencies, Amp3,'go', alpha=0.5, markersize=5.5, label = 'Data')
    ax4.plot(frequencies, Phase1,'ro', alpha=0.5, markersize=5.5, label = 'Data')
    ax5.plot(frequencies, Phase2,'bo', alpha=0.5, markersize=5.5, label = 'Data')
    ax6.plot(frequencies, Phase3,'go', alpha=0.5, markersize=5.5, label = 'Data')
    
    #Graph parts
    fig.suptitle('Trimer Resonator: Amplitude and Phase', fontsize=16)
    ax1.set_title('Resonator 1', fontsize=14)
    ax2.set_title('Resonator 2', fontsize=14)
    ax3.set_title('Resonator 3', fontsize=14)
    ax1.set_ylabel('Amplitude')
    ax4.set_ylabel('Phase')
    
    for ax in fig.get_axes():
        ax.set(xlabel='Frequency')
        ax.label_outer()

    plt.show()

#Code that loops through frequency points of different spacing 

def sweep_freq_pair(frequencies, params_guess, params_correct, e, force_all):
    
    #Graph Real vs Imaginary for the trimer
    plot_data(frequencies, params_guess, params_correct, e, force_all)
    
    # Loop over possible combinations of frequency indices, i1 and i2
    for i1 in range(len(frequencies)):
        freq1 = frequencies[i1]
        

        for i2 in range(len(frequencies)):
            freq2 = frequencies[i2]
            freqs = [freq1, freq2]
            e_2freqs = complex_noise(2,2)
            
            NetMAP_info = get_parameters_NetMAP(freqs, params_guess, params_correct, e_2freqs, force_all)
            
            #Find <e> (average across parameters) for the trial and add to dictionary
            avg_e1 = find_avg_e(NetMAP_info)
            NetMAP_info['<e>'] = avg_e1
            NetMAP_info['freq1'] = freq1
            NetMAP_info['freq2'] = freq2
            
            # Convert lists to scalars when they contain only one item
            for key in NetMAP_info:
                if isinstance(NetMAP_info[key], list) and len(NetMAP_info[key]) == 1:
                    NetMAP_info[key] = NetMAP_info[key][0]
            
            NetMAP_df = pd.DataFrame([NetMAP_info])
            
            try: # repeated experiments results
                resultsdf = pd.concat([resultsdf, NetMAP_df], ignore_index=True)
            except:
                resultsdf = NetMAP_df
    
    return resultsdf 


''' Begin work here. '''

##Create the System
#Randomly chosen one that "looks easy"
# params_correct = [3, 3, 3, 3, 0.5, 0.5, 0.1, 1, 2, 5, 5]
# params_guess = automate_guess(params_correct, 20)

#Worst system - System 8 from ‘15 systems - 10 Freqs NetMAP & Better Parameters’
params_correct = [7.731, 1.693, 2.051, 8.091, 0.427, 0.363, 0.349, 1, 7.07, 7.195, 4.814]
params_guess = [7.2806, 1.8748, 1.8077, 8.7478, 0.3767, 0.2974, 0.3744, 1, 7.4933, 6.7781, 4.2136]

force_all = False
e = complex_noise(200,2)
frequencies = np.linspace(0.001, 4, 200)

#Test each pair of frequencies
result = sweep_freq_pair(frequencies, params_guess, params_correct, e, force_all)
result.to_excel('Frequency_Study.xlsx', index=False)

#Recall the data if I need to
# result = pd.read_excel('/Users/Student/Desktop/Summer Research 2024/Multiple Curve Fit - Which Type/Frequency Study/Frequency_Study_200.xlsx')

#Pivot the DataFrame for the heatmap
heatmap_data = result.pivot_table(index='freq2', columns='freq1', values='<e>')

#Create heatmap
#For log scale!
colors = [(1, 0.439, 0), 'yellow','green', 'blue', (0.533, 0.353, 0.537)]
n_bins = 100  # Number of bins for interpolation

cmap_name = 'custom_cmap'
custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

norm = mcolors.LogNorm(vmin=heatmap_data.min().min(), vmax=heatmap_data.max().max())
ax = myheatmap(heatmap_data, cmap=custom_cmap, norm=norm, colorbarlabel='Average Error (%)')

#For regular
# ax = myheatmap(heatmap_data, cmap=custom_cmap, vmax=10, colorbarlabel='Average Error (%)')

ax.set_title('NetMAP Recovery of Trimer Parameters')
ax.set_xlabel('Frequency 1 (rad/s)')
ax.set_ylabel('Frequency 2 (rad/s)')



