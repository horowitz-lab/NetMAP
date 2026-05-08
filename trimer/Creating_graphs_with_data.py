#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:09:41 2024

@author: Lydia Bullock
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Trimer_simulator import re1, re2, re3, im1, im2, im3, c1, t1, c2, t2, c3, t3
import os
import math

#Saves graphs
def save_figure(figure, folder_name, file_name):
    # Create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Save the figure to the folder
    file_path = os.path.join(folder_name, file_name)
    figure.savefig(file_path, bbox_inches = 'tight')
    plt.close(figure)

''' Redoing the histogram for 2269 systems with 1 trial '''

# #Recall the data from first sheet
# file_path1 = '/Users/Student/Desktop/Summer Research 2024/Curve Fit vs NetMAP/More Systems 1 Trial - <e> Histograms/All_Systems_1_Trial_1.xlsx'   
# array_amp_phase1 = pd.read_excel(file_path1, sheet_name = 'Polar').to_numpy()
# array_X_Y1 = pd.read_excel(file_path1, sheet_name = 'Cartesian').to_numpy()
# array_NetMAP1 = pd.read_excel(file_path1, sheet_name = 'NetMAP').to_numpy()

# #Recall the data from second sheet
# file_path2 = '/Users/Student/Desktop/Summer Research 2024/Curve Fit vs NetMAP/More Systems 1 Trial - <e> Histograms/All_Systems_1_Trial_2.xlsx'   
# array_amp_phase2 = pd.read_excel(file_path2, sheet_name = 'Polar').to_numpy()
# array_X_Y2 = pd.read_excel(file_path2, sheet_name = 'Cartesian').to_numpy()
# array_NetMAP2 = pd.read_excel(file_path2, sheet_name = 'NetMAP').to_numpy()

# #Pull out <e>_bar for each type from first sheet
# amp_phase_error1 = array_amp_phase1[:,50]
# X_Y_error1 = array_X_Y1[:, 50]
# NetMAP_error1 = array_NetMAP1[:,44]

# #Pull out <e>_bar for each type from first sheet
# amp_phase_error2 = array_amp_phase2[:,50]
# X_Y_error2 = array_X_Y2[:, 50]
# NetMAP_error2 = array_NetMAP2[:,44]

# #Concatenate
# all_polar_error = np.concatenate((amp_phase_error1, amp_phase_error2))
# all_NetMAP_error = np.concatenate((NetMAP_error1, NetMAP_error2))
# almost_all_cartesian_error = np.concatenate((X_Y_error1, X_Y_error2))
# all_cartesian_error = almost_all_cartesian_error[almost_all_cartesian_error != np.max(almost_all_cartesian_error)]


# #Graph histogram of <e>_bar for both curve fits

# # Compute max of data and set the bin limits so all data is included on graph
# data_max = np.max(np.concatenate((all_cartesian_error, all_polar_error, all_NetMAP_error)))
# if data_max > 39: 
#     linearbins = np.linspace(0, data_max + 2,50)
# else:
#     linearbins = np.linspace(0, 40, 50)

# #Graph linear!
# fig = plt.figure(figsize=(5, 4))
# plt.xlabel(r'$\overline{\langle e \rangle}$ (%)', fontsize = 16)
# plt.ylabel('Counts', fontsize = 16)
# plt.yticks(fontsize=14)
# plt.xticks(fontsize=14)
# plt.hist(all_cartesian_error , bins = linearbins, alpha=0.5, color='green', label='Cartesian', edgecolor='green', histtype= 'step')
# plt.hist(all_polar_error  , bins = linearbins, alpha=0.5, color='blue', label='Polar', edgecolor='blue', histtype= 'step')
# plt.hist(all_NetMAP_error , bins = linearbins, alpha=0.5, color='red', label='NetMAP', edgecolor='red', histtype= 'step')
# plt.legend(loc='best', fontsize = 13)

# plt.show()
# save_figure(fig, 'Final', '<e> Bar Lin Hist Total without Largest Value.pdf' )

# # Set the bin limits so all data is included on graph
# if data_max > 100: 
#     logbins = np.logspace(-2, math.log10(data_max)+0.1, 50)
# else:
#     logbins = np.logspace(-2, 1.8, 50)

# #Graph log!
# fig = plt.figure(figsize=(5, 4))
# plt.xlabel(r'$\overline{\langle e \rangle}$ (%)', fontsize = 16)
# plt.ylabel('Counts', fontsize = 16)
# plt.xscale('log')
# plt.yticks(fontsize=14)
# plt.xticks(fontsize=14)
# plt.hist(all_cartesian_error ,  bins = logbins, alpha=0.5, color='green', label='Cartesian', edgecolor='green', histtype= 'step', lw = 2)
# plt.hist(all_polar_error , bins = logbins, alpha=0.5, color='blue', label='Polar', edgecolor='blue', histtype= 'step', lw = 2)
# plt.hist(all_NetMAP_error , bins = logbins, alpha=0.5, color='red', label='NetMAP', edgecolor='red', histtype= 'step', lw = 2)
# plt.legend(loc='best', fontsize = 13)

# plt.show()
# save_figure(fig, 'Final', '<e> Bar Log Hist Total without Largest Value.pdf' )

''' Redoing the histogram for 15 Systems - 10 freqs, better params '''

# #Recall the data
# amp_phase_e_bar = np.zeros(15)
# X_Y_e_bar = np.zeros(15)
# NetMAP_e_bar = np.zeros(15)

# for i in range(15):
#     file_path = f'/Users/Student/Desktop/Summer Research 2024/Curve Fit vs NetMAP/15 systems - 10 Freqs NetMAP & Better Parameters/Random_Automated_Guess_{i}.xlsx'
#     array_amp_phase = pd.read_excel(file_path, sheet_name = 'Amp & Phase').to_numpy()
#     array_X_Y = pd.read_excel(file_path, sheet_name = 'X & Y').to_numpy()
#     array_NetMAP = pd.read_excel(file_path, sheet_name = 'NetMAP').to_numpy()
    
#     #Pull out <e>_bar for each trial and add to list
#     amp_phase_e_bar[i] = array_amp_phase[0, 51]
#     X_Y_e_bar[i] = array_X_Y[0, 51]
#     NetMAP_e_bar[i] = array_NetMAP[0, 45]

# #Graph!
# fig = plt.figure(figsize=(10, 6))
# linearbins = np.linspace(0,48,50)
# plt.title('Average Systematic Error Across Parameters Then Trials', fontsize = 18)
# plt.xlabel('<e> (%)', fontsize = 16)
# plt.ylabel('Counts', fontsize = 16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.hist(X_Y_e_bar, bins = linearbins, alpha=0.5, color='green', label='Cartesian (X & Y)', edgecolor='green')
# plt.hist(amp_phase_e_bar, bins =linearbins, alpha=0.5, color='blue', label='Polar (Amp & Phase)', edgecolor='blue')
# plt.hist(NetMAP_e_bar, bins = linearbins, alpha=0.5, color='red', label='NetMAP', edgecolor='red')
# plt.legend(loc='upper right', fontsize = 14)
# plt.show()

# fig = plt.figure(figsize=(10, 6))
# logbins = np.logspace(-2,1.5,50)
# plt.title('Average Systematic Error Across Parameters Then Trials', fontsize = 18)
# plt.xlabel('<e> (%)', fontsize = 16)
# plt.ylabel('Counts', fontsize = 16)
# plt.xscale('log')
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.hist(X_Y_e_bar, bins = logbins, alpha=0.5, color='green', label='Cartesian (X & Y)', edgecolor='green')
# plt.hist(amp_phase_e_bar, bins = logbins, alpha=0.5, color='blue', label='Polar (Amp & Phase)', edgecolor='blue')
# plt.hist(NetMAP_e_bar, bins = logbins, alpha=0.5, color='red', label='NetMAP', edgecolor='red')
# plt.legend(loc='upper right', fontsize = 14)
# plt.show()

''' Redoing the histogram for Case Study'''

#Recall the data
file_path = '/Users/Student/Desktop/Summer Research 2024/Curve Fit vs NetMAP/Case Study - 10 Freqs NetMAP Better Params 1000 Trials/Case_Study_1000_Trials.xlsx'   
array_amp_phase = pd.read_excel(file_path, sheet_name = 'Amp & Phase').to_numpy()
array_X_Y = pd.read_excel(file_path, sheet_name = 'X & Y').to_numpy()
array_NetMAP = pd.read_excel(file_path, sheet_name = 'NetMAP').to_numpy()

#Pull out <e> for each type
amp_phase_error = array_amp_phase[:,50]
X_Y_error = array_X_Y[:, 50]
NetMAP_error = array_NetMAP[:,44]

#Graph histograms!
linearbins = np.linspace(0,15,50)
fig = plt.figure(figsize=(5, 4))
plt.xlabel(r'$\langle e \rangle$ (%)', fontsize = 16)
plt.ylabel('Counts', fontsize = 16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.hist(X_Y_error,  bins = linearbins, alpha=0.5, color='green', label='Cartesian', edgecolor='green')
plt.hist(amp_phase_error, bins=linearbins, alpha=0.5, color='blue', label='Polar', edgecolor='blue')
plt.hist(NetMAP_error, bins = linearbins, alpha=0.5, color='red', label='NetMAP', edgecolor='red')
plt.legend(loc='upper right', fontsize = 13)
plt.show()
save_figure(fig, 'Final', 'Case Study 1000 Lin Err Hist.pdf' )

logbins = np.logspace(-2,1.5,50)
fig = plt.figure(figsize=(5, 4))
plt.xlabel(r'$\langle e \rangle$ (%)', fontsize = 16)
plt.ylabel('Counts', fontsize = 16)
plt.xscale('log')
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.hist(X_Y_error,  bins = logbins, alpha=0.5, color='green', label='Cartesian', edgecolor='green')#, histtype= 'step', lw = 2)
plt.hist(amp_phase_error, bins = logbins, alpha=0.5, color='blue', label='Polar', edgecolor='blue')#, histtype= 'step', lw = 2)
plt.hist(NetMAP_error, bins = logbins, alpha=0.5, color='red', label='NetMAP', edgecolor='red')#, histtype= 'step', lw = 2)
plt.legend(loc='upper right', fontsize = 13)
plt.show()
save_figure(fig, 'Final', 'Case Study 1000 Log Err Hist.pdf' )

def nonlinearhistc(X, bins, thresh=3, verbose=False):
    map_to_bins = np.digitize(X, bins) - 1  # Adjusting to match zero-indexing
    r = np.zeros(len(bins) - 1)  # Adjusted to match the number of intervals
    
    # Populate counts for each bin
    for i in map_to_bins:
        if 0 <= i < len(r):
            r[i] += 1 # count for bin i.
    
    if verbose:
        print(f"Counts per bin: {r}")
    
    # Normalize by bin width
    probabilitydensity = np.zeros(len(bins) - 1)
    area = 0
    thinbincount = 0

    for i in range(len(bins) - 1): # iterate through bins
        if r[i] <= 1:
            thinbincount += 1
        thisbinwidth = bins[i + 1] - bins[i]
        probabilitydensity[i] = r[i] / thisbinwidth
        area += probabilitydensity[i] * thisbinwidth # calculate total area
    
    print('Divide by area to make P dens. Area:', area)
    
    if thinbincount > thresh:
        print(f"Warning: too many bins for data, thinbincount={thinbincount}")
    elif verbose:
        print(f"thinbincount={thinbincount}")

    # Normalize area
    normedprobabilitydensity = [eachPdens / area 
                                for eachPdens in probabilitydensity]
    return normedprobabilitydensity, map_to_bins

# # Graph probability densities
# bins_i_want = np.logspace(-2, 1.5, 100)

# normprobXY, map_to_bins = nonlinearhistc(X_Y_error, bins_i_want)
# normprobampphase, map_to_bins = nonlinearhistc(amp_phase_error, bins_i_want)
# normprobNetMAP, map_to_bins = nonlinearhistc(NetMAP_error, bins_i_want)

# plt.figure(figsize=(10, 6))

# plt.loglog(bins_i_want[:-1], normprobXY , '.', color='green', alpha = 0.5, label = 'Cartesian')
# plt.loglog(bins_i_want[:-1], normprobampphase ,'.', color='blue', alpha = 0.5, label = 'Polar')
# plt.loglog(bins_i_want[:-1], normprobNetMAP , '.', color='red', alpha = 0.5, label = 'NetMAP')

# plt.xlabel('<e> (%)', fontsize=16)
# plt.ylabel('Normalized Probability Density', fontsize=16)
# plt.title('Normalized Probability Density of Average Systematic Error Across Parameters')
# plt.legend(loc='upper center', fontsize = 14)
# plt.show()



'''Creating graphs - one example
   Using the case study data (10 freq/better params 1000 trials), trial 1. 
   '''
   
# #Recall the data
# file_path = '/Users/Student/Desktop/Summer Research 2024/Curve Fit vs NetMAP/Case Study - 10 Freqs NetMAP Better Params 1000 Trials/Case_Study_1000_Trials.xlsx'   
# array_amp_phase = pd.read_excel(file_path, sheet_name = 'Amp & Phase').to_numpy()
# array_X_Y = pd.read_excel(file_path, sheet_name = 'X & Y').to_numpy()

# #True and guessed parameters
# true_params = array_amp_phase[1,:11]
# guess_params = array_amp_phase[1,11:22]
# freq = np.linspace(0.001, 4, 800)
# freq1 = np.linspace(0.001, 4, 700)

# #The recovered parameters
# recovered_params_amp_phase = array_amp_phase[1,22:33]
# recovered_params_X_Y = array_X_Y[1,22:33]

# #Error for each parameter from Amp/Phase Plots
# e_k1_amp = array_amp_phase[:, 33]
# e_k2_amp = array_amp_phase[:, 34]
# e_k3_amp = array_amp_phase[:, 35]
# e_k4_amp = array_amp_phase[:, 36]
# e_b1_amp = array_amp_phase[:, 37]
# e_b2_amp = array_amp_phase[:, 38]
# e_b3_amp = array_amp_phase[:, 39]
# e_m1_amp = array_amp_phase[:, 41]
# e_m2_amp = array_amp_phase[:, 42]
# e_m3_amp = array_amp_phase[:, 43]

# #Error for each parameter from X/Y Plots
# e_k1_XY = array_X_Y[:, 33]
# e_k2_XY = array_X_Y[:, 34]
# e_k3_XY = array_X_Y[:, 35]
# e_k4_XY = array_X_Y[:, 36]
# e_b1_XY = array_X_Y[:, 37]
# e_b2_XY = array_X_Y[:, 38]
# e_b3_XY = array_X_Y[:, 39]
# e_m1_XY = array_X_Y[:, 41]
# e_m2_XY = array_X_Y[:, 42]
# e_m3_XY = array_X_Y[:, 43]

# #Total error
# err_amp_phase = array_amp_phase[:,50]
# err_X_Y = array_X_Y[:,50]

# #1 - R^2 values
# amp1_1minusR2 = 1 - array_amp_phase[:,44]
# amp2_1minusR2 = 1 - array_amp_phase[:,45]
# amp3_1minusR2 = 1 - array_amp_phase[:,46]
# phase1_1minusR2 = 1 - array_amp_phase[:,47]
# phase2_1minusR2 = 1 - array_amp_phase[:,48]
# phase3_1minusR2 = 1 - array_amp_phase[:,49]

# X1_1minusR2 = 1 - array_X_Y[:,44]
# X2_1minusR2 = 1 - array_X_Y[:,45]
# X3_1minusR2 = 1 - array_X_Y[:,46]
# Y1_1minusR2 = 1 - array_X_Y[:,47]
# Y2_1minusR2 = 1 - array_X_Y[:,48]
# Y3_1minusR2 = 1 - array_X_Y[:,49]

'''Box plots of recovered parameter spread - only 50 trials'''
# plt.boxplot([e_k1_amp, e_k2_amp, e_k3_amp, e_k4_amp, e_b1_amp, e_b2_amp, e_b3_amp, e_m1_amp, e_m2_amp, e_m3_amp], positions=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['k1', 'k2', 'k3', 'k4', 'b1', 'b2', 'b3', 'm1', 'm2', 'm3'])
# plt.xlabel('Parameters')
# plt.ylabel('Error (%)')
# plt.title('Amplitude and Phase')
# plt.savefig('parameter_box_plot.pdf')
# plt.show()

''' How does error compare to 1-R^2?'''
#Amp, Phase
# fig = plt.figure(figsize=(16,8))
# gs = fig.add_gridspec(2, 3, hspace=0.1, wspace=0.1)
# ((ax1, ax2, ax3), (ax4, ax5, ax6)) = gs.subplots(sharex=False, sharey='row')

# ax1.plot(amp1_1minusR2, err_amp_phase,'ro', alpha=0.5, markersize=5.5)
# ax2.plot(amp2_1minusR2, err_amp_phase,'bo', alpha=0.5, markersize=5.5)
# ax3.plot(amp3_1minusR2, err_amp_phase,'go', alpha=0.5, markersize=5.5)
# ax4.plot(phase1_1minusR2, err_amp_phase,'ro', alpha=0.5, markersize=5.5)
# ax5.plot(phase2_1minusR2, err_amp_phase,'bo', alpha=0.5, markersize=5.5)
# ax6.plot(phase3_1minusR2, err_amp_phase,'go', alpha=0.5, markersize=5.5)

# ax1.set_title('Amp 1', fontsize=18)
# ax2.set_title('Amp 2', fontsize=18)
# ax3.set_title('Amp 3', fontsize=18)
# ax4.set_title('Phase 1', fontsize=18)
# ax5.set_title('Phase 2', fontsize=18)
# ax6.set_title('Phase 3', fontsize=18)
# ax1.set_ylabel('<e> (%)', fontsize=16)
# ax4.set_ylabel('<e> (%)', fontsize=16)
# ax4.set_xlabel('1-R^2', fontsize=16)
# ax5.set_xlabel('1-R^2', fontsize=16)
# ax6.set_xlabel('1-R^2', fontsize=16)

# for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
#     ax.set_xscale('log')
#     ax.set_yscale('log')

# plt.savefig('err_vs_rsquared_amp_phase.pdf')
# plt.show()

#X and Y
# fig = plt.figure(figsize=(16,8))
# gs = fig.add_gridspec(2, 3, hspace=0.1, wspace=0.1)
# ((ax1, ax2, ax3), (ax4, ax5, ax6)) = gs.subplots(sharex=False, sharey='row')

# ax1.plot(X1_1minusR2, err_X_Y,'ro', alpha=0.2, markersize=5.5)
# ax2.plot(X2_1minusR2, err_X_Y,'bo', alpha=0.2, markersize=5.5)
# ax3.plot(X3_1minusR2, err_X_Y,'go', alpha=0.2, markersize=5.5)
# ax4.plot(Y1_1minusR2, err_X_Y,'ro', alpha=0.2, markersize=5.5)
# ax5.plot(Y2_1minusR2, err_X_Y,'bo', alpha=0.2, markersize=5.5)
# ax6.plot(Y3_1minusR2, err_X_Y,'go', alpha=0.2, markersize=5.5)

# ax1.set_title('X 1', fontsize=18)
# ax2.set_title('X 2', fontsize=18)
# ax3.set_title('X 3', fontsize=18)
# ax4.set_title('Y 1', fontsize=18)
# ax5.set_title('Y 2', fontsize=18)
# ax6.set_title('Y 3', fontsize=18)
# ax1.set_ylabel('<e> (%)', fontsize=16)
# ax4.set_ylabel('<e> (%)', fontsize=16)
# ax4.set_xlabel('1-R^2', fontsize=16)
# ax5.set_xlabel('1-R^2', fontsize=16)
# ax6.set_xlabel('1-R^2', fontsize=16)

# for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
#     ax.set_xscale('log')
#     ax.set_yscale('log')
    
# plt.savefig('err_vs_rsquared_XY.pdf')
# plt.show()

# '''Graphing Amp/Phase with addition of complex plots'''
# #Create the true data - not including complex noise (so not using curve1, etc functions) because I didn't save the exact noise
# #for each trial and also this is just for visualization so it doesn't matter so much because I have the recovered parameters regardless and the noise is not noticable on the graph
# Amp1 = c1(freq1, *true_params)
# Phase1 = t1(freq1, *true_params)
# Amp2 = c2(freq1, *true_params)
# Phase2 = t2(freq1, *true_params)
# Amp3 = c3(freq1, *true_params)
# Phase3 = t3(freq1, *true_params)
# X1 = re1(freq1, *true_params)
# Y1 = im1(freq1, *true_params)
# X2 = re2(freq1, *true_params)
# Y2 = im2(freq1, *true_params)
# X3 = re3(freq1, *true_params)
# Y3 = im3(freq1, *true_params)

# #Create the initial guesses
# Amp1_guess = c1(freq, *guess_params)
# Phase1_guess = t1(freq, *guess_params)
# Amp2_guess = c2(freq, *guess_params)
# Phase2_guess = t2(freq, *guess_params)
# Amp3_guess = c3(freq, *guess_params)
# Phase3_guess = t3(freq, *guess_params)
# X1_guess = re1(freq, *guess_params)
# Y1_guess = im1(freq, *guess_params)
# X2_guess = re2(freq, *guess_params)
# Y2_guess = im2(freq, *guess_params)
# X3_guess = re3(freq, *guess_params)
# Y3_guess = im3(freq, *guess_params)

# #Create the final fit! 
# Amp1_fitted = c1(freq, *recovered_params_amp_phase)
# Phase1_fitted = t1(freq, *recovered_params_amp_phase)
# Amp2_fitted = c2(freq, *recovered_params_amp_phase)
# Phase2_fitted = t2(freq, *recovered_params_amp_phase)
# Amp3_fitted = c3(freq, *recovered_params_amp_phase)
# Phase3_fitted = t3(freq, *recovered_params_amp_phase)
# X1_fitted = re1(freq, *recovered_params_X_Y)
# Y1_fitted = im1(freq, *recovered_params_X_Y)
# X2_fitted = re2(freq, *recovered_params_X_Y)
# Y2_fitted = im2(freq, *recovered_params_X_Y)
# X3_fitted = re3(freq, *recovered_params_X_Y)
# Y3_fitted = im3(freq, *recovered_params_X_Y)

# # Begin graphing for Amp and Phase
# fig = plt.figure(figsize=(16,11))
# gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.05)

# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
# ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
# ax4 = fig.add_subplot(gs[1, 0], sharex=ax1)
# ax5 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax4)
# ax6 = fig.add_subplot(gs[1, 2], sharex=ax1, sharey=ax4)
# ax7 = fig.add_subplot(gs[2, 0], aspect='equal')
# ax8 = fig.add_subplot(gs[2, 1], sharex=ax7, sharey=ax7, aspect='equal')
# ax9 = fig.add_subplot(gs[2, 2], sharex=ax7, sharey=ax7, aspect='equal')

# #original data
# ax1.plot(freq1, Amp1,'ro-', alpha=0.5, markersize=5.5, label = 'Data')
# ax2.plot(freq1, Amp2,'bo-', alpha=0.5, markersize=5.5, label = 'Data')
# ax3.plot(freq1, Amp3,'go-', alpha=0.5, markersize=5.5, label = 'Data')
# ax4.plot(freq1, Phase1,'ro-', alpha=0.5, markersize=5.5, label = 'Data')
# ax5.plot(freq1, Phase2,'bo-', alpha=0.5, markersize=5.5, label = 'Data')
# ax6.plot(freq1, Phase3,'go-', alpha=0.5, markersize=5.5, label = 'Data')
# ax7.plot(X1,Y1,'ro-', alpha=0.5, markersize=5.5, label = 'Data')
# ax8.plot(X2,Y2,'bo-', alpha=0.5, markersize=5.5, label = 'Data')
# ax9.plot(X3,Y3,'go-', alpha=0.5, markersize=5.5, label = 'Data')

# #fitted curves
# ax1.plot(freq, Amp1_fitted,'c-', label='Fit', lw=2.5)
# ax2.plot(freq, Amp2_fitted,'r-', label='Fit', lw=2.5)
# ax3.plot(freq, Amp3_fitted,'m-', label='Fit', lw=2.5)
# ax4.plot(freq, Phase1_fitted,'c-', label='Fit', lw=2.5)
# ax5.plot(freq, Phase2_fitted,'r-', label='Fit', lw=2.5)
# ax6.plot(freq, Phase3_fitted,'m-', label='Fit', lw=2.5)
# ax7.plot(X1_fitted, Y1_fitted, 'c-', label='Fit', lw=2.5)
# ax8.plot(X2_fitted, Y2_fitted, 'r-', label='Fit', lw=2.5)
# ax9.plot(X3_fitted, Y3_fitted, 'm-', label='Fit', lw=2.5)

# #inital guess curves
# ax1.plot(freq, Amp1_guess, color='#4682B4', linestyle='dashed', label='Guess')
# ax2.plot(freq, Amp2_guess, color='#4682B4', linestyle='dashed', label='Guess')
# ax3.plot(freq, Amp3_guess, color='#4682B4', linestyle='dashed', label='Guess')
# ax4.plot(freq, Phase1_guess, color='#4682B4', linestyle='dashed', label='Guess')
# ax5.plot(freq, Phase2_guess, color='#4682B4', linestyle='dashed', label='Guess')
# ax6.plot(freq, Phase3_guess, color='#4682B4', linestyle='dashed', label='Guess')
# ax7.plot(X1_guess, Y1_guess, color='#4682B4', linestyle='dashed', label='Guess')
# ax8.plot(X2_guess, Y2_guess, color='#4682B4', linestyle='dashed', label='Guess')
# ax9.plot(X3_guess, Y3_guess, color='#4682B4', linestyle='dashed', label='Guess')


# #Graph parts
# fig.suptitle('Trimer Resonator: Amplitude and Phase', fontsize=32)
# ax1.set_title('Mass 1', fontsize=26)
# ax2.set_title('Mass 2', fontsize=26)
# ax3.set_title('Mass 3', fontsize=26)
# ax1.set_ylabel('Amplitude', fontsize=26)
# ax4.set_ylabel('Phase', fontsize=26)
# ax7.set_ylabel('Imaginary', fontsize=26)

# ax1.label_outer()
# ax2.label_outer()
# ax3.label_outer()
# ax5.tick_params(labelleft=False)
# ax6.tick_params(labelleft=False)
# ax7.label_outer()
# ax8.label_outer()
# ax9.label_outer()
    
# ax4.set_xlabel('Frequency', fontsize=26)
# ax5.set_xlabel('Frequency', fontsize=26)
# ax6.set_xlabel('Frequency', fontsize=26)
# ax7.set_xlabel('Real', fontsize=26)
# ax8.set_xlabel('Real', fontsize=26)
# ax9.set_xlabel('Real', fontsize=26)

# ax1.legend(fontsize=20)
# ax2.legend(fontsize=20)
# ax3.legend(fontsize=20)
# # ax4.legend(fontsize=20)
# # ax5.legend(fontsize=20)
# # ax6.legend(fontsize=20, loc = 'upper right')
# # ax7.legend(fontsize=20, bbox_to_anchor=(1, 1))
# # ax8.legend(fontsize=20, bbox_to_anchor=(1, 1))
# # ax9.legend(fontsize=20, bbox_to_anchor=(1, 1))

# axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
# for ax in axes:
#     ax.tick_params(axis='both', labelsize=18)  # Change tick font size

# plt.show()

# # Begin graphing for X and Y
# fig = plt.figure(figsize=(16,11))
# gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.05)

# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
# ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
# ax4 = fig.add_subplot(gs[1, 0], sharex=ax1)
# ax5 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax4)
# ax6 = fig.add_subplot(gs[1, 2], sharex=ax1, sharey=ax4)
# ax7 = fig.add_subplot(gs[2, 0], aspect='equal')
# ax8 = fig.add_subplot(gs[2, 1], sharex=ax7, sharey=ax7, aspect='equal')
# ax9 = fig.add_subplot(gs[2, 2], sharex=ax7, sharey=ax7, aspect='equal')

# #original data
# ax1.plot(freq1, X1,'ro-', alpha=0.5, markersize=5.5, label = 'Data')
# ax2.plot(freq1, X2,'bo-', alpha=0.5, markersize=5.5, label = 'Data')
# ax3.plot(freq1, X3,'go-', alpha=0.5, markersize=5.5, label = 'Data')
# ax4.plot(freq1, Y1,'ro-', alpha=0.5, markersize=5.5, label = 'Data')
# ax5.plot(freq1, Y2,'bo-', alpha=0.5, markersize=5.5, label = 'Data')
# ax6.plot(freq1, Y3,'go-', alpha=0.5, markersize=5.5, label = 'Data')
# ax7.plot(X1,Y1,'ro-', alpha=0.5, markersize=5.5, label = 'Data')
# ax8.plot(X2,Y2,'bo-', alpha=0.5, markersize=5.5, label = 'Data')
# ax9.plot(X3,Y3,'go-', alpha=0.5, markersize=5.5, label = 'Data')

# #fitted curves
# ax1.plot(freq, X1_fitted,'c-', label='Fit', lw=2.5)
# ax2.plot(freq, X2_fitted,'r-', label='Fit', lw=2.5)
# ax3.plot(freq, X3_fitted,'m-', label='Fit', lw=2.5)
# ax4.plot(freq, Y1_fitted,'c-', label='Fit', lw=2.5)
# ax5.plot(freq, Y2_fitted,'r-', label='Fit', lw=2.5)
# ax6.plot(freq, Y3_fitted,'m-', label='Fit', lw=2.5)
# ax7.plot(X1_fitted, Y1_fitted, 'c-', label='Fit', lw=2.5)
# ax8.plot(X2_fitted, Y2_fitted, 'r-', label='Fit', lw=2.5)
# ax9.plot(X3_fitted, Y3_fitted, 'm-', label='Fit', lw=2.5)

# #inital guess curves
# ax1.plot(freq, X1_guess, color='#4682B4', linestyle='dashed', label='Guess')
# ax2.plot(freq, X2_guess, color='#4682B4', linestyle='dashed', label='Guess')
# ax3.plot(freq, X3_guess, color='#4682B4', linestyle='dashed', label='Guess')
# ax4.plot(freq, Y1_guess, color='#4682B4', linestyle='dashed', label='Guess')
# ax5.plot(freq, Y2_guess, color='#4682B4', linestyle='dashed', label='Guess')
# ax6.plot(freq, Y3_guess, color='#4682B4', linestyle='dashed', label='Guess')
# ax7.plot(X1_guess, Y1_guess, color='#4682B4', linestyle='dashed', label='Guess')
# ax8.plot(X2_guess, Y2_guess, color='#4682B4', linestyle='dashed', label='Guess')
# ax9.plot(X3_guess, Y3_guess, color='#4682B4', linestyle='dashed', label='Guess')

# #Graph parts
# fig.suptitle('Trimer Resonator: Real and Imaginary', fontsize=24)
# ax1.set_title('Mass 1', fontsize=26)
# ax2.set_title('Mass 2', fontsize=26)
# ax3.set_title('Mass 3', fontsize=26)
# ax1.set_ylabel('Real', fontsize=26)
# ax4.set_ylabel('Imaginary', fontsize=26)
# ax7.set_ylabel('Imaginary', fontsize=26)

# ax1.label_outer()
# ax2.label_outer()
# ax3.label_outer()
# ax5.tick_params(labelleft=False)
# ax6.tick_params(labelleft=False)
# ax7.label_outer()
# ax8.label_outer()
# ax9.label_outer()

# ax4.set_xlabel('Frequency', fontsize=26)
# ax5.set_xlabel('Frequency', fontsize=26)
# ax6.set_xlabel('Frequency', fontsize=26)
# ax7.set_xlabel('Real', fontsize=26)
# ax8.set_xlabel('Real', fontsize=26)
# ax9.set_xlabel('Real', fontsize=26)

# ax1.legend(fontsize=20)
# ax2.legend(fontsize=20)
# ax3.legend(fontsize=20)
# # ax4.legend(fontsize=13)
# # ax5.legend(fontsize=13)
# # ax6.legend(fontsize=13)
# # ax7.legend(fontsize=13, loc='upper left', bbox_to_anchor=(1, 1))
# # ax8.legend(fontsize=13, loc='upper left', bbox_to_anchor=(1, 1))
# # ax9.legend(fontsize=13, loc='upper left', bbox_to_anchor=(1, 1))

# axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
# for ax in axes:
#     ax.tick_params(axis='both', labelsize=18)  # Change tick font size


# plt.show()






