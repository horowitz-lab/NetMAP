#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:42:41 2024

@author: lydiabullock
"""

''' Which has more accurated recovered parameters: Amp & Phase or X & Y? 
    Using method of fixing F. '''

import os
import pandas as pd
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import matplotlib.ticker as ticker
from curve_fitting_amp_phase_all import multiple_fit_amp_phase
from curve_fitting_X_Y_all import multiple_fit_X_Y
from Trimer_simulator import calculate_spectra, curve1, theta1, curve2, theta2, curve3, theta3, c1, t1, c2, t2, c3, t3, realamp1, realamp2, realamp3, imamp1, imamp2, imamp3, re1, re2, re3, im1, im2, im3
from Trimer_NetMAP import Zmatrix, unnormalizedparameters, normalize_parameters_1d_by_force
import warnings

''' Functions contained:
    complex_noise - creates noise, e
    syserr - Calculates systematic error
    find_avg_e - Calculates average across systematic error for each parameter
                 for one trial of the same system
    artithmetic_then_logarithmic - Calculates arithmetic average across parameters first, 
                                   then logarithmic average across trials 
    generate_random_system - Randomly generates parameters for system. All parameter values btw 0.1 and 10
    plot_guess - Used for the Case Study. Plots just the data and the guessed parameters curve. No curve fitting.
    automate_guess - Randomly generates guess parameters within a certain percent of the true parameters
    save_figure - Saves figures to a folder of your naming choice. Also allows you to name the figure whatever.
    get_parameters_NetMAP - Recovers parameters for a system given the guessed parameters
    run_trials - Runs a set number of trials for one system, graphs curvefit result,
                 puts data and averages into spreadsheet, returns <e>_bar for both types of curves
               - Must include number of trials to run and name of excel sheet  
    histogram_3_data_sets - incomplete but tries to graph histograms better
    
    This file also imports multiple_fit_amp_phase, which performs curve fitting on Amp vs Freq and Phase vs Freq curves for all 3 masses simultaneously,
    and multiple_fit_X_Y, which performs curve fitting on X vs Freq and Y vs Freq curves for all 3 masses simulatenously.
'''

def complex_noise(n, noiselevel):
    global complexamplitudenoisefactor
    complexamplitudenoisefactor = 0.0005
    return noiselevel* complexamplitudenoisefactor * np.random.randn(n,)

def syserr(x_found,x_set, absval = True):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        se = 100*(x_found-x_set)/x_set
    if absval:
        return abs(se)
    else:
        return se

#Calculate <e> for one trial of the same system
def find_avg_e(dictionary):
    sum_e = dictionary['e_k1'][0] + \
        dictionary['e_k2'][0] +  \
        dictionary['e_k3'][0] +  \
        dictionary['e_k4'][0] +  \
        dictionary['e_b1'][0] +  \
        dictionary['e_b2'][0] +  \
        dictionary['e_b3'][0] +  \
        dictionary['e_F'][0] +  \
        dictionary['e_m1'][0] +  \
        dictionary['e_m2'][0] +  \
        dictionary['e_m3'][0] 
    avg_e = sum_e/10
    return avg_e

#Calculate <e>_bar
def arithmetic_then_logarithmic(avg_e_list, num_trials):
    ln_avg_e = []
    for item in avg_e_list:
        ln_avg_e.append(math.log(item))
    avg_ln_avg_e = sum(ln_avg_e)/num_trials
    e_raised_to_sum = math.exp(avg_ln_avg_e)
    return e_raised_to_sum

#Randomly generates parameters of a system. All parameters between 0.1 and 10
def generate_random_system():
    system_params = []
    for i in range(11):
        if i==7: #Doing this because we must keep force the same throughout
            system_params.append(1)
        elif i==4 or i==5 or i==6:
            param = random.uniform(0.01,1)
            round_param = round(param, 3)
            system_params.append(round_param)
        else: 
            param = random.uniform(1,10)
            round_param = round(param, 3)
            system_params.append(round_param)
    return system_params

#Plots data and guessed parameters curve
def plot_guess(params_guess, params_correct):
    ##Create data - this is the same as what I use in the curve fit functions
    freq = np.linspace(0.001, 4, 300)
    
    #Create noise
    e = complex_noise(300, 2)
    force_all = False
    
    #Original Data
    X1 = realamp1(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    Y1 = imamp1(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) 
    
    X2 = realamp2(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    Y2 = imamp2(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) 
    
    X3 = realamp3(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    Y3 = imamp3(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    
    Amp1 = curve1(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    Phase1 = theta1(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) \
        + 2 * np.pi
    Amp2 = curve2(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    Phase2 = theta2(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) \
        + 2 * np.pi
    Amp3 = curve3(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    Phase3 = theta3(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) \
        + 2 * np.pi

    #Guessed Curve
    re1_guess = re1(freq, params_guess[0], params_guess[1], params_guess[2], params_guess[3], params_guess[4], params_guess[5], params_guess[6], params_guess[7], params_guess[8], params_guess[9], params_guess[10])
    re2_guess = re2(freq, params_guess[0], params_guess[1], params_guess[2], params_guess[3], params_guess[4], params_guess[5], params_guess[6], params_guess[7], params_guess[8], params_guess[9], params_guess[10])
    re3_guess = re3(freq, params_guess[0], params_guess[1], params_guess[2], params_guess[3], params_guess[4], params_guess[5], params_guess[6], params_guess[7], params_guess[8], params_guess[9], params_guess[10])
    im1_guess = im1(freq, params_guess[0], params_guess[1], params_guess[2], params_guess[3], params_guess[4], params_guess[5], params_guess[6], params_guess[7], params_guess[8], params_guess[9], params_guess[10])
    im2_guess = im2(freq, params_guess[0], params_guess[1], params_guess[2], params_guess[3], params_guess[4], params_guess[5], params_guess[6], params_guess[7], params_guess[8], params_guess[9], params_guess[10])
    im3_guess = im3(freq, params_guess[0], params_guess[1], params_guess[2], params_guess[3], params_guess[4], params_guess[5], params_guess[6], params_guess[7], params_guess[8], params_guess[9], params_guess[10])
    c1_guess = c1(freq, params_guess[0], params_guess[1], params_guess[2], params_guess[3], params_guess[4], params_guess[5], params_guess[6], params_guess[7], params_guess[8], params_guess[9], params_guess[10])
    c2_guess = c2(freq, params_guess[0], params_guess[1], params_guess[2], params_guess[3], params_guess[4], params_guess[5], params_guess[6], params_guess[7], params_guess[8], params_guess[9], params_guess[10])
    c3_guess = c3(freq, params_guess[0], params_guess[1], params_guess[2], params_guess[3], params_guess[4], params_guess[5], params_guess[6], params_guess[7], params_guess[8], params_guess[9], params_guess[10])
    t1_guess = t1(freq, params_guess[0], params_guess[1], params_guess[2], params_guess[3], params_guess[4], params_guess[5], params_guess[6], params_guess[7], params_guess[8], params_guess[9], params_guess[10])
    t2_guess = t2(freq, params_guess[0], params_guess[1], params_guess[2], params_guess[3], params_guess[4], params_guess[5], params_guess[6], params_guess[7], params_guess[8], params_guess[9], params_guess[10])
    t3_guess = t3(freq, params_guess[0], params_guess[1], params_guess[2], params_guess[3], params_guess[4], params_guess[5], params_guess[6], params_guess[7], params_guess[8], params_guess[9], params_guess[10])
    
    ## Begin graphing
    fig = plt.figure(figsize=(16,11))
    gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.05)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax5 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax4)
    ax6 = fig.add_subplot(gs[1, 2], sharex=ax1, sharey=ax4)
    ax7 = fig.add_subplot(gs[2, 0], aspect='equal')
    ax8 = fig.add_subplot(gs[2, 1], sharex=ax7, sharey=ax7, aspect='equal')
    ax9 = fig.add_subplot(gs[2, 2], sharex=ax7, sharey=ax7, aspect='equal')
    
    #original data
    ax1.plot(freq, X1,'ro', alpha=0.5, markersize=5.5, label = 'Data')
    ax2.plot(freq, X2,'bo', alpha=0.5, markersize=5.5, label = 'Data')
    ax3.plot(freq, X3,'go', alpha=0.5, markersize=5.5, label = 'Data')
    ax4.plot(freq, Y1,'ro', alpha=0.5, markersize=5.5, label = 'Data')
    ax5.plot(freq, Y2,'bo', alpha=0.5, markersize=5.5, label = 'Data')
    ax6.plot(freq, Y3,'go', alpha=0.5, markersize=5.5, label = 'Data')
    ax7.plot(X1,Y1,'ro', alpha=0.5, markersize=5.5, label = 'Data')
    ax8.plot(X2,Y2,'bo', alpha=0.5, markersize=5.5, label = 'Data')
    ax9.plot(X3,Y3,'go', alpha=0.5, markersize=5.5, label = 'Data')
    
    #inital guess curves
    ax1.plot(freq, re1_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax2.plot(freq, re2_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax3.plot(freq, re3_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax4.plot(freq, im1_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax5.plot(freq, im2_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax6.plot(freq, im3_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax7.plot(re1_guess, im1_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax8.plot(re2_guess, im2_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax9.plot(re3_guess, im3_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    
    #Graph parts
    fig.suptitle('Trimer Resonator: Real and Imaginary', fontsize=16)
    ax1.set_title('Mass 1', fontsize=14)
    ax2.set_title('Mass 2', fontsize=14)
    ax3.set_title('Mass 3', fontsize=14)
    ax1.set_ylabel('Real')
    ax4.set_ylabel('Imaginary')
    ax7.set_ylabel('Imaginary')
    
    ax1.label_outer()
    ax2.label_outer()
    ax3.label_outer()
    ax5.tick_params(labelleft=False)
    ax6.tick_params(labelleft=False)
    ax7.label_outer()
    ax8.label_outer()
    ax9.label_outer()
        
    ax4.set_xlabel('Frequency')
    ax5.set_xlabel('Frequency')
    ax6.set_xlabel('Frequency')
    ax7.set_xlabel('Real')
    ax8.set_xlabel('Real')
    ax9.set_xlabel('Real')
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    ax7.legend(fontsize='10')
    ax8.legend(fontsize='10')
    ax9.legend(fontsize='10')
    
    plt.show()
    
    ## Begin graphing
    fig = plt.figure(figsize=(16,8))
    gs = fig.add_gridspec(2, 3, hspace=0.1, wspace=0.1)
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = gs.subplots(sharex=True, sharey='row')
    
    #original data
    ax1.plot(freq, Amp1,'ro', alpha=0.5, markersize=5.5, label = 'Data')
    ax2.plot(freq, Amp2,'bo', alpha=0.5, markersize=5.5, label = 'Data')
    ax3.plot(freq, Amp3,'go', alpha=0.5, markersize=5.5, label = 'Data')
    ax4.plot(freq, Phase1,'ro', alpha=0.5, markersize=5.5, label = 'Data')
    ax5.plot(freq, Phase2,'bo', alpha=0.5, markersize=5.5, label = 'Data')
    ax6.plot(freq, Phase3,'go', alpha=0.5, markersize=5.5, label = 'Data')
    
    #inital guess curves
    ax1.plot(freq, c1_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax2.plot(freq, c2_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax3.plot(freq, c3_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax4.plot(freq, t1_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax5.plot(freq, t2_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax6.plot(freq, t3_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    
    
    #Graph parts
    fig.suptitle('Trimer Resonator: Amplitude and Phase', fontsize=16)
    ax1.set_title('Mass 1', fontsize=14)
    ax2.set_title('Mass 2', fontsize=14)
    ax3.set_title('Mass 3', fontsize=14)
    ax1.set_ylabel('Amplitude')
    ax4.set_ylabel('Phase')
    
    for ax in fig.get_axes():
        ax.set(xlabel='Frequency')
        ax.label_outer()
        ax.legend()
    
    print(f"Graphing guessed curve with guessed parameters: {params_guess}")
    
    plt.show()

#Generates random guess parameters that are within a certain percent of the true parameters
def automate_guess(true_params, percent_threshold):
    params_guess = []
    for index, value in enumerate(true_params):
        if index == 7: #Doing this because we must know what Force is going in
            params_guess.append(value)
        else:
            threshold = value * (percent_threshold / 100)
            num = random.uniform(value-threshold, value+threshold)
            rounded_num = round(num, 4) # Round to 4 decimal places
            params_guess.append(rounded_num)
    return params_guess

#Saves graphs
def save_figure(figure, folder_name, file_name):
    # Create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Save the figure to the folder
    file_path = os.path.join(folder_name, file_name)
    figure.savefig(file_path)
    plt.close(figure)

def get_parameters_NetMAP(frequencies, params_guess, params_correct, e, force_all):
    
    #Getting the complex amplitudes (data) with a function from Trimer_simulator
    #Still part of the simulation
    Z1, Z2, Z3 = calculate_spectra(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
        
    #Create the Zmatrix:
    #This is where we begin NetMAP
    trizmatrix = Zmatrix(frequencies, Z1, Z2, Z3, False)

    #Get the unnormalized parameters:
    notnormparam_tri = unnormalizedparameters(trizmatrix)

    #Normalize the parameters
    final_tri = normalize_parameters_1d_by_force(notnormparam_tri, 1)
    # parameters vector: 'm1', 'm2', 'm3', 'b1', 'b2', 'b3', 'k1', 'k2', 'k3', 'k4', 'Driving Force'

    #Put everything into dictionary
    data = {'k1_true': [params_correct[0]], 'k2_true': [params_correct[1]], 'k3_true': [params_correct[2]], 'k4_true': [params_correct[3]],
            'b1_true': [params_correct[4]], 'b2_true': [params_correct[5]], 'b3_true': [params_correct[6]],
            'm1_true': [params_correct[8]], 'm2_true': [params_correct[9]], 'm3_true': [params_correct[10]],  'F_true': [params_correct[7]],
            'k1_guess': [params_guess[0]], 'k2_guess': [params_guess[1]], 'k3_guess': [params_guess[2]], 'k4_guess': [params_guess[3]],
            'b1_guess': [params_guess[4]], 'b2_guess': [params_guess[5]], 'b3_guess': [params_guess[6]],
            'm1_guess': [params_guess[8]], 'm2_guess': [params_guess[9]], 'm3_guess': [params_guess[10]],  'F_guess': [params_guess[7]],  
            'k1_recovered': [final_tri[6]], 'k2_recovered': [final_tri[7]], 'k3_recovered': [final_tri[8]], 'k4_recovered': [final_tri[9]], 
            'b1_recovered': [final_tri[3]], 'b2_recovered': [final_tri[4]], 'b3_recovered': [final_tri[5]], 
            'm1_recovered': [final_tri[0]], 'm2_recovered': [final_tri[1]], 'm3_recovered': [final_tri[2]], 'F_recovered': [final_tri[10]], 
            'e_k1': [], 'e_k2': [], 'e_k3': [], 'e_k4': [],
            'e_b1': [], 'e_b2': [], 'e_b3': [], 'e_F': [], 
            'e_m1': [], 'e_m2': [], 'e_m3': []}
    
    #Calculate systematic error and add to data dictionary
    for param_name in ['k1','k2','k3','k4','b1','b2','b3','F','m1','m2','m3']:
        param_true = data[f'{param_name}_true'][0]
        param_fit = data[f'{param_name}_recovered'][0]
        systematic_error = syserr(param_fit, param_true)
        data[f'e_{param_name}'].append(systematic_error)
    
    return data

#Runs a set number of trials for one system, graphs curvefit result,
# puts data and averages into spreadsheet, returns <e>_bar for both types of curves
def run_trials(true_params, guessed_params, freqs_NetMAP, length_noise_NetMAP, num_trials, excel_file_name, graph_folder_name):
    
    starting_row = 0
    avg_e1_list = [] #Polar
    avg_e2_list = [] #Cartesian
    avg_e3_list = [] #NetMAP
    
    #Put data into excel spreadsheet
    with pd.ExcelWriter(excel_file_name, engine='xlsxwriter') as writer:
        for i in range(num_trials):
            
            #Create noise
            e = complex_noise(300, 2)
            
            ##For NetMAP
            #create error
            e_NetMAP = complex_noise(length_noise_NetMAP,2)
        
            #Get the data!
            dictionary1 = multiple_fit_amp_phase(guessed_params, true_params, e, False, True, False, graph_folder_name, f'Polar_fig_{i}') #Polar, Fixed force
            dictionary2 = multiple_fit_X_Y(guessed_params, true_params, e, False, True, graph_folder_name, f'Cartesian_fig_{i}') #Cartesian, Fixed force
            dictionary3 = get_parameters_NetMAP(freqs_NetMAP, guessed_params, true_params, e_NetMAP, False) #NetMAP
        
            #Find <e> (average across parameters) for each trial and add to dictionary
            avg_e1 = find_avg_e(dictionary1) #Polar
            dictionary1['<e>'] = avg_e1
            
            avg_e2 = find_avg_e(dictionary2) #Cartesian
            dictionary2['<e>'] = avg_e2
            
            avg_e3 = find_avg_e(dictionary3) #NetMAP
            dictionary3['<e>'] = avg_e3
            
            #Append to list for later graphing
            avg_e1_list.append(avg_e1)
            avg_e2_list.append(avg_e2)
            avg_e3_list.append(avg_e3)
        
            #Turn data into dataframe for excel
            dataframe1 = pd.DataFrame(dictionary1)
            dataframe2 = pd.DataFrame(dictionary2)
            dataframe3 = pd.DataFrame(dictionary3)
            
            #Add to excel spreadsheet
            dataframe1.to_excel(writer, sheet_name='Amp & Phase', startrow=starting_row, index=False, header=(i==0))
            dataframe2.to_excel(writer, sheet_name='X & Y', startrow=starting_row, index=False, header=(i==0))
            dataframe3.to_excel(writer, sheet_name='NetMAP', startrow=starting_row, index=False, header=(i==0))
            
            starting_row += len(dataframe1) + (1 if i==0 else 0)
        
        avg_e1_bar = arithmetic_then_logarithmic(avg_e1_list, num_trials) 
        avg_e2_bar = arithmetic_then_logarithmic(avg_e2_list, num_trials)
        avg_e3_bar = arithmetic_then_logarithmic(avg_e3_list, num_trials)
        
        dataframe1.at[0,'<e>_bar'] = avg_e1_bar
        dataframe2.at[0,'<e>_bar'] = avg_e2_bar
        dataframe3.at[0,'<e>_bar'] = avg_e3_bar
        
        dataframe1.to_excel(writer, sheet_name='Amp & Phase', index=False)
        dataframe2.to_excel(writer, sheet_name='X & Y', index=False)
        dataframe3.to_excel(writer, sheet_name='NetMAP', index=False)
        
        return avg_e1_list, avg_e2_list, avg_e3_list, avg_e1_bar, avg_e2_bar, avg_e3_bar
    
#Incomplete
def histogram_3_data_sets(data1, data2, data3, data1_name, data2_name, data3_name, graph_title, x_label):
    #Data 1 = polar, Data 2 = X&Y, Data 3 = NetMAP
    
    fig = plt.figure(figsize=(10, 6))
    spread1 = (max(data1)-min(data1))
    spread2 = (max(data2)-min(data2))
    spread3 = (max(data3)-min(data3))
    
    #If 1 and 2 overlap but no overlap with 3
    #3 can be above or below 1 and 2
    if (max(data1)>=min(data2) or max(data2)>=min(data1)) and ((max(data1)<min(data3) and max(data2)<min(data3)) or (max(data1)>min(data3) and max(data2)>min(data3))):
        
        #If 2 is greater than 1
        if max(data1) >= min(data2) and (max(data1)<min(data3) and max(data2)<min(data3)):
            bax = brokenaxes(xlims=((min(data1)-min(data1)*0.1, max(data2)+max(data2)*0.1), (min(data3)-min(data3)*0.1, max(data3)+max(data3)*0.1)), hspace=.05)
            bax.set_title(graph_title)
            bax.set_xlabel(x_label)
            bax.set_ylabel('Counts')
            bax.hist(data1, bins=10, alpha=0.75, color='blue', label=data1_name, edgecolor='black')
            bax.hist(data2, bins=10, alpha=0.75, color='green', label=data2_name, edgecolor='black')
            bax.hist(data3, bins=10, alpha=0.75, color='red', label=data3_name, edgecolor='black')
            bax.legend(loc='upper center')
            
            # Adjust the scales
            bax.axs[0].set_xlim(min(data1)-spread1*0.1,  max(data2)+spread2*0.1) #left
            bax.axs[1].set_xlim(min(data3)-spread3*0.1, max(data3)+spread3*0.1)  #right
        
            bax.axs[0].xaxis.set_major_locator(ticker.MaxNLocator(5))
            bax.axs[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.3f'))
            bax.axs[1].xaxis.set_major_locator(ticker.MaxNLocator(5))
            bax.axs[1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.3f'))
            
        #If 1 is greater than 2
        elif max(data1) >= min(data2) and (max(data1)>min(data3) and max(data2)>min(data3)):
            bax = brokenaxes(xlims=((min(data2)-min(data2)*0.1, max(data1)+max(data1)*0.1), (min(data3)-min(data3)*0.1, max(data3)+max(data3)*0.1)), hspace=.05)
            bax.set_title(graph_title)
            bax.set_xlabel(x_label)
            bax.set_ylabel('Counts')
            bax.hist(data1, bins=10, alpha=0.75, color='blue', label=data1_name, edgecolor='black')
            bax.hist(data2, bins=10, alpha=0.75, color='green', label=data2_name, edgecolor='black')
            bax.hist(data3, bins=10, alpha=0.75, color='red', label=data3_name, edgecolor='black')
            bax.legend(loc='upper center')
            
            # Adjust the scales
            bax.axs[0].set_xlim(min(data2)-spread2*0.1,  max(data1)+spread1*0.1) #left
            bax.axs[1].set_xlim(min(data3)-spread3*0.1, max(data3)+spread3*0.1)  #right
        
            bax.axs[0].xaxis.set_major_locator(ticker.MaxNLocator(5))
            bax.axs[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.3f'))
            bax.axs[1].xaxis.set_major_locator(ticker.MaxNLocator(5))
            bax.axs[1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.3f'))
    
    #If 2 and 3 overlap but no overlap with 1
    elif (max(data2)>=min(data3) or max(data3)>=min(data2)) and max(data1)<min(data3) and max(data2)<min(data3):
        
        #If 2 is greater than 1
        if max(data1) >= min(data2):
            bax = brokenaxes(xlims=((min(data1)-min(data1)*0.1, max(data2)+max(data2)*0.1), (min(data3)-min(data3)*0.1, max(data3)+max(data3)*0.1)), hspace=.05)
            bax.set_title(graph_title)
            bax.set_xlabel(x_label)
            bax.set_ylabel('Counts')
            bax.hist(data1, bins=10, alpha=0.75, color='blue', label=data1_name, edgecolor='black')
            bax.hist(data2, bins=10, alpha=0.75, color='green', label=data2_name, edgecolor='black')
            bax.hist(data3, bins=10, alpha=0.75, color='red', label=data3_name, edgecolor='black')
            bax.legend(loc='upper center')
            
            # Adjust the scales
            bax.axs[0].set_xlim(min(data1)-spread1*0.1,  max(data2)+spread2*0.1) #left
            bax.axs[1].set_xlim(min(data3)-spread3*0.1, max(data3)+spread3*0.1)  #right
        
            bax.axs[0].xaxis.set_major_locator(ticker.MaxNLocator(5))
            bax.axs[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.3f'))
            bax.axs[1].xaxis.set_major_locator(ticker.MaxNLocator(5))
            bax.axs[1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.3f'))
            
        #If 1 is greater than 2
        else:
            bax = brokenaxes(xlims=((min(data2)-min(data2)*0.1, max(data1)+max(data1)*0.1), (min(data3)-min(data3)*0.1, max(data3)+max(data3)*0.1)), hspace=.05)
            bax.set_title(graph_title)
            bax.set_xlabel(x_label)
            bax.set_ylabel('Counts')
            bax.hist(data1, bins=10, alpha=0.75, color='blue', label=data1_name, edgecolor='black')
            bax.hist(data2, bins=10, alpha=0.75, color='green', label=data2_name, edgecolor='black')
            bax.hist(data3, bins=10, alpha=0.75, color='red', label=data3_name, edgecolor='black')
            bax.legend(loc='upper center')
            
            # Adjust the scales
            bax.axs[0].set_xlim(min(data2)-spread2*0.1,  max(data1)+spread1*0.1) #left
            bax.axs[1].set_xlim(min(data3)-spread3*0.1, max(data3)+spread3*0.1)  #right
        
            bax.axs[0].xaxis.set_major_locator(ticker.MaxNLocator(5))
            bax.axs[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.3f'))
            bax.axs[1].xaxis.set_major_locator(ticker.MaxNLocator(5))
            bax.axs[1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.3f'))
    
    plt.show()

''' Begin work here. Case Study. '''

# #Make parameters/initial guesses - [k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3]
# #Note that right now we only scale/fix by F, so make sure to keep F correct in guesses
# true_params = generate_random_system()
# guessed_params = [1,1,1,1,1,1,1,1,1,1,1]

# # Start the loop
# while True:
#     # Graph
#     plot_guess(guessed_params, true_params) 
    
#     # Ask the user for the new list of guessed parameters
#     print(f'Current list of parameter guesses is {guessed_params}')
#     indices = input("Enter the indices of the elements you want to update (comma-separated, or 'c' to continue to curve fit): ")
    
#     # Check if the user wants to quit
#     if indices.lower() == 'c':
#         break
    
#     # Parse and validate the indices
#     try:
#         index_list = [int(idx.strip()) for idx in indices.split(',')]
#         if any(index < 0 or index >= len(guessed_params) for index in index_list):
#             print(f"Invalid indices. Please enter values between 0 and {len(guessed_params)-1}.")
#             continue
#     except ValueError:
#         print("Invalid input. Please enter valid indices or 'c' to continue to curve fit.")
#         continue
    
#     # Ask the user for the new values
#     values = input(f"Enter the new values for indices {index_list} (comma-separated): ")
    
#     # Parse and validate the new values
#     try:
#         value_list = [float(value.strip()) for value in values.split(',')]
#         if len(value_list) != len(index_list):
#             print("The number of values must match the number of indices.")
#             continue
#     except ValueError:
#         print("Invalid input. Please enter valid numbers.")
#         continue
    
#     # Update the list with the new values
#     for index, new_value in zip(index_list, value_list):
#         guessed_params[index] = new_value
        
# #Curve fit with the guess made above and get average lists
# #Will not do anything with <e>_bar for a single case study
# freqs_NetMAP = np.linspace(0.001, 4, 10)
# length_noise = 10
# avg_e1_list, avg_e2_list, avg_e3_list, avg_e1_bar, avg_e2_bar, avg_e3_bar = run_trials(true_params, guessed_params, freqs_NetMAP, length_noise, 50, 'Case_Study.xlsx', 'Case Study Plots')

# #Graph histogram of <e> for curve fits

# plt.title('Average Systematic Error Across Parameters')
# plt.xlabel('<e>')
# plt.ylabel('Counts')
# plt.hist(avg_e2_list, alpha=0.5, color='green', label='Cartesian (X & Y)', edgecolor='black')
# plt.hist(avg_e1_list, alpha=0.5, color='blue', label='Polar (Amp & Phase)', edgecolor='black')
# plt.hist(avg_e3_list, bins=50, alpha=0.5, color='red', label='NetMAP', edgecolor='black')
# plt.legend(loc='upper center')

# plt.show()

''' Begin work here. Automated guesses. '''

# avg_e1_bar_list = [] 
# avg_e2_bar_list = []
# avg_e3_bar_list = []

# for i in range(15):
    
#     #Generate system and guess parameters
#     true_params = generate_random_system()
#     guessed_params = automate_guess(true_params, 20)
    
#     #Curve fit with the guess made above
#     freqs_NetMAP = np.linspace(0.001, 4, 10)
#     length_noise = 10
#     avg_e1_list, avg_e2_list, avg_e3_list, avg_e1_bar, avg_e2_bar, avg_e3_bar = run_trials(true_params, guessed_params, freqs_NetMAP, length_noise, 50, f'Random_Automated_Guess_{i}.xlsx', f'Sys {i} - Rand Auto Guess Plots')
    
#     #Add <e>_bar to lists to make one graph at the end
#     avg_e1_bar_list.append(avg_e1_bar) #Polar
#     avg_e2_bar_list.append(avg_e2_bar) #Cartesian
#     avg_e3_bar_list.append(avg_e3_bar) #NetMAP
    
#     #Graph histogram of <e> for curve fits
#     fig = plt.figure(figsize=(10, 6))
#     plt.title('Average Systematic Error Across Parameters')
#     plt.xlabel('<e>')
#     plt.ylabel('Counts')
#     plt.hist(avg_e2_list, alpha=0.5, color='green', label='Cartesian (X & Y)', edgecolor='black')
#     plt.hist(avg_e1_list, alpha=0.5, color='blue', label='Polar (Amp & Phase)', edgecolor='black')
#     plt.hist(avg_e3_list, bins=50, alpha=0.5, color='red', label='NetMAP', edgecolor='black')
#     plt.legend(loc='upper center')

#     plt.show()
#     save_figure(fig, f'Sys {i} - Rand Auto Guess Plots', '<e> Histogram ' )
    
# #Graph histogram of <e>_bar for both curve fits
# fig = plt.figure(figsize=(10, 6))

# # if max(avg_e2_bar_list) >= min(avg_e1_bar_list):
# plt.hist(avg_e2_bar_list, bins=10, alpha=0.75, color='green', label='Cartesian (X & Y)', edgecolor='black')
# plt.hist(avg_e1_bar_list, bins=10, alpha=0.75, color='blue', label='Polar (Amp & Phase)', edgecolor='black')
# plt.hist(avg_e3_bar_list, bins=10, alpha=0.75, color='red', label='NetMAP', edgecolor='black')
# plt.title('Average Error Across Parameters Then Across Trials')
# plt.xlabel('<e> (%)')
# plt.ylabel('Counts')
# plt.legend(loc='upper center')

# plt.show()
# fig.savefig('<e>_bar_Histogram.png')  

''' Begin work here. Checking Worst System. '''

## System 0 from 15 Systems - 10 Freqs NetMAP
## Expecting there to be no error in recovery for everything
# true_parameters = [1.045, 0.179, 3.852, 1.877, 5.542, 1.956, 3.71, 1, 3.976, 0.656, 3.198]
# guessed_parameters = [1.2379, 0.1764, 3.7327, 1.8628, 5.93, 2.1793, 4.2198, 1, 4.3335, 0.7016, 3.0719]

# #Run the trials with 0 error 
# # MUST CHANGE ERROR IN run_trials AND IN get_parameters_NetMAP
# freqs_NetMAP = np.linspace(0.001, 4, 10)
# length_noise = 0
# avg_e1_list, avg_e2_list, avg_e3_list, avg_e1_bar, avg_e2_bar, avg_e3_bar = run_trials(true_parameters, guessed_parameters, freqs_NetMAP, length_noise, 50, 'Sys0_No_Error.xlsx', 'Sys0_No_Error - Plots')

# #Plot histogram
# plt.title('Average Systematic Error Across Parameters')
# plt.xlabel('<e>')
# plt.ylabel('Counts')
# plt.hist(avg_e2_list, alpha=0.5, color='green', label='Cartesian (X & Y)', edgecolor='black')
# plt.hist(avg_e1_list, alpha=0.5, color='blue', label='Polar (Amp & Phase)', edgecolor='black')
# plt.hist(avg_e3_list, bins=50, alpha=0.5, color='red', label='NetMAP', edgecolor='black')
# plt.legend(loc='upper center')
# plt.show()
# plt.savefig('<e>_Histogram_Sys0_no_error.png')


