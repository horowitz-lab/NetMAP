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
from curve_fitting_amp_phase_all import multiple_fit_amp_phase
from curve_fitting_X_Y_all import multiple_fit_X_Y
from Trimer_simulator import calculate_spectra, curve1, theta1, curve2, theta2, curve3, theta3, c1, t1, c2, t2, c3, t3, realamp1, realamp2, realamp3, imamp1, imamp2, imamp3, re1, re2, re3, im1, im2, im3
from Trimer_NetMAP import Zmatrix, unnormalizedparameters, normalize_parameters_1d_by_force
import warnings
import time

''' Functions contained:
    complex_noise - creates noise, e
    syserr - Calculates systematic error
    generate_random_system - Randomly generates parameters for system. All parameter values btw 0.1 and 10
    plot_guess - Used for the Case Study. Plots just the data and the guessed parameters curve. No curve fitting.
    automate_guess - Randomly generates guess parameters within a certain percent of the true parameters
    save_figure - Saves figures to a folder of your naming choice. Also allows you to name the figure whatever.
    get_parameters_NetMAP - Recovers parameters for a system given the guessed parameters
    run_trials - Runs a set number of trials for one system, graphs curvefit result,
                 puts data and averages into spreadsheet, returns <e>_bar for both types of curves
               - Must include number of trials to run and name of excel sheet  

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
    figure.savefig(file_path, bbox_inches = 'tight')
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
    
    #Put everything into a np array
    #Order added: k1, k2, k3, k4, b1, b2, b3, F,  m1, m2, m3
    data_array = np.zeros(45)
    data_array[:11] += np.array(params_correct)
    data_array[11:22] += np.array(params_guess)
    #Adding the recovered parameters and fixing the order
    data_array[22:26] += np.array(final_tri[6:10])
    data_array[26:29] += np.array(final_tri[3:6])
    data_array[29] += np.array(final_tri[-1])
    data_array[30:33] += np.array(final_tri[:3])
    #adding systematic error calculations
    syserr_result = syserr(data_array[22:33], data_array[:11])
    data_array[33:44] += np.array(syserr_result)
    data_array[-1] += np.sum(data_array[33:44]/10) #dividing by 10 because we aren't counting the error in Force because it is 0
    
    return data_array

#Runs a set number of trials for one system, graphs curvefit result,
# puts data and averages into spreadsheet, returns avg_e arrays and <e>_bar for all types of curves
def run_trials(true_params, guessed_params, freqs_NetMAP, length_noise_NetMAP, num_trials, excel_file_name, graph_folder_name):

    #Needed for calculating e_bar and for graphing - also these are things that will be returned
    avg_e1_array = np.zeros(num_trials) #Polar
    avg_e2_array = np.zeros(num_trials) #Cartesian
    avg_e3_array = np.zeros(num_trials) #NetMAP
    
    #Needed to add all the data to a spreadsheet at the end
    all_data1 = np.empty((0, 51))  #Polar
    all_data2 = np.empty((0, 51))  #Cartesian
    all_data3 = np.empty((0, 45))  #NetMAP
    
    with pd.ExcelWriter(excel_file_name, engine='xlsxwriter') as writer:
        for i in range(num_trials):
            
            #Create noise
            e = complex_noise(300, 2)
            
            ##For NetMAP
            #create error
            e_NetMAP = complex_noise(length_noise_NetMAP,2)
        
            #Get the data!
            array1 = multiple_fit_amp_phase(guessed_params, true_params, e, False, True, False, graph_folder_name, f'Polar_fig_{i}') #Polar, Fixed force
            array2 = multiple_fit_X_Y(guessed_params, true_params, e, False, True, graph_folder_name, f'Cartesian_fig_{i}') #Cartesian, Fixed force
            array3 = get_parameters_NetMAP(freqs_NetMAP, guessed_params, true_params, e_NetMAP, False) #NetMAP
            
            #Find <e> (average across parameters) for each trial and add to arrays
            avg_e1_array[i] += array1[-1]
            avg_e2_array[i] += array2[-1]
            avg_e3_array[i] += array3[-1]
            
            #Stack to the larger array
            all_data1 = np.vstack((all_data1, array1))
            all_data2 = np.vstack((all_data2, array2))
            all_data3 = np.vstack((all_data3, array3))
            
            
        avg_e1_bar = math.exp(sum(np.log(avg_e1_array))/num_trials)
        avg_e2_bar = math.exp(sum(np.log(avg_e2_array))/num_trials)
        avg_e3_bar = math.exp(sum(np.log(avg_e3_array))/num_trials)
        
        
        #For labeling the excel sheet
        param_names = ['k1_true', 'k2_true', 'k3_true', 'k4_true',
                       'b1_true', 'b2_true', 'b3_true',
                       'F_true', 'm1_true', 'm2_true', 'm3_true',
                       'k1_guess', 'k2_guess', 'k3_guess', 'k4_guess',
                       'b1_guess', 'b2_guess', 'b3_guess',
                       'F_guess', 'm1_guess', 'm2_guess', 'm3_guess',
                       'k1_recovered', 'k2_recovered', 'k3_recovered', 'k4_recovered', 
                       'b1_recovered', 'b2_recovered', 'b3_recovered',
                       'F_recovered', 'm1_recovered', 'm2_recovered', 'm3_recovered', 
                       'e_k1', 'e_k2', 'e_k3', 'e_k4',
                       'e_b1', 'e_b2', 'e_b3', 'e_F',
                       'e_m1', 'e_m2', 'e_m3',
                       'Amp1_rsqrd', 'Amp2_rsqrd', 'Amp3_rsqrd',
                       'Phase1_rsqrd', 'Phase2_rsqrd', 'Phase3_rsqrd', '<e>']
        
        #Turn the final data arrays into a dataframe so they can be written to excel
        dataframe1 = pd.DataFrame(all_data1, columns=param_names)
        dataframe2 = pd.DataFrame(all_data2, columns=param_names)
        dataframe3 = pd.DataFrame(all_data3, columns=param_names[:44]+[param_names[-1]])
        
        #Add <e>_bar values to data frame
        dataframe1.at[0,'<e>_bar'] = avg_e1_bar
        dataframe2.at[0,'<e>_bar'] = avg_e2_bar
        dataframe3.at[0,'<e>_bar'] = avg_e3_bar
        
        dataframe1.to_excel(writer, sheet_name='Amp & Phase', index=False)
        dataframe2.to_excel(writer, sheet_name='X & Y', index=False)
        dataframe3.to_excel(writer, sheet_name='NetMAP', index=False)
        
        # return avg_e1_array, avg_e2_array, avg_e3_array, avg_e1_bar, avg_e2_bar, avg_e3_bar
        return avg_e1_array, avg_e2_array, avg_e3_array, avg_e1_bar, avg_e2_bar, avg_e3_bar


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
# avg_e1_array, avg_e2_array, avg_e3_array, avg_e1_bar, avg_e2_bar, avg_e3_bar = run_trials(true_params, guessed_params, freqs_NetMAP, length_noise, 10, 'Case_Study.xlsx', 'Case Study Plots')

# #Graph histogram of <e> for curve fits

# plt.title('Average Systematic Error Across Parameters')
# plt.xlabel('<e>')
# plt.ylabel('Counts')
# plt.hist(avg_e2_array, alpha=0.5, color='green', label='Cartesian (X & Y)', edgecolor='black')
# plt.hist(avg_e1_array, alpha=0.5, color='blue', label='Polar (Amp & Phase)', edgecolor='black')
# plt.hist(avg_e3_array, bins=50, alpha=0.5, color='red', label='NetMAP', edgecolor='black')
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
#     avg_e1_array, avg_e2_array, avg_e3_array, avg_e1_bar, avg_e2_bar, avg_e3_bar = run_trials(true_params, guessed_params, freqs_NetMAP, length_noise, 50, f'Random_Automated_Guess_{i}.xlsx', f'Sys {i} - Rand Auto Guess Plots')
    
#     #Add <e>_bar to lists to make one graph at the end
#     avg_e1_bar_list.append(avg_e1_bar) #Polar
#     avg_e2_bar_list.append(avg_e2_bar) #Cartesian
#     avg_e3_bar_list.append(avg_e3_bar) #NetMAP
    
#     #Graph histogram of <e> for curve fits
#     fig = plt.figure(figsize=(10, 6))
#     plt.title('Average Systematic Error Across Parameters')
#     plt.xlabel('<e>')
#     plt.ylabel('Counts')
#     plt.hist(avg_e2_array, alpha=0.5, color='green', label='Cartesian (X & Y)', edgecolor='black')
#     plt.hist(avg_e1_array, alpha=0.5, color='blue', label='Polar (Amp & Phase)', edgecolor='black')
#     plt.hist(avg_e3_array, bins=50, alpha=0.5, color='red', label='NetMAP', edgecolor='black')
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
# avg_e1_array, avg_e2_array, avg_e3_array, avg_e1_bar, avg_e2_bar, avg_e3_bar = run_trials(true_parameters, guessed_parameters, freqs_NetMAP, length_noise, 50, 'Sys0_No_Error.xlsx', 'Sys0_No_Error - Plots')

# #Plot histogram
# plt.title('Average Systematic Error Across Parameters')
# plt.xlabel('<e>')
# plt.ylabel('Counts')
# plt.hist(avg_e2_array, alpha=0.5, color='green', label='Cartesian (X & Y)', edgecolor='black')
# plt.hist(avg_e1_array, alpha=0.5, color='blue', label='Polar (Amp & Phase)', edgecolor='black')
# plt.hist(avg_e3_array, bins=50, alpha=0.5, color='red', label='NetMAP', edgecolor='black')
# plt.legend(loc='upper center')
# plt.show()
# plt.savefig('<e>_Histogram_Sys0_no_error.png')

'''Begin work here. Redoing Case Study - 10 Freqs Better Params with 1000 trials instead of 50 '''

# file_path = '/Users/Student/Desktop/Summer Research 2024/Curve Fit vs NetMAP/Case Study - 10 Freqs NetMAP & Better Parameters/Case_Study_10_Freqs_Better_Parameters.xlsx'   
# array_amp_phase = pd.read_excel(file_path, sheet_name = 'Amp & Phase').to_numpy()
# array_X_Y = pd.read_excel(file_path, sheet_name = 'X & Y').to_numpy()

# true_params = np.concatenate((array_amp_phase[1,:7], [array_amp_phase[1,10]], array_amp_phase[1,7:10]))
# guessed_params = np.concatenate((array_amp_phase[1,11:18], [array_amp_phase[1,21]], array_amp_phase[1,18:21]))

# freq = np.linspace(0.001, 4, 300)
# freqs_NetMAP = np.linspace(0.001, 4, 10)
# length_noise = 10

# run_trials(true_params, guessed_params, freqs_NetMAP, length_noise, 1000, 'Case_Study_1000_Trials.xlsx', 'Case Study 1000 Trials Plots')

'''Begin work here. Redoing 15 systems data. Still using 10 Freqs and Better Params.
   I want to do many more systems and 500 trials per system. Seeing how many systems it can do in 3 hours.'''


## 1. Make sure I save the error used for each trial. NOT DONE
## 2. Set a runtime limit of 2-3 hours perhaps. NOT DONE
## 3. Don't graph all the curvefits. DONE
## 4. Guesses are automated to within 20% of generated parameters, 10 evenly spaced frequencies for NetMAP, noise level 2 and n=300. 

# Set the time limit in seconds
time_limit = 10800  # 3 hours

# Record the start time
start_time = time.time()

avg_e_bar_list_polar = [] 
avg_e_bar_list_cartesian = []
avg_e_bar_list_NetMAP = []

for i in range(100):
    
    # Check if the time limit has been exceeded
    elapsed_time = time.time() - start_time
    if elapsed_time > time_limit:
        print("Time limit exceeded. Exiting loop.")
        break
    
    loop_start_time = time.time()
    
    #Generate system and guess parameters
    true_params = generate_random_system()
    guessed_params = automate_guess(true_params, 20)
    print(true_params)
    print(guessed_params)
    
    #Curve fit with the guess made above
    freqs_NetMAP = np.linspace(0.001, 4, 10)
    length_noise = 10
    avg_e1_array, avg_e2_array, avg_e3_array, avg_e1_bar, avg_e2_bar, avg_e3_bar = run_trials(true_params, guessed_params, freqs_NetMAP, length_noise, 250, f'System_{i}_500.xlsx', f'Sys {i} - Rand Auto Guess Plots')
    
    #Add <e>_bar to lists to make one graph at the end
    avg_e_bar_list_polar.append(avg_e1_bar) #Polar
    avg_e_bar_list_cartesian.append(avg_e2_bar) #Cartesian
    avg_e_bar_list_NetMAP.append(avg_e3_bar) #NetMAP
    
    linearbins = np.linspace(0,15,50)
    #Graph histogram of <e> for curve fits
    fig = plt.figure(figsize=(5, 4))
    # plt.title('Average Systematic Error Across Parameters')
    plt.xlabel('<e> (%)', fontsize = 16)
    plt.ylabel('Counts', fontsize = 16)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.hist(avg_e2_array, bins = linearbins, alpha=0.5, color='green', label='Cartesian (X & Y)', edgecolor='black')
    plt.hist(avg_e1_array, bins = linearbins, alpha=0.5, color='blue', label='Polar (Amp & Phase)', edgecolor='black')
    plt.hist(avg_e3_array, bins = linearbins, alpha=0.5, color='red', label='NetMAP', edgecolor='black')
    plt.legend(loc='upper right', fontsize = 13)

    plt.show()
    save_figure(fig, 'More Systems 250 - <e> Histograms', f'<e> Lin Hist System {i}')
    
    # logbins = np.logspace(-2,1.5,50)
    # #Graph histogram of <e> for curve fits
    # fig = plt.figure(figsize=(5, 4))
    # # plt.title('Average Systematic Error Across Parameters')
    # plt.xlabel('<e> (%)', fontsize = 16)
    # plt.ylabel('Counts', fontsize = 16)
    # plt.yticks(fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.hist(avg_e2_array, bins = logbins, alpha=0.5, color='green', label='Cartesian (X & Y)', edgecolor='black')
    # plt.hist(avg_e1_array, bins = logbins, alpha=0.5, color='blue', label='Polar (Amp & Phase)', edgecolor='black')
    # plt.hist(avg_e3_array, bins = logbins, alpha=0.5, color='red', label='NetMAP', edgecolor='black')
    # plt.legend(loc='upper right', fontsize = 13)

    # plt.show()
    # save_figure(fig, 'More Systems 500 - <e> Histograms', f'<e> Log Hist System {i}')
    
    loop_end_time = time.time()
    loop_time = loop_end_time - loop_start_time
    
    print(f"Iteration {i + 1} completed. Loop time: {loop_time}")
    

#Graph histogram of <e>_bar for both curve fits

linearbins = np.linspace(0,15,50)
#Graph!
fig = plt.figure(figsize=(5, 4))
plt.xlabel('<e> Bar (%)', fontsize = 16)
plt.ylabel('Counts', fontsize = 16)
plt.xlim(0.02, 15)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.hist(avg_e_bar_list_cartesian, bins = linearbins, alpha=0.5, color='green', label='Cartesian (X & Y)', edgecolor='green')
plt.hist(avg_e_bar_list_polar, bins = linearbins, alpha=0.5, color='blue', label='Polar (Amp & Phase)', edgecolor='blue')
plt.hist(avg_e_bar_list_NetMAP, bins = linearbins, alpha=0.5, color='red', label='NetMAP', edgecolor='red')
plt.legend(loc='upper right', fontsize = 13)
plt.show()
save_figure(fig, 'More Systems 250 - <e> Histograms', '<e> Bar Lin Hist' )


logbins = np.logspace(-2,1.5,50)
#Graph!
fig = plt.figure(figsize=(5, 4))
plt.xlabel('<e> Bar (%)', fontsize = 16)
plt.ylabel('Counts', fontsize = 16)
plt.xscale('log')
plt.xlim(0.02, 15)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.hist(avg_e_bar_list_cartesian,  bins = logbins, alpha=0.5, color='green', label='Cartesian (X & Y)', edgecolor='green')
plt.hist(avg_e_bar_list_polar, bins = logbins, alpha=0.4, color='blue', label='Polar (Amp & Phase)', edgecolor='blue')
plt.hist(avg_e_bar_list_NetMAP, bins = logbins, alpha=0.5, color='red', label='NetMAP', edgecolor='red')
plt.legend(loc='upper right', fontsize = 13)
plt.show()
save_figure(fig, 'More Systems 250 - <e> Histograms', '<e> Bar Log Hist' )


# End time
end_time = time.time()
print("Time Elapsed:", end_time - start_time)

