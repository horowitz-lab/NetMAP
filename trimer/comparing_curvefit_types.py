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
import timeit
import statistics

''' Functions contained:
    complex_noise - creates noise, e
    syserr - Calculates systematic error
    generate_random_system - Randomly generates parameters for system. Parameter values btw 0.1 and 10 for all but the coefficients of friction which is between 0.1 and 1.  
    plot_guess - Used for the Case Study. Plots just the data and the guessed parameters curve. No curve fitting.
    automate_guess - Randomly generates guess parameters within a certain percent of the true parameters
    save_figure - Saves figures to a folder of your naming choice. Also allows you to name the figure whatever.
    timeit_function - Uses the timeit package to time how long a function takes to run. 
                    - Runs it multiple times (number of your choosing) and returns the average time and std dev for more accurate results.
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
# np.random.radn returns a number from a gaussian distribution with variance 1 and mean 0
# noiselevel* complexamplitudenoisefactor is standard deviation


def syserr(x_found,x_set, absval = True):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        se = 100*(x_found-x_set)/x_set
    if absval:
        return abs(se)
    else:
        return se

#Randomly generates parameters of a system. k1, k2, k3, k4, b1, b2, b3, F,  m1, m2, m3
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

# runs > 1 if you want to run one function several times to get the average time
def timeit_function(func, args=None, kwargs=None, runs=7):
    args = args or ()
    kwargs = kwargs or {}

    times = []
    for _ in range(runs):
        t = timeit.timeit(lambda: func(*args, **kwargs), number=1)
        times.append(t)

    mean_time = statistics.mean(times)
    std_dev = statistics.stdev(times) if runs > 1 else 0.0
    return mean_time, std_dev, times

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
    data_array = np.zeros(46) #44 elements are generated in this code, but I leave the last entry empty because I want to time how long it takes the function to run in other code, so I'm giving the array space to add the time if necessary
    data_array[:11] += np.array(params_correct)
    data_array[11:22] += np.array(params_guess)
    #Adding the recovered parameters and fixing the order
    data_array[22:26] += np.array(final_tri[6:10])
    data_array[26:29] += np.array(final_tri[3:6])
    data_array[29] += np.array(final_tri[-1])
    data_array[30:33] += np.array(final_tri[:3])
    #adding systematic error calculations
    syserr_result = syserr(data_array[22:33], data_array[:11])
    data_array[33:44] += np.array(syserr_result) #individual errors for each parameter
    data_array[-2] += np.sum(data_array[33:44]/10) #this is average error <e>... dividing by 10 (not 11) because we aren't counting the error in Force because the error is 0
    
    return data_array

#Runs a set number of trials for one system, graphs curvefit result,
# puts data and averages into spreadsheet, returns avg_e arrays and <e>_bar for all types of curves
def run_trials(true_params, guessed_params, freqs_NetMAP, freqs_curvefit, length_noise_NetMAP, length_noise_curvefit, num_trials, excel_file_name, graph_folder_name):

    #Needed for calculating e_bar and for graphing - also these are things that will be returned
    avg_e1_array = np.zeros(num_trials) #Polar
    avg_e2_array = np.zeros(num_trials) #Cartesian
    avg_e3_array = np.zeros(num_trials) #NetMAP
    
    #Needed to add all the data to a spreadsheet at the end
    all_data1 = np.empty((0, 52))  #Polar
    all_data2 = np.empty((0, 52))  #Cartesian
    all_data3 = np.empty((0, 46))  #NetMAP
    
    #FOR ONLY when I'm running 1 trial per system:
    # with pd.ExcelWriter(excel_file_name, engine='xlsxwriter') as writer:
    
    #Creating arrays to store the time it takes each curvefit/NetMAP function to run - will average them at the end
    times_polar = np.empty(num_trials)
    times_cartesian = np.empty(num_trials)
    times_NetMAP = np.empty(num_trials)
    
    #For more than 1 trial per system: 
    for i in range(num_trials):
        
        #Create noise  - noise level 2
        e = complex_noise(length_noise_curvefit, 2)
        
        ##For NetMAP
        #create noise - noise level 2
        e_NetMAP = complex_noise(length_noise_NetMAP,2)
        
        #Get the data!
        array1 = multiple_fit_amp_phase(guessed_params, true_params, e, freqs_curvefit, False, True, False, graph_folder_name, f'Polar_fig_{i}') #Polar, Fixed force
        array2 = multiple_fit_X_Y(guessed_params, true_params, e, freqs_curvefit, False, True, graph_folder_name, f'Cartesian_fig_{i}') #Cartesian, Fixed force
        array3 = get_parameters_NetMAP(freqs_NetMAP, guessed_params, true_params, e_NetMAP, False) #NetMAP
        
        #Time how long it takes to get the data and add the time to the larger array:
        #NOTE THAT - if you are outputting graphs within the curve fitting functions, the run time will be longer than it takes to get the actual data
        #that is, only use the timeit functions below when show_curvefit_graphs = False
        t_polar = timeit.timeit(lambda: multiple_fit_amp_phase(guessed_params, true_params, e, freqs_curvefit, False, True, False, graph_folder_name, f'Polar_fig_{i}'), number=1)
        times_polar[i] = t_polar
        t_cartesian = timeit.timeit(lambda: multiple_fit_X_Y(guessed_params, true_params, e, freqs_curvefit, False, True, graph_folder_name, f'Cartesian_fig_{i}'), number=1)
        times_cartesian[i] = t_cartesian
        t_NetMAP = timeit.timeit(lambda: get_parameters_NetMAP(freqs_NetMAP, guessed_params, true_params, e_NetMAP, False), number=1)
        times_NetMAP[i] = t_NetMAP
        
        #add each individual time to the array for each method so it can be stored with the data for each trial
        #array1, array2, array3 to be stacked into the larger all_data arrays
        array1[-1] = t_polar
        array2[-1] = t_cartesian
        array3[-1] = t_NetMAP
        
        #Pull out <e> (average across parameters) for each trial and add to arrays for e_bar calculation later
        #it is the second the last entry in the array (times is the last)
        avg_e1_array[i] += array1[-2]
        avg_e2_array[i] += array2[-2]
        avg_e3_array[i] += array3[-2]
        
        #Stack each trial's data to the larger array
        all_data1 = np.vstack((all_data1, array1))
        all_data2 = np.vstack((all_data2, array2))
        all_data3 = np.vstack((all_data3, array3))
        
    #Calculate average time it took for each method to recover parameters, along with standard deviation
    mean_time_polar = statistics.mean(times_polar)
    std_dev_polar = statistics.stdev(times_polar)
    mean_time_cartesian = statistics.mean(times_cartesian)
    std_dev_cartesian = statistics.stdev(times_cartesian)
    mean_time_NetMAP = statistics.mean(times_NetMAP)
    std_dev_NetMAP = statistics.stdev(times_NetMAP)
    
    #Calculate average error across parameters    
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
                   'Phase1_rsqrd', 'Phase2_rsqrd', 'Phase3_rsqrd', '<e>', 'trial time']
    
    #Turn the final data arrays into a dataframe so they can be written to excel
    dataframe_polar = pd.DataFrame(all_data1, columns=param_names)
    dataframe_cart = pd.DataFrame(all_data2, columns=param_names)
    dataframe_net = pd.DataFrame(all_data3, columns=param_names[:44] + param_names[-2:]) #cutting out the 6 r-squared columns because those values can only be found for the curvefits
    
    #Add <e>_bar values to data frame (one value for the whole system)
    dataframe_polar.at[0,'<e>_bar'] = avg_e1_bar
    dataframe_cart.at[0,'<e>_bar'] = avg_e2_bar
    dataframe_net.at[0,'<e>_bar'] = avg_e3_bar
    
    #Add the mean time and std dev to the data frame (one value each for the whole system)
    dataframe_polar.at[0,'mean trial time'] = mean_time_polar
    dataframe_polar.at[0,'std dev trial time'] = std_dev_polar
    dataframe_cart.at[0,'mean trial time'] = mean_time_cartesian
    dataframe_cart.at[0,'std dev trial time'] = std_dev_cartesian
    dataframe_net.at[0,'mean trial time'] = mean_time_NetMAP
    dataframe_net.at[0,'std dev trial time'] = std_dev_NetMAP
    
    
    #FOR ONLY when I'm running 1 trial per system:
    # dataframe_polar.to_excel(writer, sheet_name='Amp & Phase', index=False)
    # dataframe_cart.to_excel(writer, sheet_name='X & Y', index=False)
    # dataframe_net.to_excel(writer, sheet_name='NetMAP', index=False)
    
    # return avg_e1_array, avg_e2_array, avg_e3_array, avg_e1_bar, avg_e2_bar, avg_e3_bar
    
    #For more than 1 trial per system: 
    return avg_e1_array, avg_e2_array, avg_e3_array, avg_e1_bar, avg_e2_bar, avg_e3_bar, dataframe_polar, dataframe_cart, dataframe_net


''' Begin work here. Case Study. 
Randomly generate a system, then graph the data (no noise) and make a guess of parameters based on visual accuracy of the curve.
Use this guess to curvefit to the data. NetMAP does not require this initial guess to function.'''

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
# freqs_curvefit = np.linspace(0.001, 4, 10)
# length_noise_NetMAP = 10
# length_noise_curvefit = 10
# avg_e1_array, avg_e2_array, avg_e3_array, avg_e1_bar, avg_e2_bar, avg_e3_bar = run_trials(true_params, guessed_params, freqs_NetMAP, freqs_curvefit, length_noise_NetMAP, length_noise_curvefit 10, 'Case_Study.xlsx', 'Case Study Plots')

# #Graph histogram of <e> for curve fits

# plt.title('Average Systematic Error Across Parameters')
# plt.xlabel('<e>')
# plt.ylabel('Counts')
# plt.hist(avg_e2_array, alpha=0.5, color='green', label='Cartesian (X & Y)', edgecolor='black')
# plt.hist(avg_e1_array, alpha=0.5, color='blue', label='Polar (Amp & Phase)', edgecolor='black')
# plt.hist(avg_e3_array, bins=50, alpha=0.5, color='red', label='NetMAP', edgecolor='black')
# plt.legend(loc='upper center')

# plt.show()

''' Begin work here. Automated guesses. Multiple systems.
Instead of manually guessing the intial parameters, guess is generated to be within a certain percentage of the true parameters.
Error across trials and across parameters is calculated. Error across parameters is graphed (e_bar) at the end to visualize error for all the systems on one graph.'''

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

''' Begin work here. Checking Worst System - System 0 from 15 Systems - 10 Freqs NetMAP.
Running the system with no noise to understand why recovered error was so bad. 
'''

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
'''Additionally, I am going to use the same frequencies for all three methods of parameter recovery: 
   300 or 10 evenly spaced frequencies from 0.001 to 4.'''
''' Note that all information saves to the same folder that this code is located in.'''

# #Recover the system information from a file on my computer
# file_path = '/Users/Student/Desktop/Summer Research 2024/Curve Fit vs NetMAP/Case Study - 10 Freqs NetMAP & Better Parameters/Case_Study_10_Freqs_Better_Parameters.xlsx'   
# array_amp_phase = pd.read_excel(file_path, sheet_name = 'Amp & Phase').to_numpy()
# array_X_Y = pd.read_excel(file_path, sheet_name = 'X & Y').to_numpy()

# #These are the true and the guessed parameters for the system
# #Guessed parameters were the same ones guesssed by hand the first time we ran this case study
# true_params = np.concatenate((array_amp_phase[1,:7], [array_amp_phase[1,10]], array_amp_phase[1,7:10]))
# guessed_params = np.concatenate((array_amp_phase[1,11:18], [array_amp_phase[1,21]], array_amp_phase[1,18:21]))

# #Create the frequencies that both NetMAP and the Curvefitting functions require
# #Note that if the number of frequencies are not the same, the noise must be adjusted
# # freq_curvefit = np.linspace(0.001, 4, 300)
# freq_curvefit = np.linspace(0.001, 4, 10)
# freqs_NetMAP = np.linspace(0.001, 4, 10)
# length_noise_curvefit = 10
# length_noise_NetMAP = 10

# #Run the trials (1000 in this case) 
# #Currently saves saves all plots to a folder called "Case Study 1000 Trials Same Frequencies Plots" 
# #(the excel name is not used here - it is only required when doing multiple systems with one trial per system)
# #returns average error across trials (e_bar) and parameters (e), and dataframes for all three methods that include all the information 
# #there is only one e_bar for each when doing a case study, so it will not be used
# #NOTE: error is different every time, to simulate a real experiment
# avg_e1_array, avg_e2_array, avg_e3_array, avg_e1_bar, avg_e2_bar, avg_e3_bar, dataframe_polar, dataframe_cart, dataframe_net = run_trials(true_params, guessed_params, freqs_NetMAP, freq_curvefit, length_noise_NetMAP, length_noise_curvefit, 1000, 'Second_Case_Study_1000_Trials_10_Frequencies.xlsx', 'Second Case Study 1000 Trials 10 Frequencies Plots')

# #Save the new data to a new excel spreadsheet:
# with pd.ExcelWriter('Second_Case_Study_1000_Trials_10_Frequencies.xlsx', engine='xlsxwriter') as writer:
#     dataframe_polar.to_excel(writer, sheet_name='Amp & Phase', index=False)
#     dataframe_cart.to_excel(writer, sheet_name='X & Y', index=False)
#     dataframe_net.to_excel(writer, sheet_name='NetMAP', index=False)

# #Graph lin and log histograms of <e> for both curve fits:

# #Compute max of data and set the bin limits so all data is seen/included on graph
# data_max = max(avg_e1_array + avg_e2_array + avg_e3_array)
# if data_max > 39: 
#     linearbins = np.linspace(0, data_max + 2,50)
# else:
#     linearbins = np.linspace(0, 40, 50)

# #Graph linear plots
# fig = plt.figure(figsize=(5, 4))
# plt.xlabel('<e> Bar (%)', fontsize = 16)
# plt.ylabel('Counts', fontsize = 16)
# plt.yticks(fontsize=14)
# plt.xticks(fontsize=14)
# plt.hist(avg_e1_array, bins = linearbins, alpha=0.5, color='blue', label='Polar', edgecolor='blue')
# plt.hist(avg_e2_array, bins = linearbins, alpha=0.5, color='green', label='Cartesian', edgecolor='green')
# plt.hist(avg_e3_array, bins = linearbins, alpha=0.5, color='red', label='NetMAP', edgecolor='red')
# plt.legend(loc='best', fontsize = 13)

# plt.show()
# save_figure(fig, 'Second Case Study 1000 Trials 10 Frequencies', 'Linear <e> Histogram')

# # Set the bin limits so all data is seen/included on graph
# if data_max > 100: 
#     logbins = np.logspace(-2, math.log10(data_max)+0.25, 50)
# else:
#     logbins = np.logspace(-2, 1.8, 50)

# #Graph log!
# fig = plt.figure(figsize=(5, 4))
# plt.xlabel('<e> Bar (%)', fontsize = 16)
# plt.ylabel('Counts', fontsize = 16)
# plt.xscale('log')
# plt.yticks(fontsize=14)
# plt.xticks(fontsize=14)
# plt.hist(avg_e1_array, bins = logbins, alpha=0.5, color='blue', label='Polar', edgecolor='blue')
# plt.hist(avg_e2_array,  bins = logbins, alpha=0.5, color='green', label='Cartesian', edgecolor='green')
# plt.hist(avg_e3_array, bins = logbins, alpha=0.5, color='red', label='NetMAP', edgecolor='red')
# plt.legend(loc='best', fontsize = 13)

# plt.show()
# save_figure(fig, 'Second Case Study 1000 Trials 10 Frequencies', 'Logarithmic <e> Histogram')


'''Begin work here. Case Study - 10 Freqs Better Params with 1000 trials
    GOAL: graph runtime versus number of frequencies given to each method.
    Create a for loop that varies frequencies from 2 to 300. (2 because that is the minimum required by NetMAP. 300 because that produces a very nice graph for curvefitting (and is what I have been using as a standard up until now.'''

# #Recover the system information from a file on my computer
# file_path = '/Users/Student/Desktop/Summer Research 2024/Curve Fit vs NetMAP/Case Study - 10 Freqs NetMAP & Better Parameters/Case_Study_10_Freqs_Better_Parameters.xlsx'   
# array_amp_phase = pd.read_excel(file_path, sheet_name = 'Amp & Phase').to_numpy()

# #These are the true and the guessed parameters for the system
# #Guessed parameters were the same ones guesssed by hand the first time we ran this case study
# true_params = np.concatenate((array_amp_phase[1,:7], [array_amp_phase[1,10]], array_amp_phase[1,7:10]))
# guessed_params = np.concatenate((array_amp_phase[1,11:18], [array_amp_phase[1,21]], array_amp_phase[1,18:21]))

# #create array to store the run times for the given number of frequencies
# #there will be a total of 98 different times since we start with 2 frequencies and end with 100 
# run_times_polar = np.zeros(99)
# run_times_cartesian = np.zeros(99)
# run_times_NetMAP = np.zeros(99)

# #used for graphing (below for loop) 
# num_freq = np.arange(2,101,1) #arange does not include the "stop" number, so the array goes from 2 to 100

# #loop to change which frequency is used to recover parameters
# for i in range(0,99): #range does not include the "stop" number, so the index actually goes up to 98
#     #Create the frequencies that both NetMAP and the Curvefitting functions require
#     #Frequencies are values between 0.001 and 4, evenly spaced depending on how many frequencies we use
#     #Note that the number of frequencies must match the length of the noise
#     #minimum 2 frequencies required - max of 300 because that how high I was going before (gives a very good curve for curvefit)
#     freq_curvefit = np.linspace(0.001, 4, i+2)
#     freqs_NetMAP = np.linspace(0.001, 4, i+2)
#     length_noise_curvefit = i+2
#     length_noise_NetMAP = i+2

#     #Run the trials (1000 in this case) 
#     #Currently saves saves all plots to a folder called "Case Study 1000 Trials Varying Frequencies Plots" 
#     #(the excel name is not used here - it is only required when doing multiple systems with one trial per system)
#     #returns average error across trials (e_bar) and parameters (e), and dataframes for all three methods that include all the information 
#     #there is only one e_bar for each when doing a case study, so those arrays will not be used in any graphing moving forward
#     #NOTE: error is different every time, to simulate a real experiment
#     avg_e1_array, avg_e2_array, avg_e3_array, avg_e1_bar, avg_e2_bar, avg_e3_bar, dataframe_polar, dataframe_cart, dataframe_net = run_trials(true_params, guessed_params, freqs_NetMAP, freq_curvefit, length_noise_NetMAP, length_noise_curvefit, 50, f'Second_Case_Study_50_Trials_{i+2}_Frequencies.xlsx', f'Second Case Study 50 Trials {i+2} Frequencies Plots')

#     #Save the new data to a new excel spreadsheet:
#     with pd.ExcelWriter(f'Case_Study_50_Trials_{i+2}_Frequencies.xlsx', engine='xlsxwriter') as writer:
#         dataframe_polar.to_excel(writer, sheet_name='Amp & Phase', index=False)
#         dataframe_cart.to_excel(writer, sheet_name='X & Y', index=False)
#         dataframe_net.to_excel(writer, sheet_name='NetMAP', index=False)
        
#     #The run times are stored in the dataframes, so we extract the mean here and add it to the run_times arrays so we can graph it later
#     run_times_polar[i] = dataframe_polar.at[0,'mean trial time']
#     run_times_cartesian[i] = dataframe_cart.at[0,'mean trial time']
#     run_times_NetMAP[i] = dataframe_net .at[0,'mean trial time']
    
#     print(f"Frequency {i+2} Complete")

''' Graphing the above didn't work, so I'm doing it again below '''

run_times_polar = np.zeros(99)
run_times_cartesian = np.zeros(99)
run_times_NetMAP = np.zeros(99)
std_dev_time_polar = np.zeros(99)
std_dev_time_cartesian = np.zeros(99)
std_dev_time_NetMAP = np.zeros(99)
num_freq = np.arange(2,101,1)

for i in range(99):
    file_path = f'/Users/Student/Desktop/Summer Research 2024/Curve Fit vs NetMAP/Case Study -  Number of Frequencies vs Average Run Time/50 Trials/Case_Study_50_Trials_{i+2}_Frequencies.xlsx'   
    polar = pd.read_excel(file_path, sheet_name = 'Amp & Phase').to_numpy()
    cartesian = pd.read_excel(file_path, sheet_name = 'X & Y').to_numpy()
    NetMAP = pd.read_excel(file_path, sheet_name = 'NetMAP').to_numpy()
    
    run_times_polar[i] = polar[0,53]
    run_times_cartesian[i] = cartesian[0,53]
    run_times_NetMAP[i] = NetMAP[0,47]
    std_dev_time_polar[i] = polar[0,54]
    std_dev_time_cartesian[i] = cartesian[0,54]
    std_dev_time_NetMAP[i] = NetMAP[0,48]

    
#Plot number of frequencies versus run time: 
fig = plt.figure(figsize=(5, 4))
plt.xlabel('Number of Frequencies', fontsize = 16)
plt.ylabel('Mean Time to Run (s)', fontsize = 16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.yscale('log')
plt.plot(num_freq, run_times_polar, 'o', color='blue', label='Polar')
plt.plot(num_freq, run_times_cartesian, 'o', color='green', label='Cartesian')
plt.plot(num_freq, run_times_NetMAP, 'o', color='red', label='NetMAP')
plt.legend(loc='best', fontsize = 13)
plt.show()


fig = plt.figure(figsize=(5, 4))
plt.xlabel('Number of Frequencies', fontsize = 16)
plt.ylabel('Mean Time to Run (s)', fontsize = 16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.yscale('log')
plt.plot(num_freq, run_times_polar, 'o', color='blue', label='Polar')
plt.legend(loc='best', fontsize = 13)
plt.show()

fig = plt.figure(figsize=(5, 4))
plt.xlabel('Number of Frequencies', fontsize = 16)
plt.ylabel('Mean Time to Run (s)', fontsize = 16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.yscale('log')
plt.plot(num_freq, run_times_cartesian, 'o', color='green', label='Cartesian')
plt.legend(loc='best', fontsize = 13)
plt.show()

fig = plt.figure(figsize=(5, 4))
plt.xlabel('Number of Frequencies', fontsize = 16)
plt.ylabel('Mean Time to Run (s)', fontsize = 16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.plot(num_freq, run_times_NetMAP, 'o', color='red', label='NetMAP')
plt.legend(loc='best', fontsize = 13)
plt.show()


# polar_outliers = run_times_polar[run_times_polar > 20]
# cartesian_outliers = run_times_cartesian[run_times_cartesian > 20]
# polar_outlier_indices = np.nonzero(run_times_polar > 20)
# cartesian_outlier_indices = np.nonzero(run_times_cartesian > 20)

# no_outliers_polar_times = np.empty
# no_outliers_cartesian_times = np.empty
# new_freq_polar = np.empty

# for i in range(len(run_times_polar)):
#     if run_times_polar[i] not in polar_outliers:
#         no_outliers_polar_times[i] = run_times_polar[i]
#     if run_times_cartesian[i] not in cartesian_outliers:
#         no_outliers_cartesian_times[i] = run_times_cartesian[i]
#     if i not in polar_outlier_indices:
        
    
        

'''Begin work here. Redoing 15 systems data. Still using 10 Freqs and Better Params.
   I want to run parameter recovery for many more systems but only 1 trial per system. 
   Seeing how many systems it can do in 2 hours or 2000 systems.'''


## 1. What am I doing for error? 
    ## 300 frequencies (n=300 -- so 300 different noises for each frequency used) and noise level 2
    ## 10 evenly spaced frequencies for NetMAP (n=10) and noise level 2. 
## 2. Set a runtime limit of 2-3 hours. DONE
## 3. Don't graph all the curvefits. DONE
## 4. Guesses are automated to within 20% of generated parameters, 10 evenly spaced frequencies for NetMAP 


# # Set the time limit in seconds
# time_limit = 14400  # 4 hours

# # Record the start time
# start_time = time.time()

# # Compile a list of all the e bars so we can graph at the end
# avg_e_bar_list_polar = [] 
# avg_e_bar_list_cartesian = []
# avg_e_bar_list_NetMAP = []

# # Initialize an array so I can put each system into one spreadsheet since I'm only doing one trial per system
# all_data1 = pd.DataFrame()  #Polar
# all_data2 = pd.DataFrame()  #Cartesian
# all_data3 = pd.DataFrame()  #NetMAP

# for i in range(2000):
    
#     # Check if the time limit has been exceeded
#     elapsed_time = time.time() - start_time
#     if elapsed_time > time_limit:
#         print("Time limit exceeded. Exiting loop.")
#         break
    
#     loop_start_time = time.time()
    
#     #Generate system and guess parameters
#     true_params = generate_random_system()
#     guessed_params = automate_guess(true_params, 20)
    
#     #Curve fit with the guess made above
#     freqs_NetMAP = np.linspace(0.001, 4, 10)
#     length_noise = 10
#     avg_e1_array, avg_e2_array, avg_e3_array, avg_e1_bar, avg_e2_bar, avg_e3_bar, dataframe_polar, dataframe_cart, dataframe_net = run_trials(true_params, guessed_params, freqs_NetMAP, length_noise, 1, f'System_{i+1}_1.xlsx', f'Sys {i+1} - Rand Auto Guess Plots')
    
#     #Add each system data to one big dataframe so I can store everything in the same spreadsheet

#     all_data1 = pd.concat([all_data1, dataframe_polar], ignore_index=True)
#     all_data2 = pd.concat([all_data2, dataframe_cart], ignore_index=True)
#     all_data3 = pd.concat([all_data3, dataframe_net], ignore_index=True)
    
#     #Add <e>_bar to lists to make one graph at the end
#     avg_e_bar_list_polar.append(avg_e1_bar) #Polar
#     avg_e_bar_list_cartesian.append(avg_e2_bar) #Cartesian
#     avg_e_bar_list_NetMAP.append(avg_e3_bar) #NetMAP
    
#     ## FOR NOW - don't need this either
    
#     # # Compute max of data and set the bin limits so all data is included on graph
#     # data_max1 = max(avg_e2_array + avg_e1_array + avg_e3_array)
#     # if data_max1 > 39:
#     #     linearbins = np.linspace(0, data_max1 + 2,50)
#     # else: 
#     #     linearbins = np.linspace(0, 40, 50)

#     # #Graph histogram of <e> for curve fits - linear
#     # fig = plt.figure(figsize=(5, 4))
#     # # plt.title('Average Systematic Error Across Parameters')
#     # plt.xlabel('<e> (%)', fontsize = 16)
#     # plt.ylabel('Counts', fontsize = 16)
#     # plt.yticks(fontsize=14)
#     # plt.xticks(fontsize=14)
#     # plt.hist(avg_e2_array, bins = linearbins, alpha=0.5, color='green', label='Cartesian (X & Y)', edgecolor='green')
#     # plt.hist(avg_e1_array, bins = linearbins, alpha=0.5, color='blue', label='Polar (Amp & Phase)', edgecolor='blue')
#     # plt.hist(avg_e3_array, bins = linearbins, alpha=0.5, color='red', label='NetMAP', edgecolor='red')
#     # plt.legend(loc='best', fontsize = 13)

#     # # plt.show()
#     # save_figure(fig, 'More Systems 1 Trial - <e> Histograms', f'<e> Lin Hist System {i+1}')
   
#     # # Set the bin limits so all data is included on graph
#     # if data_max > 100: 
#     #     logbins = np.logspace(-2, math.log10(data_max), 50)
#     # else:
#     #     logbins = np.logspace(-2, 1.8, 50)
#     # #Graph histogram of <e> for curve fits - log
#     # fig = plt.figure(figsize=(5, 4))
#     # # plt.title('Average Systematic Error Across Parameters')
#     # plt.xlabel('<e> (%)', fontsize = 16)
#     # plt.ylabel('Counts', fontsize = 16)
#     # plt.xscale('log')
#     # plt.yticks(fontsize=14)
#     # plt.xticks(fontsize=14)
#     # plt.hist(avg_e2_array, bins = logbins, alpha=0.5, color='green', label='Cartesian (X & Y)', edgecolor='green')
#     # plt.hist(avg_e1_array, bins = logbins, alpha=0.5, color='blue', label='Polar (Amp & Phase)', edgecolor='blue')
#     # plt.hist(avg_e3_array, bins = logbins, alpha=0.5, color='red', label='NetMAP', edgecolor='red')
#     # plt.legend(loc='best', fontsize = 13)

#     # # plt.show()
#     # save_figure(fig, 'More Systems 1 Trial - <e> Histograms', f'<e> Log Hist System {i+1}')
    
#     loop_end_time = time.time()
#     loop_time = loop_end_time - loop_start_time
    
#     print(f"Iteration {i + 1} completed. Loop time: {loop_time} secs ")
    
# #Write the data for each system (which is now in one big dataframe) to excel
# with pd.ExcelWriter('All_Systems_1_Trial_2.xlsx') as writer:
#     all_data1.to_excel(writer, sheet_name='Polar', index=False)
#     all_data2.to_excel(writer, sheet_name='Cartesian', index=False)
#     all_data3.to_excel(writer, sheet_name='NetMAP', index=False)


# #Graph histogram of <e>_bar for both curve fits

# # Compute max of data and set the bin limits so all data is included on graph
# data_max = max(avg_e_bar_list_cartesian + avg_e_bar_list_polar + avg_e_bar_list_NetMAP)
# if data_max > 39: 
#     linearbins = np.linspace(0, data_max + 2,50)
# else:
#     linearbins = np.linspace(0, 40, 50)

# #Graph linear!
# fig = plt.figure(figsize=(5, 4))
# plt.xlabel('<e> Bar (%)', fontsize = 16)
# plt.ylabel('Counts', fontsize = 16)
# plt.yticks(fontsize=14)
# plt.xticks(fontsize=14)
# plt.hist(avg_e_bar_list_cartesian, bins = linearbins, alpha=0.5, color='green', label='Cartesian', edgecolor='green')
# plt.hist(avg_e_bar_list_polar, bins = linearbins, alpha=0.5, color='blue', label='Polar', edgecolor='blue')
# plt.hist(avg_e_bar_list_NetMAP, bins = linearbins, alpha=0.5, color='red', label='NetMAP', edgecolor='red')
# plt.legend(loc='best', fontsize = 13)

# plt.show()
# save_figure(fig, 'More Systems 1 Trial - <e> Histograms', '<e> Bar Lin Hist 2' )

# # Set the bin limits so all data is included on graph
# if data_max > 100: 
#     logbins = np.logspace(-2, math.log10(data_max)+0.25, 50)
# else:
#     logbins = np.logspace(-2, 1.8, 50)

# #Graph log!
# fig = plt.figure(figsize=(5, 4))
# plt.xlabel('<e> Bar (%)', fontsize = 16)
# plt.ylabel('Counts', fontsize = 16)
# plt.xscale('log')
# plt.yticks(fontsize=14)
# plt.xticks(fontsize=14)
# plt.hist(avg_e_bar_list_cartesian,  bins = logbins, alpha=0.5, color='green', label='Cartesian', edgecolor='green')
# plt.hist(avg_e_bar_list_polar, bins = logbins, alpha=0.5, color='blue', label='Polar', edgecolor='blue')
# plt.hist(avg_e_bar_list_NetMAP, bins = logbins, alpha=0.5, color='red', label='NetMAP', edgecolor='red')
# plt.legend(loc='best', fontsize = 13)

# plt.show()
# save_figure(fig, 'More Systems 1 Trial - <e> Histograms', '<e> Bar Log Hist 2' )

# # End time
# end_time = time.time()
# print(f"Time Elapsed: {end_time - start_time} secs -- {(end_time - start_time)/3600} hrs")

