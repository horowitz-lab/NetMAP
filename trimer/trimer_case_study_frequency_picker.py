#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:55:10 2024

@author: lydiabullock
"""
''' Case Study for System 0/2 from '15 Systems - 10 Freqs NetMAP' 
    Using "ideal" frequencies to test NetMAP. 
    The frequencies picked are only based off of the first two resonators but use the trimer information.'''

from comparing_curvefit_types import run_trials
from kind_of_trimer_resonatorfrequencypicker import res_freq_numeric
import math
import matplotlib.pyplot as plt

''' Begin Work Here. '''

MONOMER = False
forceall = False

## System 0 from '15 Systems - 10 Freqs NetMAP'
# true_parameters = [1.045, 0.179, 3.852, 1.877, 5.542, 1.956, 3.71, 1, 3.976, 0.656, 3.198]
# guessed_parameters = [1.2379, 0.1764, 3.7327, 1.8628, 5.93, 2.1793, 4.2198, 1, 4.3335, 0.7016, 3.0719]

## System 2 from '15 Systems - 10 Freqs NetMAP'
# true_parameters = [3.264, 7.71, 6.281, 3.564, 5.859, 0.723, 3.087, 1, 3.391, 3.059, 7.796]
# guessed_parameters = [3.1169, 7.0514, 6.9721, 3.6863, 4.9006, 0.707, 3.2658, 1, 2.9289, 2.7856, 6.8323]

## System 8 from '15 systems - 10 Freqs NetMAP & Better Parameters'
true_parameters = [7.731, 1.693, 2.051, 8.091, 0.427, 0.363, 0.349, 1, 7.07, 7.195, 4.814]
guessed_parameters = [7.2806, 1.8748, 1.8077, 8.7478, 0.3767, 0.2974, 0.3744, 1, 7.4933, 6.7781, 4.2136]

best_frequencies_list = res_freq_numeric(true_parameters, MONOMER, forceall)
best_frequencies_list = [x for x in best_frequencies_list if not math.isnan(x)]
length_noise_NetMAP = len(best_frequencies_list)

#Run Trials
if length_noise_NetMAP == 0:
    print('No Possible Frequencies.')
else:
    print(f'Best frequencies to use are: {best_frequencies_list}')
    avg_e1_list, avg_e2_list, avg_e3_list, avg_e1_bar, avg_e2_bar, avg_e3_bar = run_trials(true_parameters, guessed_parameters, best_frequencies_list, length_noise_NetMAP, 50, 'Sys8_Better_Params_Freq_Pick.xlsx', 'Sys8_Better_Params_Freq_Pick - Plots')
    
    #Create histogram
    plt.title('Average Systematic Error Across Parameters')
    plt.xlabel('<e>')
    plt.ylabel('Counts')
    plt.hist(avg_e2_list, alpha=0.5, color='green', label='Cartesian (X & Y)', edgecolor='black')
    plt.hist(avg_e1_list, alpha=0.5, color='blue', label='Polar (Amp & Phase)', edgecolor='black')
    plt.hist(avg_e3_list, bins=50, alpha=0.5, color='red', label='NetMAP', edgecolor='black')
    plt.legend(loc='upper center')
    
    plt.savefig('<e>_Histogram_Sys8_Better_Params_Freq_Pick.png') 

