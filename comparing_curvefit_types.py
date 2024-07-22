#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:42:41 2024

@author: lydiabullock
"""

''' Which has more accurated recovered parameters: Amp & Phase or X & Y? 
    Same system, different noise - this replicates doing experiment many times.
    Using method of fixing F. '''

import pandas as pd
import math
import matplotlib.pyplot as plt
from curve_fitting_amp_phase_all import multiple_fit_amp_phase
from curve_fitting_X_Y_all import multiple_fit_X_Y
from resonatorsimulator import complex_noise

''' Functions contained:
    find_avg_e - calculates average across systematic error for each parameter
                 for one trial of the same system
    artithmetic_then_logarithmic - calculates arithmetic average across parameters first, 
                                   then logarithmic average across trials 
    run_trials - Runs a set number of trials for one system, graphs curvefit result,
                 puts data and averages into spreadsheet, returns <e>_bar for both types of curves
               - Must include number of trials to run and name of excel sheet  
'''

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
def arithmetic_then_logarithmic(avg_e_list):
    ln_avg_e = []
    for item in avg_e_list:
        ln_avg_e.append(math.log(item))
    sum_ln_avg_e = sum(ln_avg_e)
    e_raised_to_sum = math.exp(sum_ln_avg_e)
    return e_raised_to_sum

#Runs a set number of trials for one system, graphs curvefit result,
# puts data and averages into spreadsheet, returns <e>_bar for both types of curves
def run_trials(true_params, guessed_params, num_trials, file_name):
    
    starting_row = 0
    avg_e1_list = []
    avg_e2_list = []
    
    #Put data into excel spreadsheet
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        for i in range(num_trials):
            
            #Create noise
            e = complex_noise(300, 2)
        
            #Get the data!
            dictionary1 = multiple_fit_amp_phase(guessed_params, true_params, e, False, True) #Polar, Fixed force
            dictionary2 = multiple_fit_X_Y(guessed_params, true_params, e, False, True) #Cartesian, Fixed force
        
            #Find <e> (average across parameters) for each trial and add to dictionary
            avg_e1 = find_avg_e(dictionary1)
            dictionary1['<e>'] = avg_e1
            
            avg_e2 = find_avg_e(dictionary2)
            dictionary2['<e>'] = avg_e2
            
            #Append to list for later graphing
            avg_e1_list.append(avg_e1)
            avg_e2_list.append(avg_e2)
        
            #Turn data into dataframe for excel
            dataframe1 = pd.DataFrame(dictionary1)
            dataframe2 = pd.DataFrame(dictionary2)
            
            #Add to excel spreadsheet
            dataframe1.to_excel(writer, sheet_name='Amp & Phase', startrow=starting_row, index=False, header=(i==0))
            dataframe2.to_excel(writer, sheet_name='X & Y', startrow=starting_row, index=False, header=(i==0))
        
            starting_row += len(dataframe1) + (1 if i==0 else 0)
        
        avg_e1_bar = arithmetic_then_logarithmic(avg_e1_list) 
        avg_e2_bar = arithmetic_then_logarithmic(avg_e2_list)
        
        dataframe1.at[0,'<e>_bar'] = avg_e1_bar
        dataframe2.at[0,'<e>_bar'] = avg_e2_bar
        
        dataframe1.to_excel(writer, sheet_name='Amp & Phase', index=False)
        dataframe2.to_excel(writer, sheet_name='X & Y', index=False)
        
        return avg_e1_list, avg_e2_list, arithmetic_then_logarithmic(avg_e1_list), arithmetic_then_logarithmic(avg_e2_list)
    

''' Begin work here. '''

#Make parameters/initial guesses - [k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3]
#Note that right now we only scale/fix by F, so make sure to keep F correct in guesses
sys5_true_params = [5, 5, 1, 1, 2, 2, 2, 1, 1.5, 1.5, 6.5]
sys5_guessed_params = [5.23, 4.5, 1.39, 0.47, 1.983, 2.01, 2.76, 1, 2.025, 1.7, 5.739]

sys5_avg_e1_list, sys5_avg_e2_list, sys5_avg_e1_bar, sys5_avg_e2_bar = run_trials(sys5_true_params, sys5_guessed_params, 50, 'System_5.xlsx')

plt.hist(sys5_avg_e1_list, bins=30, alpha=0.75, color='blue', edgecolor='black')
plt.hist(sys5_avg_e2_list, bins=30, alpha=0.75, color='green', edgecolor='black')




