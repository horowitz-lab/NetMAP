#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:43:54 2024

@author: lydiabullock
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
import random
import pandas as pd
from Trimer_simulator import t1, t2, t3

#A list to store the dataframe for each of the three masses in the trimer system
all_data = []

#Run the following for the first, second, and third masses
for mass in ['Mass 1', 'Mass 2', 'Mass 3']:
    
    #type of function to fit for all three amplitude curves
    #t = phase 
    def phase_function(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3):
        if mass == 'Mass 1':
            return t1(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
        elif mass == 'Mass 2':
            return t2(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
        else:
            return t3(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
        
    #create data for graph
    freq = np.linspace(0.001, 5, 100)
    phase = phase_function(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5)
    
    #create dictionary for storing data
    data = {'m1_guess': [], 'm2_guess': [], 'm3_guess': [], 
            'b1_guess': [], 'b2_guess': [], 'b3_guess': [],
            'k1_guess': [], 'k2_guess': [], 'k3_guess': [],  'k4_guess': [], 
            'F_guess': [],
            'm1_recovered': [], 'm2_recovered': [], 'm3_recovered': [], 
            'b1_recovered': [], 'b2_recovered': [], 'b3_recovered': [], 
            'k1_recovered': [], 'k2_recovered': [], 'k3_recovered': [], 'k4_recovered': [], 
            'F_recovered': [], 
            'r_squared': []}
    
    #correct parameters
    parameters = [3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5]
    
    #define number of times we generate random guesses and get the data
    num = 50
    
    #create funtion that generates a random number in intervals of 0.5
    def random_num(start, end, interval):
        num = random.uniform(start, end)
        rounded_num = round(num / interval) * interval # Round the number to the nearest interval
        return rounded_num
    
    
    #generate random initial guess, curve fit, and calculate R^2
    for trial in range(num):
        try:
            #create the random guess that's within 2 units of correct parameters
            random_initial_guess = [param + random_num(-2,3, 0.5) for param in parameters]
            #add guesses to dictionary
            data['k1_guess'].append(random_initial_guess[0])
            data['k2_guess'].append(random_initial_guess[1])
            data['k3_guess'].append(random_initial_guess[2])
            data['k4_guess'].append(random_initial_guess[3])
            data['b1_guess'].append(random_initial_guess[4])
            data['b2_guess'].append(random_initial_guess[5])
            data['b3_guess'].append(random_initial_guess[6])
            data['F_guess'].append(random_initial_guess[7])
            data['m1_guess'].append(random_initial_guess[8])
            data['m2_guess'].append(random_initial_guess[9])
            data['m3_guess'].append(random_initial_guess[10])
            
            #curve fitting
            model = Model(phase_function)
            params = model.make_params(k1=random_initial_guess[0],k2=random_initial_guess[1],k3=random_initial_guess[2],
                                       k4=random_initial_guess[3],b1=random_initial_guess[4],b2=random_initial_guess[5],
                                       b3=random_initial_guess[6],F=random_initial_guess[7],m1=random_initial_guess[8],
                                       m2=random_initial_guess[9],m3=random_initial_guess[10])
            result = model.fit(phase, params, w=freq)
            
            #add recovered parameters to dictionary
            fit_params = result.params
            
            for param_name in ['m1', 'm2', 'm3', 'b1', 'b2', 'b3', 'k1', 'k2', 'k3', 'k4', 'F']:
                param_value = result.params[param_name].value
                data[f'{param_name}_recovered'].append(param_value)
            
            
            # Extracting the R-squared value
            r_squared = 1 - result.residual.var() / np.var(phase)
            #add r_squared to dictionary
            data['r_squared'].append(r_squared)
            
            #Graph!
            #graph original data
            plt.figure(figsize=(8,6))
            plt.plot(freq, phase, 'bo', label='Original Data')
            
            #generate points for fitted curve
            freq_fit = np.linspace(min(freq),max(freq), 500) #more w-values than before
            phase_fit = result.model.func(freq_fit, **result.best_values)
            
            #graph fitted curve
            plt.plot(freq_fit, phase_fit, '-', label='Fitted Curve')
            
            #graph parts
            plt.legend(loc='best')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Phase (Rad)')
            plt.title(f'Trimer Curve Fitting for {mass} - #{trial+1}')
            
            plt.savefig(f'/Users/Student/Desktop/Summer Research 2024/GitHub/NetMAP/Generating Random Params - Curve Fit/{mass} plots - phase/plot_{trial+1}.png')
            plt.show()
                
        except RuntimeError:
            #If it takes too long, appends 0s to the dictionary
            #R^2 = 0 means parameters not recovered
            #Sam did this - I've run it a couple times and I haven't had any 0s yet
            data['k1_guess'].append(0)
            data['k2_guess'].append(0)
            data['k3_guess'].append(0)
            data['k4_guess'].append(0)
            data['b1_guess'].append(0)
            data['b2_guess'].append(0)
            data['b3_guess'].append(0)
            data['F_guess'].append(0)
            data['m1_guess'].append(0)
            data['m2_guess'].append(0)
            data['m3_guess'].append(0)
            data['k1_recovered'].append(0)
            data['k2_recovered'].append(0)
            data['k3_recovered'].append(0)
            data['k4_recovered'].append(0)
            data['b1_recovered'].append(0)
            data['b2_recovered'].append(0)
            data['b3_recovered'].append(0)
            data['F_recovered'].append(0)
            data['m1_recovered'].append(0)
            data['m2_recovered'].append(0)
            data['m3_recovered'].append(0)
            data['r_squared'].append(0)
            
    #put all this data in a spreadsheet!
    df = pd.DataFrame(data)
    all_data.append(df)
    
file_path = '/Users/Student/Desktop/Summer Research 2024/GitHub/NetMAP/Generating Random Params - Curve Fit/Generating_Random_Params_Phase.xlsx'
#write each DataFrame to a specific sheet
with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
    all_data[0].to_excel(writer, sheet_name='M1', index=False)
    all_data[1].to_excel(writer, sheet_name='M2', index=False)
    all_data[2].to_excel(writer, sheet_name='M3', index=False)

    










