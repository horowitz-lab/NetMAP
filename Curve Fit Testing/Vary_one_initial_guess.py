#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:01:48 2024

@author: lydiabullock
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
import pandas as pd
from Trimer_simulator import c1, c2, c3, t1, t2, t3, re1, re2, re3, im1, im2, im3

''' Function below varies one initial guess parameter at a time for the mass of choice (1,2,3) of a trimer system.
Curve fits for dependent variable of choice (amp, phase, real, im) versus frequency.
Takes a number (1,2,3) for the mass you want to analyze
Takes a string (amp, phase, real, im) for the dependent variable of your data (independent variable is frequency)
'''

def vary_one_initial_guess(which_mass, which_graph):

    #list to store the dataframe for each time we change a different parameter
    all_data = []
    
    for param in ['m1', 'm2', 'm3', 'b1', 'b2', 'b3', 'k1', 'k2', 'k3', 'k4', 'F']:
        
        #type of function to fit for  curves
        #c = amplitude
        #t = phase
        #re = real part
        #im = imaginary part
        def function(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3):
            if which_mass == 1:
                if which_graph == 'amp':
                    return c1(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
                elif which_graph == 'phase':
                    return t1(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
                elif which_graph == 'real':
                    return re1(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
                else: 
                    return im1(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
            if which_mass == 2:
                if which_graph == 'amp':
                    return c2(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
                elif which_graph == 'phase':
                    return t2(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
                elif which_graph == 'real':
                    return re2(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
                else: 
                    return im2(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
            if which_mass == 3:
                if which_graph == 'amp':
                    return c3(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
                elif which_graph == 'phase':
                    return t3(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
                elif which_graph == 'real':
                    return re3(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
                else: 
                    return im3(w, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
            
        #create data for graph
        freq = np.linspace(0.001, 5, 100)
        dependent = function(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5)
        
        #create dictionary for storing data - only varying one guess
        data = {f'{param}_guess': [],
                'm1_recovered': [], 'm2_recovered': [], 'm3_recovered': [], 
                'b1_recovered': [], 'b2_recovered': [], 'b3_recovered': [], 
                'k1_recovered': [], 'k2_recovered': [], 'k3_recovered': [], 'k4_recovered': [], 
                'F_recovered': [], 
                '1-r_squared': []}
        
        #correct parameters
        parameters = {'k1':3, 'k2': 3, 'k3': 3, 'k4': 0, 
                      'b1': 2, 'b2': 2, 'b3': 2, 'F': 1, 
                      'm1': 5, 'm2': 5, 'm3': 5}
        
        #changes the first parameter (k1) by increments of 0.1 (starting at 0)
        increments_list = [i / 10 for i in range(0,5)]
        for trial in range(len(increments_list)):
            altered_guess = parameters.copy()
            altered_guess[param] = increments_list[trial]
            data[f'{param}_guess'].append(altered_guess[param]) #taken from Sam
        
            try:
                
                #curve fitting
                model = Model(function)
                params = model.make_params(k1=altered_guess['k1'], k2=altered_guess['k2'], k3=altered_guess['k3'], k4=altered_guess['k4'],
                                           b1=altered_guess['b1'], b2=altered_guess['b2'], b3=altered_guess['b3'],
                                           F=altered_guess['F'], m1=altered_guess['m1'], m2=altered_guess['m2'], m3=altered_guess['m3']
                                           )
                result = model.fit(dependent, params, w=freq)
                
                #add recovered parameters to dictionary
                for param_name in ['m1', 'm2', 'm3', 'b1', 'b2', 'b3', 'k1', 'k2', 'k3', 'k4', 'F']:
                    param_value = result.params[param_name].value
                    data[f'{param_name}_recovered'].append(param_value)
                
                
                # Extracting the R-squared value
                r_squared = 1 - result.residual.var() / np.var(dependent)
                #add r_squared to dictionary
                data['1-r_squared'].append(1-r_squared)
                
                #Graph!
                #graph original data
                plt.figure(figsize=(8,6))
                plt.plot(freq, dependent, 'go', label='Original Data')
                
                #generate points for fitted curve
                freq_fit = np.linspace(min(freq),max(freq), 500) #more w-values than before
                dependent_fit = result.model.func(freq_fit, **result.best_values)
                
                #graph fitted curve
                plt.plot(freq_fit, dependent_fit, '-', label='Fitted Curve')
                
                #graph parts
                
                
                #setting labels for graph
                if which_mass == 1:
                    save_fig_name = f'Mass_1_vary_{param}_plot_{trial+1}.png'
                    title = f'Trimer Curve Fitting for Mass 1 - Varying {param} - #{trial+1}'
                    if which_graph == 'amp':
                        ylabel = 'Amplitude (Hz)'
                    elif which_graph == 'phase':
                        ylabel = 'Phase (rad)'
                    elif which_graph == 'real':
                        ylabel = 'Real Part'
                    else: 
                        ylabel = 'Imaginary Part'
                if which_mass == 2:
                    save_fig_name = f'Mass_2_vary_{param}_plot_{trial+1}.png'
                    title = f'Trimer Curve Fitting for Mass 2 - Varying {param} - #{trial+1}'
                    if which_graph == 'amp':
                        ylabel = 'Amplitude (Hz)'
                    elif which_graph == 'phase':
                        ylabel = 'Phase (rad)'
                    elif which_graph == 'real':
                        ylabel = 'Real Part'
                    else: 
                        ylabel = 'Imaginary Part'
                if which_mass == 3:
                    save_fig_name = f'Mass_3_vary_{param}_plot_{trial+1}.png'
                    title = f'Trimer Curve Fitting for Mass 3 - Varying {param} - #{trial+1}'
                    if which_graph == 'amp':
                       ylabel = 'Amplitude (Hz)'
                    elif which_graph == 'phase':
                        ylabel = 'Phase (rad)'
                    elif which_graph == 'real':
                        ylabel = 'Real Part'
                    else: 
                        ylabel = 'Imaginary Part'
                
                plt.legend(loc='best')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel(ylabel)
                plt.title(title)
                
                #file path specific to computer you are working on
                plt.savefig(save_fig_name)
                plt.show()
            
            except RuntimeError:
                #If it takes too long, appends 0s to the dictionary
                #R^2 = 0 means parameters not recovered
                #Sam did this - I've run it a couple times and I haven't had any 0s yet
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
                data['1-r_squared'].append(0)
      
        #put all this data in a spreadsheet!
        df = pd.DataFrame(data)
        all_data.append(df)
    
        #file path specific to computer you are working on        
        file_path = 'Changing_k1_M2-Amplitude.xlsx'

        if which_mass == 1:
            if which_graph == 'amp':
                file_path = 'Changing_1_Guess_M1-Amplitude.xlsx'
            elif which_graph == 'phase':
                file_path = 'Changing_1_Guess_M1-Phase.xlsx'
            elif which_graph == 'real':
                file_path = 'Changing_1_Guess_M1-Real.xlsx'
            else: 
                file_path = 'Changing_1_Guess_M1-Imaginary.xlsx'
        if which_mass == 2:
            if which_graph == 'amp':
                file_path = 'Changing_1_Guess_M2-Amplitude.xlsx'
            elif which_graph == 'phase':
                file_path = 'Changing_1_Guess_M2-Phase.xlsx'
            elif which_graph == 'real':
                file_path = 'Changing_1_Guess_M2-Real.xlsx'
            else: 
                file_path = 'Changing_1_Guess_M2-Imaginary.xlsx'
        if which_mass == 3:
            if which_graph == 'amp':
               file_path = 'Changing_1_Guess_M3-Amplitude.xlsx'
            elif which_graph == 'phase':
                file_path = 'Changing_1_Guess_M3-Phase.xlsx'
            elif which_graph == 'real':
                file_path = 'Changing_1_Guess_M2-Real.xlsx'
            else: 
                file_path = 'Changing_1_Guess_M2-Imaginary.xlsx'
    
    #Puts each dataframe into its own sheet on the spreadsheet
    #make sure in correct order: ['m1', 'm2', 'm3', 'b1', 'b2', 'b3', 'k1', 'k2', 'k3', 'k4', 'F']
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        all_data[0].to_excel(writer, sheet_name='vary m1', index=False)
        all_data[1].to_excel(writer, sheet_name='vary m2', index=False)
        all_data[2].to_excel(writer, sheet_name='vary m3', index=False)
        all_data[3].to_excel(writer, sheet_name='vary b1', index=False)
        all_data[4].to_excel(writer, sheet_name='vary b2', index=False)
        all_data[5].to_excel(writer, sheet_name='vary b3', index=False)
        all_data[6].to_excel(writer, sheet_name='vary k1', index=False)
        all_data[7].to_excel(writer, sheet_name='vary k2', index=False)
        all_data[8].to_excel(writer, sheet_name='vary k3', index=False)
        all_data[9].to_excel(writer, sheet_name='vary k4', index=False)
        all_data[10].to_excel(writer, sheet_name='vary F', index=False)
    
    print('All files saved')





