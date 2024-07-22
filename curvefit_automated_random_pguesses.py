#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:48:10 2024

@author: lydiabullock
"""
''' Function that automates guess parameters.
    Seeing success of curve fit with random guesses? '''

import random
from curve_fitting_X_Y_all import multiple_fit_X_Y
import pandas as pd
from resonatorsimulator import complex_noise

def automate_guess(true_params, threshold, interval):
    params_guess = []
    for index, value in enumerate(true_params):
        if index == 7: #Doing this because we must know what Force is going in
            params_guess.append(value)
        else:
            num = random.uniform(value-threshold, value+threshold)
            rounded_num = round(num / interval) * interval # Round the number to the nearest interval
            formatted_num = round(rounded_num, 4) # Just in case, format to 4 decimal places
            params_guess.append(formatted_num)
    return params_guess

#Create noise
e = complex_noise(300, 2)

true_params = [5, 5, 1, 1, 2, 2, 2, 1, 1.5, 1.5, 6.5]

starting_row = 0

with pd.ExcelWriter('Curve_Fit_Simultaneously_Auto_Random_Guess.xlsx', engine='xlsxwriter') as writer:
    
    for i in range(10):
        
        #Created different guess parameters
        guessed_params = automate_guess(true_params, 2, 0.001)
    
        #Get the data!
        dictionary1 = multiple_fit_X_Y(guessed_params, true_params, e, False, True) #Fixed
        
        #Convert to dataframe
        dataframe1 = pd.DataFrame(dictionary1)
    
        #Add to excel spreadsheet
        dataframe1.to_excel(writer, sheet_name='X & Y', startrow=starting_row, index=False, header=(i==0))
    
        starting_row += len(dataframe1) + (1 if i==0 else 0)
        
    