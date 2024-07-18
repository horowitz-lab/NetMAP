#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:25:50 2024

@author: Student
"""
from curve_fitting_amp_phase_all import multiple_fit_amp_phase
from curve_fitting_X_Y_all import multiple_fit_X_Y
import pandas as pd
from resonatorsimulator import complex_noise

#Make parameters/initial guesses - [k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3]
#Note that right now we only scale/fix by F, so make sure to keep F correct in guesses
true_params = [4, 3, 2, 1, 1, 2, 3, 1, 1, 1, 1]
guessed_params = [4.023, 3, 1.909, 0.80911, 1.2985, 2, 2.891, 1, 1, 1.11, 1]

starting_row = 0

with pd.ExcelWriter('Curve_Fit_Simultaneously.xlsx', engine='xlsxwriter') as writer:
    for i in range(3):
        
        #Create noise
        e = complex_noise(300, 2)
    
        #Get the data!
        dataframe1 = multiple_fit_amp_phase(guessed_params, true_params, e, False, False) #Scaled
        dataframe2 = multiple_fit_amp_phase(guessed_params, true_params, e, False, True) #Fixed
        dataframe3 = multiple_fit_X_Y(guessed_params, true_params, e, False, False) #Scaled
        dataframe4 = multiple_fit_X_Y(guessed_params, true_params, e, False, True) #Fixed
    
        #Add to excel spreadsheet
    
        dataframe1.to_excel(writer, sheet_name='Amp & Phase - Scaled vs Fixed F', startrow=starting_row, index=False)
        dataframe1.to_excel(writer, sheet_name='Amp & Phase - Scaled vs Fixed F', startrow=starting_row+2, index=False, header=False)
        dataframe3.to_excel(writer, sheet_name='X & Y - Scaled vs Fixed F', startrow=starting_row, index=False)
        dataframe4.to_excel(writer, sheet_name='X & Y - Scaled vs Fixed F', startrow=starting_row+2, index=False, header=False)
       
        starting_row += 4