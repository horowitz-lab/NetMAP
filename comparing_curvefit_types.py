#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:42:41 2024

@author: lydiabullock
"""

''' Which has more accurated recovered parameters: Amp & Phase or X & Y? 
    Same system, different noise - this replicates doing experiment many times 
    Using method of fixing F '''

from curve_fitting_amp_phase_all import multiple_fit_amp_phase
from curve_fitting_X_Y_all import multiple_fit_X_Y
import pandas as pd
from resonatorsimulator import complex_noise

#Make parameters/initial guesses - [k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3]
#Note that right now we only scale/fix by F, so make sure to keep F correct in guesses
true_params = [5, 5, 1, 1, 2, 2, 2, 1, 1.5, 1.5, 6.5]
guessed_params = [5.23, 4.5, 1.39, 0.47, 1.983, 2.01, 2.76, 1, 2.025, 1.7, 5.739]

starting_row = 0

with pd.ExcelWriter('Curve_Fit_Simultaneously_Which_More_Accurate.xlsx', engine='xlsxwriter') as writer:
    for i in range(50):
        
        #Create noise
        e = complex_noise(300, 2)
    
        #Get the data!
        dataframe1 = multiple_fit_amp_phase(guessed_params, true_params, e, False, True) #Fixed
        dataframe2 = multiple_fit_X_Y(guessed_params, true_params, e, False, True) #Fixed
    
        #Add to excel spreadsheet
        dataframe1.to_excel(writer, sheet_name='Amp & Phase', startrow=starting_row, index=False, header=(i==0))
        dataframe2.to_excel(writer, sheet_name='X & Y', startrow=starting_row, index=False, header=(i==0))
    
        starting_row += len(dataframe1) + (1 if i==0 else 0)

