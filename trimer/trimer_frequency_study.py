#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:35:48 2024

@author: lydiabullock
"""
from comparing_curvefit_types import complex_noise, get_parameters_NetMAP, find_avg_e
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Trimer_simulator import realamp1, realamp2, realamp3, imamp1, imamp2, imamp3

#Code that loops through frequency points of different spacing 

def sweep_freq_pair(frequencies, params_guess, params_correct, e, force_all):
    
    #Graph Real vs Imaginary for the trimer
    X1 = realamp1(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    Y1 = imamp1(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) 
    plt.plot(X1, Y1)
    plt.xlabel('Re(Z) (m)')
    plt.ylabel('Im(Z) (m)')
    plt.title('Resonator 1')
    plt.show()
    
    # X2 = realamp2(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    # Y2 = imamp2(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) 
    
    # X3 = realamp3(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    # Y3 = imamp3(frequencies, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    
    # Loop over possible combinations of frequency indices, i1 and i2
    for i1 in range(len(frequencies)):
        freq1 = frequencies[i1]
        

        for i2 in range(len(frequencies)):
            freq2 = frequencies[i2]
            freqs = [freq1, freq2]

            NetMAP_info = get_parameters_NetMAP(freqs, params_guess, params_correct, e, force_all)
            
            #Find <e> (average across parameters) for the trial and add to dictionary
            avg_e1 = find_avg_e(NetMAP_info)
            NetMAP_info['<e>'] = avg_e1
            
        
            try: # repeated experiments results
                resultsdf = pd.concat([resultsdf, NetMAP_info], ignore_index=True)
            except:
                resultsdf = NetMAP_info
    
    return resultsdf 


''' Begin work here. '''

e = complex_noise(5,2)
frequencies = np.linspace(0.001, 4, 5)

params_guess = 
params_correct = 
force_all = False

result = sweep_freq_pair(frequencies, params_guess, params_correct, e, force_all)


