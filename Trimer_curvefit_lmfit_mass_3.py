#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:13:48 2024

@author: lydiabullock
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from Trimer_simulator import c3, t3

#type of function to fit for all three amplitude curves
def c3_function(w, k_1, k_2, k_3, k_4, b1, b2, b3, F, m1, m2, m3):
        return c3(w, k_1, k_2, k_3, k_4, b1, b2, b3, F, m1, m2, m3)
def t3_function(w, k_1, k_2, k_3, k_4, b1, b2, b3, F, m1, m2, m3):
        return t3(w, k_1, k_2, k_3, k_4, b1, b2, b3, F, m1, m2, m3)
    
#create data for all three amplitudes
freq = np.linspace(0.001, 5, 300)
Amp = c3_function(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5)
Phase = t3_function(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5)

model1 = Model(c3_function)
model2 = Model(t3_function)

#make parameters/initial guesses
#true parameters = [3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5]
initial_guesses = {
    'k_1': 3,
    'k_2': 3,
    'k_3': 3.53,
    'k_4': 0,
    'b1': 1,
    'b2': 0.9,
    'b3': 0.1,
    'F': 1,
    'm1': 5,
    'm2': 5,
    'm3': 4.5
}

params1 = model1.make_params(**initial_guesses)
params2 = model2.make_params(**initial_guesses)

graph1 = model1.fit(Amp, params1, w=freq)
graph2 = model2.fit(Phase, params2, w=freq)

#print(graph1.fit_report())
#print(graph2.fit_report())

##Graph it!

#generate points for fitted curve
freq_fit = np.linspace(min(freq),max(freq), 500) #more w-values than before
Amp_fit = graph1.model.func(freq_fit, **graph1.best_values)
Phase_fit = graph2.model.func(freq_fit, **graph2.best_values)

#generate points for guessed parameters curve
freq_guess = np.linspace(min(freq),max(freq), 500)
Amp_guess = c3_function(freq_guess, **initial_guesses)
Phase_guess = t3_function(freq_guess, **initial_guesses)

plt.figure(figsize=(8,6))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

#original data
ax1.plot(freq, Amp,'ro', label='Amplitude')
ax2.plot(freq, Phase,'bo', label='Phase')

#fitted curve
ax1.plot(freq_fit, Amp_fit, 'm-', label='Fitted Curve Amp')
ax2.plot(freq_fit, Phase_fit, 'g-', label='Fitted Curve Phase')

#guessed parameters curve
ax1.plot(freq_guess, Amp_guess, linestyle='dashed', color='magenta', label='Guessed Parameters Amp 1')
ax2.plot(freq_guess, Phase_guess, linestyle='dashed', color='green', label='Guessed Parameters Phase 1')

#Graph parts
ax1.set_title('Mass 3')
ax1.set_xlabel('Frequency')
ax1.set_ylabel('Amplitude')
ax2.set_ylabel('Phase')
ax1.legend(loc='upper right')
ax2.legend(loc='center right')
