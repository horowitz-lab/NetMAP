#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:13:48 2024

@author: lydiabullock
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from Trimer_simulator import c2, t2, curve2, theta2
from resonatorsimulator import complex_noise

#type of function to fit for curves
def c2_function(w, k_1, k_2, k_3, k_4, b1, b2, b3, F, m1, m2, m3):
        return c2(w, k_1, k_2, k_3, k_4, b1, b2, b3, F, m1, m2, m3)
def t2_function(w, k_1, k_2, k_3, k_4, b1, b2, b3, F, m1, m2, m3):
        return t2(w, k_1, k_2, k_3, k_4, b1, b2, b3, F, m1, m2, m3)
    
#create data 
freq = np.linspace(0.001, 5, 300)
force_all = False
#noise
e = complex_noise(300, 2)

Amp = curve2(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5, e, force_all)
Phase = theta2(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5, e, force_all)

model1 = Model(c2_function)
model2 = Model(t2_function)

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
Amp_guess = c2_function(freq_guess, **initial_guesses)
Phase_guess = t2_function(freq_guess, **initial_guesses)

plt.figure(figsize=(8,6))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

#original data
ax1.plot(freq, Amp,'ro', label='Amplitude')
ax2.plot(freq, Phase,'bo', label='Phase')

#fitted curve
ax1.plot(freq_fit, Amp_fit, 'm-', label='Fitted Curve Amp 2')
ax2.plot(freq_fit, Phase_fit, 'g-', label='Fitted Curve Phase 2')

#guessed parameters curve
ax1.plot(freq_guess, Amp_guess, linestyle='dashed', color='magenta', label='Guessed Parameters Amp 2')
ax2.plot(freq_guess, Phase_guess, linestyle='dashed', color='green', label='Guessed Parameters Phase 2')

#Graph parts
ax1.set_title('Mass 2')
ax1.set_xlabel('Frequency')
ax1.set_ylabel('Amplitude')
ax2.set_ylabel('Phase')
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
