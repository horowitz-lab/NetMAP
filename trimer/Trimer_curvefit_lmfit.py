#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:13:48 2024

@author: lydiabullock
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from Trimer_simulator import c1, c2, c3

#type of function to fit for all three amplitude curves
def c1_function(w, k_1, k_2, k_3, k_4, b1, b2, b3, F, m1, m2, m3):
        return c1(w, k_1, k_2, k_3, k_4, b1, b2, b3, F, m1, m2, m3)
def c2_function(w, k_1, k_2, k_3, k_4, b1, b2, b3, F, m1, m2, m3):
        return c2(w, k_1, k_2, k_3, k_4, b1, b2, b3, F, m1, m2, m3)
def c3_function(w, k_1, k_2, k_3, k_4, b1, b2, b3, F, m1, m2, m3):
        return c3(w, k_1, k_2, k_3, k_4, b1, b2, b3, F, m1, m2, m3)
    
#create data for all three amplitudes
freq = np.linspace(0, 5, 300)
A_c1 = c1_function(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5)
# A_c2 = c2_function(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5)
# A_c3 = c3_function(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5)

model1 = Model(c1_function)
# model2 = Model(c2_function)
# model3 = Model(c3_function)

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
# params2 = model2.make_params(**initial_guesses)
# params3 = model3.make_params(**initial_guesses)

graph1 = model1.fit(A_c1, params1, w=freq)
# graph2 = model2.fit(A_c2, params2, w=freq)
# graph3 = model3.fit(A_c3, params3, w=freq)

#print(graph1.fit_report())
#print(graph2.fit_report())
#print(graph3.fit_report())

##Graph it!

#original data
plt.plot(freq, A_c1, 'bo', label='Data C1')
# plt.plot(freq, A_c2, 'go', label='Data C2')
# plt.plot(freq, A_c3, 'ro', label='Data C3')

#generate points for fitted curve
freq_fit = np.linspace(min(freq),max(freq), 500) #more w-values than before
A_c1_fit = graph1.model.func(freq_fit, **graph1.best_values)
#A_c2_fit = graph2.model.func(freq_fit, **graph2.best_values)
#A_c3_fit = graph3.model.func(freq_fit, **graph3.best_values)

#fitted curve
plt.plot(freq_fit, A_c1_fit, '-', label='Fitted Curve 1')
# plt.plot(freq_fit, A_c2_fit, '-', label='Fitted Curve 2')
# plt.plot(freq_fit, A_c3_fit, '-', label='Fitted Curve 3')

#generate points for guessed parameters curve
freq_guess = np.linspace(min(freq),max(freq), 500)
A_c1_guess = c1_function(freq_guess, **initial_guesses)
# A_c2_guess = c2_function(freq_guess, initial_guesses)
# A_c3_guess = c3_function(freq_guess, initial_guesses)

#guessed parameters curve
plt.plot(freq_guess, A_c1_guess, linestyle='dashed', label='Guessed Parameters 1')
# plt.plot(freq_guess, A_c2_guess, linestyle='dashed', label='Guessed Parameters 2')
# plt.plot(freq_guess, A_c3_guess, linestyle='dashed', label='Guessed Parameters 3')

#graph parts
plt.legend(loc='best')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m)')
plt.title('Curve Fitting Trimer')
plt.show()


