#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:53:26 2024

@author: samfeldman
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from sklearn.metrics import r2_score #Lydia - I don't have this package on my computer
from Trimer_simulator import c1

freq = np.linspace(.01, 5, 500)
A = c1(freq, 5, 5, 3, 5, .1, .1, .1, 1, 5, 2, 1)

def curve_func(freq, k_1, k_2, k_3, k_4, b1, b2, b3, F, m1, m2, m3):
    return c1(freq, k_1, k_2, k_3, k_4, b1, b2, b3, F, m1, m2, m3)

initial_guess = [5, 5, 3, 5, .1, .1, .1, 1, 5, 2, 1]
# Perform curve fitting
popt, pcov = curve_fit(curve_func, freq, A, p0=initial_guess)

# Extract fitting constants
k_1_fit, k_2_fit, k_3_fit, k_4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit = popt

# Print the fitting parameters
print("Fitting Parameters:")
print("k1:", k_1_fit)
print("k2:", k_2_fit)
print("k3:", k_3_fit)
print("k4:", k_4_fit)
print("b1:", b1_fit)
print("b2:", b2_fit)
print("b3:", b3_fit)
print("F:", F_fit)
print("m1:", m1_fit)
print("m2:", m2_fit)
print("m3:", m3_fit)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(freq, A, label='Original Data')
plt.xlabel('Frequency (f)')
plt.ylabel('A')
plt.title('Curve Fitting with Three Peaks')
plt.grid(True)

# Generate points for the fitted curve
freq_fit = np.linspace(min(freq), max(freq), 500)
A_fit = curve_func(freq_fit, *popt)

# Plot the fitted curve
plt.plot(freq_fit, A_fit, color='red', label='Fitted Curve')

# Add legend
plt.legend()

# Show plot
plt.show()

# Calculate R-squared
#r_squared = r2_score(A, curve_func(freq, *popt))

# Print R-squared value
#print("R-squared:", r_squared)
 