#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:31:59 2024

@author: lydiabullock
"""

import numpy as np
import matplotlib.pyplot as plt
import lmfit
from Trimer_simulator import re1, re2, re3, im1, im2, im3, realamp1, realamp2, realamp3, imamp1, imamp2, imamp3
from resonatorsimulator import complex_noise
from resonatorstats import rsqrd, syserr

##Create data - functions from simulator code
freq = np.linspace(0.001, 5, 300)
force_all = False

#noise
e = complex_noise(300, 2)

X1 = realamp1(freq, 3, 3, 3, 0.5, 2, 2, 2, 1, 5, 5, 5, e, force_all)
Y1 = imamp1(freq, 3, 3, 3, 0.5, 2, 2, 2, 1, 5, 5, 5, e, force_all) 

X2 = realamp2(freq, 3, 3, 3, 0.5, 2, 2, 2, 1, 5, 5, 5, e, force_all)
Y2 = imamp2(freq, 3, 3, 3, 0.5, 2, 2, 2, 1, 5, 5, 5, e, force_all) 

X3 = realamp3(freq, 3, 3, 3, 0.5, 2, 2, 2, 1, 5, 5, 5, e, force_all)
Y3 = imamp3(freq, 3, 3, 3, 0.5, 2, 2, 2, 1, 5, 5, 5, e, force_all) 

#make parameters/initial guesses
true_params = {'k1': 3, 'k2': 3, 'k3': 3, 'k4': 0.5,
               'b1': 2, 'b2': 2, 'b3': 2, 'F': 1,
               'm1': 5, 'm2': 5, 'm3': 5}

params = lmfit.Parameters()
params.add('k1', value = 3, min=0)
params.add('k2', value = 3, min=0)
params.add('k3', value = 3.109, min=0)
params.add('k4', value = 0.47, min=0)
params.add('b1', value = 2, min=0)
params.add('b2', value = 1.99, min=0)
params.add('b3', value = 2.76, min=0)
params.add('F', value = 1, min=0)
params.add('m1', value = 5, min=0)
params.add('m2', value = 5.1568, min=0)
params.add('m3', value = 4.739, min=0)

#get residuals
def residuals(params, wd, X1_data, X2_data, X3_data, Y1_data, Y2_data, Y3_data):
    k1 = params['k1'].value
    k2 = params['k2'].value
    k3 = params['k3'].value 
    k4 = params['k4'].value 
    b1 = params['b1'].value
    b2 = params['b2'].value 
    b3 = params['b3'].value 
    F = params['F'].value
    m1 = params['m1'].value
    m2 = params['m2'].value
    m3 = params['m3'].value
    
    modelre1 = re1(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
    modelre2 = re2(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
    modelre3 = re3(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
    modelim1 = im1(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
    modelim2 = im2(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
    modelim3 = im3(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
    
    residX1 = X1_data - modelre1
    residX2 = X2_data - modelre2
    residX3 = X3_data - modelre3
    residY1 = Y1_data - modelim1
    residY2 = Y2_data - modelim2
    residY3 = Y3_data - modelim3
    
    return np.concatenate((residX1, residX2, residX3, residY1, residY2, residY3))


result = lmfit.minimize(residuals, params, args = (freq, X1, X2, X3, Y1, Y2, Y3))
print(lmfit.fit_report(result))

#Create fitted y-values and intial guessed y-values
k1_fit = result.params['k1'].value
k2_fit = result.params['k2'].value
k3_fit = result.params['k3'].value
k4_fit = result.params['k4'].value
b1_fit = result.params['b1'].value
b2_fit = result.params['b2'].value
b3_fit = result.params['b3'].value
F_fit = result.params['F'].value
m1_fit = result.params['m1'].value
m2_fit = result.params['m2'].value
m3_fit= result.params['m3'].value

re1_fitted = re1(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)
re2_fitted = re2(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)
re3_fitted = re3(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)
im1_fitted = im1(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)
im2_fitted = im2(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)
im3_fitted = im3(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)

k1_guess = params['k1'].value
k2_guess = params['k2'].value
k3_guess = params['k3'].value 
k4_guess = params['k4'].value 
b1_guess = params['b1'].value
b2_guess = params['b2'].value 
b3_guess = params['b3'].value 
F_guess = params['F'].value
m1_guess = params['m1'].value
m2_guess = params['m2'].value
m3_guess = params['m3'].value

re1_guess = re1(freq, k1_guess, k2_guess, k3_guess, k4_guess, b1_guess, b2_guess, b3_guess, F_guess, m1_guess, m2_guess, m3_guess)
re2_guess = re2(freq, k1_guess, k2_guess, k3_guess, k4_guess, b1_guess, b2_guess, b3_guess, F_guess, m1_guess, m2_guess, m3_guess)
re3_guess = re3(freq, k1_guess, k2_guess, k3_guess, k4_guess, b1_guess, b2_guess, b3_guess, F_guess, m1_guess, m2_guess, m3_guess)
im1_guess = im1(freq, k1_guess, k2_guess, k3_guess, k4_guess, b1_guess, b2_guess, b3_guess, F_guess, m1_guess, m2_guess, m3_guess)
im2_guess = im2(freq, k1_guess, k2_guess, k3_guess, k4_guess, b1_guess, b2_guess, b3_guess, F_guess, m1_guess, m2_guess, m3_guess)
im3_guess = im3(freq, k1_guess, k2_guess, k3_guess, k4_guess, b1_guess, b2_guess, b3_guess, F_guess, m1_guess, m2_guess, m3_guess)

## Begin graphing
fig = plt.figure(figsize=(11,7))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.05)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
ax4 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax5 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax4)
ax6 = fig.add_subplot(gs[1, 2], sharex=ax1, sharey=ax4)
ax7 = fig.add_subplot(gs[2, 0])
ax8 = fig.add_subplot(gs[2, 1], sharex=ax7, sharey=ax7)
ax9 = fig.add_subplot(gs[2, 2], sharex=ax7, sharey=ax7)

#original data
ax1.plot(freq, X1,'ro',)
ax2.plot(freq, X2,'bo')
ax3.plot(freq, X3,'go')
ax4.plot(freq, Y1,'ro')
ax5.plot(freq, Y2,'bo')
ax6.plot(freq, Y3,'go')
ax7.plot(X1,Y1,'ro')
ax8.plot(X2,Y2,'bo')
ax9.plot(X3,Y3,'go')

#fitted curves
ax1.plot(freq, re1_fitted,'g-', label='Best Fit')
ax2.plot(freq, re2_fitted,'r-', label='Best Fit')
ax3.plot(freq, re3_fitted,'b-', label='Best Fit')
ax4.plot(freq, im1_fitted,'g-', label='Best Fit')
ax5.plot(freq, im2_fitted,'r-', label='Best Fit')
ax6.plot(freq, im3_fitted,'b-', label='Best Fit')


#inital guess curves
ax1.plot(freq, re1_guess, linestyle='dashed', label='Initial Guess')
ax2.plot(freq, re2_guess, linestyle='dashed', label='Initial Guess')
ax3.plot(freq, re3_guess, linestyle='dashed', label='Initial Guess')
ax4.plot(freq, im1_guess, linestyle='dashed', label='Initial Guess')
ax5.plot(freq, im2_guess, linestyle='dashed', label='Initial Guess')
ax6.plot(freq, im3_guess, linestyle='dashed', label='Initial Guess')


#Graph parts
fig.suptitle('Trimer Resonator: Real and Imaginary')
ax1.set_title('Mass 1')
ax2.set_title('Mass 2')
ax3.set_title('Mass 3')
ax1.set_ylabel('Real')
ax4.set_ylabel('Imaginary')
ax7.set_ylabel('Imaginary')

ax1.label_outer()
ax2.label_outer()
ax3.label_outer()
ax5.tick_params(labelleft=False)
ax6.tick_params(labelleft=False)
ax7.label_outer()
ax8.label_outer()
ax9.label_outer()
    
ax4.set_xlabel('Frequency')
ax5.set_xlabel('Frequency')
ax6.set_xlabel('Frequency')
ax7.set_xlabel('Real')
ax8.set_xlabel('Real')
ax9.set_xlabel('Real')

plt.show()

#create dictionary for storing data
data = {'k1_guess': [], 'k2_guess': [], 'k3_guess': [], 'k4_guess': [],
        'b1_guess': [], 'b2_guess': [], 'b3_guess': [], 'F_guess': [],
        'm1_guess': [], 'm2_guess': [], 'm3_guess': [],   
        'k1_recovered': [], 'k2_recovered': [], 'k3_recovered': [], 'k4_recovered': [], 
        'b1_recovered': [], 'b2_recovered': [], 'b3_recovered': [], 
        'm1_recovered': [], 'm2_recovered': [], 'm3_recovered': [], 'F_recovered': [], 
        'syserr_k1': [], 'syserr_k2': [], 'syserr_k3': [], 'syserr_k4': [],
        'syserr_b1': [], 'syserr_b2': [], 'syserr_b3': [], 'syserr_F': [], 
        'syserr_m1': [], 'syserr_m2': [], 'syserr_m3': []}

for param_name in ['k1','k2','k3','k4','b1','b2','b3','F','m1','m2','m3']:
    #Add guessed parameters to dataframe
    param_guess = params[param_name].value
    data[f'{param_name}_guess'].append(param_guess)
    
    #Add fitted parameters to dataframe
    param_fit = result.params[param_name].value
    data[f'{param_name}_recovered'].append(param_fit)
    
    #Calculate systematic error and add to dataframe
    param_true = true_params[param_name]
    systematic_error = syserr(param_fit, param_true)
    data[f'syserr_{param_name}'].append(systematic_error)
    
print(data)

