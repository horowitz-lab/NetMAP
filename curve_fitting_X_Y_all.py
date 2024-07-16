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

##Create data - functions from simulator code
freq = np.linspace(0.001, 5, 300)
force_all = False

#noise
e = complex_noise(300, 2)

X1 = realamp1(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5, e, force_all)
Y1 = imamp1(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5, e, force_all) + 2*np.pi

X2 = realamp2(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5, e, force_all)
Y2 = imamp2(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5, e, force_all) + 2*np.pi

X3 = realamp3(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5, e, force_all)
Y3 = imamp3(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5, e, force_all) + 2*np.pi

#make parameters/initial guesses
#true parameters = [3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5]

params = lmfit.Parameters()
params.add('k1', value = 3, min=0)
params.add('k2', value = 3, min=0)
params.add('k3', value = 3.009, min=0)
params.add('k4', value = 0, min=0)
params.add('b1', value = 2, min=0)
params.add('b2', value = 1.99, min=0)
params.add('b3', value = 2.006, min=0)
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
k_1 = result.params['k1'].value
k_2 = result.params['k2'].value
k_3 = result.params['k3'].value
k_4 = result.params['k4'].value
b_1 = result.params['b1'].value
b_2 = result.params['b2'].value
b_3 = result.params['b3'].value
F_ = result.params['F'].value
m_1 = result.params['m1'].value
m_2 = result.params['m2'].value
m_3 = result.params['m3'].value

re1_fitted = re1(freq, k_1, k_2, k_3, k_4, b_1, b_2, b_3, F_, m_1, m_2, m_3)
re2_fitted = re2(freq, k_1, k_2, k_3, k_4, b_1, b_2, b_3, F_, m_1, m_2, m_3)
re3_fitted = re3(freq, k_1, k_2, k_3, k_4, b_1, b_2, b_3, F_, m_1, m_2, m_3)
im1_fitted = im1(freq, k_1, k_2, k_3, k_4, b_1, b_2, b_3, F_, m_1, m_2, m_3)
im2_fitted = im2(freq, k_1, k_2, k_3, k_4, b_1, b_2, b_3, F_, m_1, m_2, m_3)
im3_fitted = im3(freq, k_1, k_2, k_3, k_4, b_1, b_2, b_3, F_, m_1, m_2, m_3)

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

re1_guess = re1(freq, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
re2_guess = re2(freq, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
re3_guess = re3(freq, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
im1_guess = im1(freq, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
im2_guess = im2(freq, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
im3_guess = im3(freq, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)

## Begin graphing
fig = plt.figure(figsize=(10,6))
gs = fig.add_gridspec(2, 3, hspace=0.1, wspace=0.1)
((ax1, ax2, ax3), (ax4, ax5, ax6)) = gs.subplots(sharex=True, sharey='row')

#original data
ax1.plot(freq, X1,'ro',)
ax2.plot(freq, X2,'bo')
ax3.plot(freq, X3,'go')
ax4.plot(freq, Y1,'ro')
ax5.plot(freq, Y2,'bo')
ax6.plot(freq, Y3,'go')

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
ax3.set_title('Mass 2')
ax1.set_ylabel('Real')
ax4.set_ylabel('Imaginary')

for ax in fig.get_axes():
    ax.set(xlabel='Frequency')
    ax.label_outer()
plt.show()

