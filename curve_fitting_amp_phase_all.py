#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:31:59 2024

@author: lydiabullock
"""

import numpy as np
import matplotlib.pyplot as plt
import lmfit
from Trimer_simulator import curve1, theta1, curve2, theta2, curve3, theta3, c1, t1, c2, t2, c3, t3
from resonatorsimulator import complex_noise

##Create data - functions from simulator code
freq = np.linspace(0.001, 4, 300)
force_all = False

#noise
e = complex_noise(300, 2)

Amp1 = curve1(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5, e, force_all)
Phase1 = theta1(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5, e, force_all) \
    + 2 * np.pi
Amp2 = curve2(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5, e, force_all)
Phase2 = theta2(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5, e, force_all) \
    + 2 * np.pi
Amp3 = curve3(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5, e, force_all)
Phase3 = theta3(freq, 3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5, e, force_all) \
    + 2 * np.pi

#make parameters/initial guesses
#true parameters = [3, 3, 3, 0, 2, 2, 2, 1, 5, 5, 5]

params = lmfit.Parameters()
params.add('k1', value = 3, min=0)
params.add('k2', value = 3, min=0)
params.add('k3', value = 3.109, min=0)
params.add('k4', value = 0, min=0)
params.add('b1', value = 2, min=0)
params.add('b2', value = 1.99, min=0)
params.add('b3', value = 2.76, min=0)
params.add('F', value = 1, min=0)
params.add('m1', value = 5, min=0)
params.add('m2', value = 5.1568, min=0)
params.add('m3', value = 4.739, min=0)

#get residuals
def residuals(params, wd, Amp1_data, Amp2_data, Amp3_data, Phase1_data, Phase2_data, Phase3_data):
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
    
    modelc1 = c1(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
    modelc2 = c2(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
    modelc3 = c3(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
    modelt1 = t1(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
    modelt2 = t2(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
    modelt3 = t3(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
    
    residc1 = Amp1_data - modelc1
    residc2 = Amp2_data - modelc2
    residc3 = Amp3_data - modelc3
    residt1 = Phase1_data - modelt1
    residt2 = Phase2_data - modelt2
    residt3 = Phase3_data - modelt3
    
    return np.concatenate((residc1, residc2, residc3, residt1, residt2, residt3))


result = lmfit.minimize(residuals, params, args = (freq, Amp1, Amp2, Amp3, Phase1, Phase2, Phase3))
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

c1_fitted = c1(freq, k_1, k_2, k_3, k_4, b_1, b_2, b_3, F_, m_1, m_2, m_3)
c2_fitted = c2(freq, k_1, k_2, k_3, k_4, b_1, b_2, b_3, F_, m_1, m_2, m_3)
c3_fitted = c3(freq, k_1, k_2, k_3, k_4, b_1, b_2, b_3, F_, m_1, m_2, m_3)
t1_fitted = t1(freq, k_1, k_2, k_3, k_4, b_1, b_2, b_3, F_, m_1, m_2, m_3)
t2_fitted = t2(freq, k_1, k_2, k_3, k_4, b_1, b_2, b_3, F_, m_1, m_2, m_3)
t3_fitted = t3(freq, k_1, k_2, k_3, k_4, b_1, b_2, b_3, F_, m_1, m_2, m_3)

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

c1_guess = c1(freq, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
c2_guess = c2(freq, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
c3_guess = c3(freq, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
t1_guess = t1(freq, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
t2_guess = t2(freq, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
t3_guess = t3(freq, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)

## Begin graphing
fig = plt.figure(figsize=(10,6))
gs = fig.add_gridspec(2, 3, hspace=0.1, wspace=0.1)
((ax1, ax2, ax3), (ax4, ax5, ax6)) = gs.subplots(sharex=True, sharey='row')

#original data
ax1.plot(freq, Amp1,'ro',)
ax2.plot(freq, Amp2,'bo')
ax3.plot(freq, Amp3,'go')
ax4.plot(freq, Phase1,'ro')
ax5.plot(freq, Phase2,'bo')
ax6.plot(freq, Phase3,'go')

#fitted curves
ax1.plot(freq, c1_fitted,'g-', label='Best Fit')
ax2.plot(freq, c2_fitted,'r-', label='Best Fit')
ax3.plot(freq, c3_fitted,'b-', label='Best Fit')
ax4.plot(freq, t1_fitted,'g-', label='Best Fit')
ax5.plot(freq, t2_fitted,'r-', label='Best Fit')
ax6.plot(freq, t3_fitted,'b-', label='Best Fit')

#inital guess curves
ax1.plot(freq, c1_guess, linestyle='dashed', label='Initial Guess')
ax2.plot(freq, c2_guess, linestyle='dashed', label='Initial Guess')
ax3.plot(freq, c3_guess, linestyle='dashed', label='Initial Guess')
ax4.plot(freq, t1_guess, linestyle='dashed', label='Initial Guess')
ax5.plot(freq, t2_guess, linestyle='dashed', label='Initial Guess')
ax6.plot(freq, t3_guess, linestyle='dashed', label='Initial Guess')


#Graph parts
fig.suptitle('Trimer Resonator: Amplitude and Phase')
ax1.set_title('Mass 1')
ax2.set_title('Mass 2')
ax3.set_title('Mass 3')
ax1.set_ylabel('Amplitude')
ax4.set_ylabel('Phase')

for ax in fig.get_axes():
    ax.set(xlabel='Frequency')
    ax.label_outer()
plt.show()

