#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:51:27 2024

@author: Lydia Bullock
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from Trimer_simulator import curve1, theta1, curve2, theta2, curve3, theta3, c1, t1, c2, t2, c3, t3
from comparing_curvefit_types import complex_noise
import seaborn as sns

# Example model function
def model(params, x):
    a = params['a']
    b = params['b']
    return a * np.exp(-b * x)

# Objective function to minimize
def objective(params, x, Amp1):
    model_vals = model(params, x)
    return model_vals - Amp1

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

#Callback function to plot each iteration
def plot_callback(params, iter, resid, *args, **kws):
    plt.clf()  
    if iter % 5 == 0:
        
        freq = args[0]
        Amp1 = args[1]
    
        #Recall parameters
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
    
        #Get model data to plot
        modelc1 = c1(freq, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)
        
        # sns.reset_defaults()
        # sns.set_context("talk")
        # sns.scatterplot(freq, Amp1, 'bo', label='Data')
        # sns.scatterplot(freq, modelc1, 'r-', label='Model')
        plt.plot(freq, Amp1, 'bo', label='Data')
        plt.plot(freq, modelc1, 'r-', label='Model')
        plt.ylim(ymax=0.6)
        plt.title(f"Trimer Resonator System - Iteration: {iter}", fontsize=18)
        plt.ylabel('Amplitude (m)', fontsize=16)
        plt.xlabel('Frequency (Hz)', fontsize=16)
        plt.legend(fontsize=14)
        plt.pause(0.1)  
    
'''Begin Work Here'''
#this is using System 10 of 15 Systems - 10 Freqs NetMAP Better Params
##Create data and system parameters
#x data
freq = np.linspace(0.001, 4, 300)

e = complex_noise(300, 2)
force_all = False
params_correct = [5.385, 7.276, 5.271, 4.382, 0.984, 0.646, 0.775, 1, 3.345, 9.26, 7.439]
#[k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3]

#y data
Amp1 = curve1(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
Phase1 = theta1(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) \
    + 2 * np.pi
Amp2 = curve2(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
Phase2 = theta2(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) \
    + 2 * np.pi
Amp3 = curve3(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
Phase3 = theta3(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) \
    + 2 * np.pi

#Create parameter guesses
# params_guess = automate_guess(params_correct, 0.8)
params_guess = [4.6455, 7.1909, 4.9103, 3.4398, 1.0832, 0.596, 0.6245, 1, 3.4532, 8.7681, 8.7575]
params = Parameters()
params.add('k1', value = params_guess[0], min=0)
params.add('k2', value = params_guess[1], min=0)
params.add('k3', value = params_guess[2], min=0)
params.add('k4', value = params_guess[3], min=0)
params.add('b1', value = params_guess[4], min=0)
params.add('b2', value = params_guess[5], min=0)
params.add('b3', value = params_guess[6], min=0)
params.add('F', value = params_guess[7], min=0)
params.add('m1', value = params_guess[8], min=0)
params.add('m2', value = params_guess[9], min=0)
params.add('m3', value = params_guess[10], min=0)

# Perform minimization and plot each step!
result = minimize(residuals, params, args = (freq, Amp1, Amp2, Amp3, Phase1, Phase2, Phase3), iter_cb=plot_callback)


