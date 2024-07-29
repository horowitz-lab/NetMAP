#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:31:59 2024

@author: lydiabullock
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import lmfit
import warnings
from Trimer_simulator import re1, re2, re3, im1, im2, im3, realamp1, realamp2, realamp3, imamp1, imamp2, imamp3
''' 3 functions contained:
    multiple_fit - Curve fits to multiple Amplitude and Phase Curves at once
                 - Calculates systematic error and returns a dictionary of info
                 - Graphs curve fit analysis
    residuals - calculates residuals of multiple data sets and concatenates them
              - used in multiple_fit function to minimize the residuals of 
                multiple graphs at the same time to find the best fit curve
    save_figure - saves the curve fit graph created to a named folder
    syserr - calculates systematic error
    rsqrd - calculates R^2
'''

def syserr(x_found,x_set, absval = True):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        se = 100*(x_found-x_set)/x_set
    if absval:
        return abs(se)
    else:
        return se

"""
This definition of R^2 can come out negative.
Negative means that a flat line would fit the data better than the curve.
"""
def rsqrd(model, data, plot=False, x=None, newfigure = True):
    SSres = sum((data - model)**2)
    SStot = sum((data - np.mean(data))**2)
    rsqrd = 1 - (SSres/ SStot)
    
    if plot:
        if newfigure:
            plt.figure()
        plt.plot(x,data, 'o')
        plt.plot(x, model, '--')
    
    return rsqrd

#Get residuals
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

def save_figure(figure, folder_name, file_name):
    # Create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Save the figure to the folder
    file_path = os.path.join(folder_name, file_name)
    figure.savefig(file_path)
    plt.close(figure)

#Takes in a *list* of correct parameters and a *list* of the guessed parameters,
#as well as error and three booleans (whether you want to apply force to one or all masses,
#scale by force, or fix the force)
#
#Returns a dataframe containing guessed parameters, recovered parameters,
#and systematic error
def multiple_fit_X_Y(params_guess, params_correct, e, force_all, fix_F, graph_folder_name, graph_name):
    
    ##Create data - functions from simulator code
    freq = np.linspace(0.001, 4, 300)
    
    X1 = realamp1(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    Y1 = imamp1(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) 
    
    X2 = realamp2(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    Y2 = imamp2(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) 
    
    X3 = realamp3(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    Y3 = imamp3(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)
    
    #Create intial parameters
    params = lmfit.Parameters()
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
    
    #If you plan on fixing F so it cannot be changed
    if fix_F: 
        params['F'].vary = False
    
    #Create dictionary for storing data
    data = {'k1_true': [], 'k2_true': [], 'k3_true': [], 'k4_true': [],
            'b1_true': [], 'b2_true': [], 'b3_true': [],
            'm1_true': [], 'm2_true': [], 'm3_true': [],  'F_true': [],
            'k1_guess': [], 'k2_guess': [], 'k3_guess': [], 'k4_guess': [],
            'b1_guess': [], 'b2_guess': [], 'b3_guess': [],
            'm1_guess': [], 'm2_guess': [], 'm3_guess': [],  'F_guess': [],  
            'k1_recovered': [], 'k2_recovered': [], 'k3_recovered': [], 'k4_recovered': [], 
            'b1_recovered': [], 'b2_recovered': [], 'b3_recovered': [], 
            'm1_recovered': [], 'm2_recovered': [], 'm3_recovered': [], 'F_recovered': [], 
            'e_k1': [], 'e_k2': [], 'e_k3': [], 'e_k4': [],
            'e_b1': [], 'e_b2': [], 'e_b3': [], 'e_F': [], 
            'e_m1': [], 'e_m2': [], 'e_m3': [],
            'X1_rsqrd': [], 'X2_rsqrd': [], 'X3_rsqrd': [],
            'Y1_rsqrd': [], 'Y2_rsqrd': [], 'Y3_rsqrd': []}
    
    
    #get resulting data and fit parameters by minimizing the residuals
    result = lmfit.minimize(residuals, params, args = (freq, X1, X2, X3, Y1, Y2, Y3))
    #print(lmfit.fit_report(result))
    
    #Create dictionary of true parameters from list provided (need for compiling data bc I can't do it with a list)
    true_params = {'k1': params_correct[0], 'k2': params_correct[1], 'k3': params_correct[2], 'k4': params_correct[3],
                   'b1': params_correct[4], 'b2': params_correct[5], 'b3': params_correct[6], 'F': params_correct[7],
                   'm1': params_correct[8], 'm2': params_correct[9], 'm3': params_correct[10]}
    
    #Compling the Data
    for param_name in ['k1','k2','k3','k4','b1','b2','b3','F','m1','m2','m3']:
        #Add true parameters to dictionary
        param_true = true_params[param_name]
        data[f'{param_name}_true'].append(param_true)
        
        #Add guessed parameters to dictionary
        param_guess = params[param_name].value
        data[f'{param_name}_guess'].append(param_guess)
        
        #If you planned on fixing F so it cannot be changed
        if fix_F: 
            #Add fitted parameters to dictionary
            param_fit = result.params[param_name].value
            data[f'{param_name}_recovered'].append(param_fit)
            
            #Calculate systematic error and add to dictionary
            systematic_error = syserr(param_fit, param_true)
            data[f'e_{param_name}'].append(systematic_error)
        
        else:
            #If you included F in parameters to be varied, you must scale by F
            #Scale fitted parameters by force
            param_fit = result.params[param_name].value
            scaling_factor = (true_params['F'])/(result.params['F'].value)
            scaled_param_fit = param_fit*scaling_factor
            
            #Add fitted parameters to dictionary
            data[f'{param_name}_recovered'].append(scaled_param_fit)
        
            #Calculate systematic error and add to dictionary
            param_true = true_params[param_name]
            systematic_error = syserr(scaled_param_fit, param_true)
            data[f'e_{param_name}'].append(systematic_error)
    
    #Create fitted y-values (for rsrd and graphing)
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
    
    X1_fitted = re1(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)
    X2_fitted = re2(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)
    X3_fitted = re3(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)
    Y1_fitted = im1(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)
    Y2_fitted = im2(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)
    Y3_fitted = im3(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)
    
    #Calculate R^2
    X1_rsqrd = rsqrd(X1_fitted, X1)
    X2_rsqrd = rsqrd(X2_fitted, X2)
    X3_rsqrd = rsqrd(X3_fitted, X3)
    Y1_rsqrd = rsqrd(Y1_fitted, Y1)
    Y2_rsqrd = rsqrd(Y2_fitted, Y2)
    Y3_rsqrd = rsqrd(Y3_fitted, Y3)
    data['X1_rsqrd'].append(X1_rsqrd)
    data['X2_rsqrd'].append(X2_rsqrd)
    data['X3_rsqrd'].append(X3_rsqrd)
    data['Y1_rsqrd'].append(Y1_rsqrd)
    data['Y2_rsqrd'].append(Y2_rsqrd)
    data['Y3_rsqrd'].append(Y3_rsqrd)
    
    #Create intial guessed y-values (for graphing)
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
    fig = plt.figure(figsize=(16,11))
    gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.05)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax5 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax4)
    ax6 = fig.add_subplot(gs[1, 2], sharex=ax1, sharey=ax4)
    ax7 = fig.add_subplot(gs[2, 0], aspect='equal')
    ax8 = fig.add_subplot(gs[2, 1], sharex=ax7, sharey=ax7, aspect='equal')
    ax9 = fig.add_subplot(gs[2, 2], sharex=ax7, sharey=ax7, aspect='equal')
    
    #original data
    ax1.plot(freq, X1,'ro', alpha=0.5, markersize=5.5, label = 'Data')
    ax2.plot(freq, X2,'bo', alpha=0.5, markersize=5.5, label = 'Data')
    ax3.plot(freq, X3,'go', alpha=0.5, markersize=5.5, label = 'Data')
    ax4.plot(freq, Y1,'ro', alpha=0.5, markersize=5.5, label = 'Data')
    ax5.plot(freq, Y2,'bo', alpha=0.5, markersize=5.5, label = 'Data')
    ax6.plot(freq, Y3,'go', alpha=0.5, markersize=5.5, label = 'Data')
    ax7.plot(X1,Y1,'ro', alpha=0.5, markersize=5.5, label = 'Data')
    ax8.plot(X2,Y2,'bo', alpha=0.5, markersize=5.5, label = 'Data')
    ax9.plot(X3,Y3,'go', alpha=0.5, markersize=5.5, label = 'Data')
    
    #fitted curves
    ax1.plot(freq, X1_fitted,'c-', label='Best Fit', lw=2.5)
    ax2.plot(freq, X2_fitted,'r-', label='Best Fit', lw=2.5)
    ax3.plot(freq, X3_fitted,'m-', label='Best Fit', lw=2.5)
    ax4.plot(freq, Y1_fitted,'c-', label='Best Fit', lw=2.5)
    ax5.plot(freq, Y2_fitted,'r-', label='Best Fit', lw=2.5)
    ax6.plot(freq, Y3_fitted,'m-', label='Best Fit', lw=2.5)
    ax7.plot(X1_fitted, Y1_fitted, 'c-', label='Best Fit', lw=2.5)
    ax8.plot(X2_fitted, Y2_fitted, 'r-', label='Best Fit', lw=2.5)
    ax9.plot(X3_fitted, Y3_fitted, 'm-', label='Best Fit', lw=2.5)
    
    #inital guess curves
    ax1.plot(freq, re1_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax2.plot(freq, re2_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax3.plot(freq, re3_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax4.plot(freq, im1_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax5.plot(freq, im2_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax6.plot(freq, im3_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax7.plot(re1_guess, im1_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax8.plot(re2_guess, im2_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    ax9.plot(re3_guess, im3_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')
    
    #Graph parts
    fig.suptitle('Trimer Resonator: Real and Imaginary', fontsize=16)
    ax1.set_title('Mass 1', fontsize=14)
    ax2.set_title('Mass 2', fontsize=14)
    ax3.set_title('Mass 3', fontsize=14)
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
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    ax7.legend(fontsize='10')
    ax8.legend(fontsize='10')
    ax9.legend(fontsize='10')
    
    plt.show()
    save_figure(fig, graph_folder_name, graph_name)
    
    return data
