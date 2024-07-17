#!/usr/bin/env python3# -*- coding: utf-8 -*-"""Created on Tue Jul 16 11:31:59 2024@author: lydiabullock"""import numpy as npimport pandas as pdimport matplotlib.pyplot as pltimport lmfitfrom Trimer_simulator import curve1, theta1, curve2, theta2, curve3, theta3, c1, t1, c2, t2, c3, t3from resonatorstats import syserr''' 2 functions contained:    multiple_fit - Curve fits to multiple Amplitude and Phase Curves at once                 - Calculates systematic error and returns a data frame    residuals - calculates residuals of multiple data sets and concatenates them              - used in multiple_fit function to minimize the residuals of                 multiple graphs at the same time to find the best fit curve'''#Calculates and concatenates residuals given multiple data sets#Takes in parameters, frequency, and dependent variablesdef residuals(params, wd, Amp1_data, Amp2_data, Amp3_data, Phase1_data, Phase2_data, Phase3_data):    k1 = params['k1'].value    k2 = params['k2'].value    k3 = params['k3'].value     k4 = params['k4'].value     b1 = params['b1'].value    b2 = params['b2'].value     b3 = params['b3'].value     F = params['F'].value    m1 = params['m1'].value    m2 = params['m2'].value    m3 = params['m3'].value        modelc1 = c1(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)    modelc2 = c2(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)    modelc3 = c3(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)    modelt1 = t1(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)    modelt2 = t2(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)    modelt3 = t3(wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3)        residc1 = Amp1_data - modelc1    residc2 = Amp2_data - modelc2    residc3 = Amp3_data - modelc3    residt1 = Phase1_data - modelt1    residt2 = Phase2_data - modelt2    residt3 = Phase3_data - modelt3        return np.concatenate((residc1, residc2, residc3, residt1, residt2, residt3))#Takes in a *list* of correct parameters and a *list* of the guessed parameters,#as well as error and a boolean (whether you want to apply force to one or all masses)#Returns a dataframe containing guessed parameters, recovered parameters,#and systematic errordef multiple_fit_amp_phase(params_guess, params_correct, e, force_all):        ##Create data - functions from simulator code    freq = np.linspace(0.001, 4, 300)        Amp1 = curve1(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)    Phase1 = theta1(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) \        + 2 * np.pi    Amp2 = curve2(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)    Phase2 = theta2(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) \        + 2 * np.pi    Amp3 = curve3(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all)    Phase3 = theta3(freq, params_correct[0], params_correct[1], params_correct[2], params_correct[3], params_correct[4], params_correct[5], params_correct[6], params_correct[7], params_correct[8], params_correct[9], params_correct[10], e, force_all) \        + 2 * np.pi        #Create intial parameters    params = lmfit.Parameters()    params.add('k1', value = params_guess[0], min=0)    params.add('k2', value = params_guess[1], min=0)    params.add('k3', value = params_guess[2], min=0)    params.add('k4', value = params_guess[3], min=0)    params.add('b1', value = params_guess[4], min=0)    params.add('b2', value = params_guess[5], min=0)    params.add('b3', value = params_guess[6], min=0)    params.add('F', value = params_guess[7], min=0)    params.add('m1', value = params_guess[8], min=0)    params.add('m2', value = params_guess[9], min=0)    params.add('m3', value = params_guess[10], min=0)        #Create dictionary for storing data    data = {'k1_guess': [], 'k2_guess': [], 'k3_guess': [], 'k4_guess': [],            'b1_guess': [], 'b2_guess': [], 'b3_guess': [], 'F_guess': [],            'm1_guess': [], 'm2_guess': [], 'm3_guess': [],               'k1_recovered': [], 'k2_recovered': [], 'k3_recovered': [], 'k4_recovered': [],             'b1_recovered': [], 'b2_recovered': [], 'b3_recovered': [],             'm1_recovered': [], 'm2_recovered': [], 'm3_recovered': [], 'F_recovered': [],             'syserr_k1': [], 'syserr_k2': [], 'syserr_k3': [], 'syserr_k4': [],            'syserr_b1': [], 'syserr_b2': [], 'syserr_b3': [], 'syserr_F': [],             'syserr_m1': [], 'syserr_m2': [], 'syserr_m3': []}        #get resulting data and fit parameters by minimizing the residuals    result = lmfit.minimize(residuals, params, args = (freq, Amp1, Amp2, Amp3, Phase1, Phase2, Phase3))    # print(lmfit.fit_report(result))        #Create dictionary of true parameters from list provided (need for compliting data)    true_params = {'k1': params_correct[0], 'k2': params_correct[1], 'k3': params_correct[2], 'k4': params_correct[3],                   'b1': params_correct[4], 'b2': params_correct[5], 'b3': params_correct[6], 'F': params_correct[7],                   'm1': params_correct[8], 'm2': params_correct[9], 'm3': params_correct[10]}        #Compling the Data    for param_name in ['k1','k2','k3','k4','b1','b2','b3','F','m1','m2','m3']:        #Add guessed parameters to dictionary        param_guess = params[param_name].value        data[f'{param_name}_guess'].append(param_guess)                #Add fitted parameters to dictionary        param_fit = result.params[param_name].value        data[f'{param_name}_recovered'].append(param_fit)                #Calculate systematic error and add to dictionary        param_true = true_params[param_name]        systematic_error = syserr(param_fit, param_true)        data[f'syserr_{param_name}'].append(systematic_error)                #Create fitted y-values (for graphing)    k1_fit = result.params['k1'].value    k2_fit = result.params['k2'].value    k3_fit = result.params['k3'].value    k4_fit = result.params['k4'].value    b1_fit = result.params['b1'].value    b2_fit = result.params['b2'].value    b3_fit = result.params['b3'].value    F_fit = result.params['F'].value    m1_fit = result.params['m1'].value    m2_fit = result.params['m2'].value    m3_fit= result.params['m3'].value        c1_fitted = c1(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)    c2_fitted = c2(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)    c3_fitted = c3(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)    t1_fitted = t1(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)    t2_fitted = t2(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)    t3_fitted = t3(freq, k1_fit, k2_fit, k3_fit, k4_fit, b1_fit, b2_fit, b3_fit, F_fit, m1_fit, m2_fit, m3_fit)        #Create intial guessed y-values (for graphing)    k1_guess = params['k1'].value    k2_guess = params['k2'].value    k3_guess = params['k3'].value     k4_guess = params['k4'].value     b1_guess = params['b1'].value    b2_guess = params['b2'].value     b3_guess = params['b3'].value     F_guess = params['F'].value    m1_guess = params['m1'].value    m2_guess = params['m2'].value    m3_guess = params['m3'].value        c1_guess = c1(freq, k1_guess, k2_guess, k3_guess, k4_guess, b1_guess, b2_guess, b3_guess, F_guess, m1_guess, m2_guess, m3_guess)    c2_guess = c2(freq, k1_guess, k2_guess, k3_guess, k4_guess, b1_guess, b2_guess, b3_guess, F_guess, m1_guess, m2_guess, m3_guess)    c3_guess = c3(freq, k1_guess, k2_guess, k3_guess, k4_guess, b1_guess, b2_guess, b3_guess, F_guess, m1_guess, m2_guess, m3_guess)    t1_guess = t1(freq, k1_guess, k2_guess, k3_guess, k4_guess, b1_guess, b2_guess, b3_guess, F_guess, m1_guess, m2_guess, m3_guess)    t2_guess = t2(freq, k1_guess, k2_guess, k3_guess, k4_guess, b1_guess, b2_guess, b3_guess, F_guess, m1_guess, m2_guess, m3_guess)    t3_guess = t3(freq, k1_guess, k2_guess, k3_guess, k4_guess, b1_guess, b2_guess, b3_guess, F_guess, m1_guess, m2_guess, m3_guess)        ## Begin graphing    fig = plt.figure(figsize=(16,8))    gs = fig.add_gridspec(2, 3, hspace=0.1, wspace=0.1)    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = gs.subplots(sharex=True, sharey='row')        #original data    ax1.plot(freq, Amp1,'ro', alpha=0.5, markersize=5.5)    ax2.plot(freq, Amp2,'bo', alpha=0.5, markersize=5.5)    ax3.plot(freq, Amp3,'go', alpha=0.5, markersize=5.5)    ax4.plot(freq, Phase1,'ro', alpha=0.5, markersize=5.5)    ax5.plot(freq, Phase2,'bo', alpha=0.5, markersize=5.5)    ax6.plot(freq, Phase3,'go', alpha=0.5, markersize=5.5)        #fitted curves    ax1.plot(freq, c1_fitted,'c-', label='Best Fit', lw=2.5)    ax2.plot(freq, c2_fitted,'r-', label='Best Fit', lw=2.5)    ax3.plot(freq, c3_fitted,'m-', label='Best Fit', lw=2.5)    ax4.plot(freq, t1_fitted,'c-', label='Best Fit', lw=2.5)    ax5.plot(freq, t2_fitted,'r-', label='Best Fit', lw=2.5)    ax6.plot(freq, t3_fitted,'m-', label='Best Fit', lw=2.5)        #inital guess curves    ax1.plot(freq, c1_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')    ax2.plot(freq, c2_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')    ax3.plot(freq, c3_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')    ax4.plot(freq, t1_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')    ax5.plot(freq, t2_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')    ax6.plot(freq, t3_guess, color='#4682B4', linestyle='dashed', label='Initial Guess')            #Graph parts    fig.suptitle('Trimer Resonator: Amplitude and Phase', fontsize=16)    ax1.set_title('Mass 1', fontsize=14)    ax2.set_title('Mass 2', fontsize=14)    ax3.set_title('Mass 3', fontsize=14)    ax1.set_ylabel('Amplitude')    ax4.set_ylabel('Phase')        for ax in fig.get_axes():        ax.set(xlabel='Frequency')        ax.label_outer()        ax.legend()            plt.show()        df = pd.DataFrame(data)    return df