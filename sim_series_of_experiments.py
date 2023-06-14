# -*- coding: utf-8 -*-
"""
sim_series_of_experiments
Created on Tue Oct 25 10:42:31 2022

@author: vhorowit

The following are functions that simulate a series of different experiments
"""
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from helperfunctions import read_params
from resonatorfrequencypicker import freqpoints, res_freq_numeric, \
    allmeasfreq_one_res, allmeasfreq_two_res
from simulated_experiment import simulated_experiment
from resonatorsimulator import calculate_spectra


def vary_num_p_with_fixed_freqdiff(vals_set, noiselevel, 
                                   MONOMER, forceboth,reslist,
                                   minfreq = .1, maxfreq = 5,
                                   max_num_p = 10,  
                                   n = 100, # number of frequencies for R^2
                                   freqdiff = .1,just_res1 = False, repeats = 100,
                                   verbose = False,recalculate_randomness=True, use_R2_only = False,
                                   **kwargs
                                   ):
    if True:
        print('Running vary_num_p_with_fixed_freqdiff() with max of', max_num_p, 'freqs.' )
    
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)

    if MONOMER:
        numtoreturn = 1
    else:
        numtoreturn = 2
    
    for i in range(5):## To be fair for each, I use iterations to really nail down the highest amplitudes.
        reslist = res_freq_numeric(vals_set=vals_set, MONOMER=MONOMER,forceboth=forceboth,
                    mode = 'amp', iterations = 3, includefreqs = reslist,
                    unique = True, veryunique = True, numtoreturn = numtoreturn, 
                    use_R2_only = use_R2_only,
                    verboseplot = False, verbose=verbose)
    ## measure the top two resonant frequencies
    res1 = reslist[0]
    if not MONOMER:
        res2 = reslist[1]
    
    if just_res1 or MONOMER:
        freqlist = allmeasfreq_one_res(res1, max_num_p, freqdiff)
    else:
        freqlist = allmeasfreq_two_res(res1,res2, max_num_p, freqdiff)
    
    freqlist = freqlist[:max_num_p]
    assert len(freqlist) == max_num_p
    
    #define driving frequency range (gives array of n evenly spaced numbers between 0.1 and 5) and also whatever we want to measure
    drive = np.sort(np.append(np.linspace(minfreq, maxfreq, num = n), freqlist ))
    
    R1_amp, R1_phase, R2_amp, R2_phase, R1_real_amp, R1_im_amp, R2_real_amp, R2_im_amp,_ = calculate_spectra(
            drive, vals_set, noiselevel=noiselevel, MONOMER=MONOMER, forceboth=forceboth)
    
    noiseless_spectra = calculate_spectra(drive, vals_set, noiselevel = 0, MONOMER = MONOMER, forceboth = forceboth)

    for y in range(repeats):
        if y > 0:
            verbose = False
        
        if not recalculate_randomness: # at least need to calculate it every repeat
                noisy_spectra = calculate_spectra(drive, vals_set, 
                                                  noiselevel=noiselevel, MONOMER=MONOMER, forceboth=forceboth)
            
        for this_num_p in range(2, max_num_p+1):
            if y == 0 and (this_num_p == max_num_p or  this_num_p == 2): # first time with 2 or all the frequencies
                verbose = True
            else:
                verbose = False

            ## Do we recalculate the spectra every time or use the same datapoints as before? (This is slower.)
            if recalculate_randomness:
                noisy_spectra = None # calculate it in the helper function

            ## the list of desired frequencies will grow by one element for each loop
            desiredfreqs = freqlist[:this_num_p]

            if verbose:
                print('num freq: ' + str(this_num_p))
                print('desired freqs: ' + str(desiredfreqs))

            p = freqpoints(desiredfreqs = desiredfreqs, drive = drive)

            thisres, plot_info_1D = simulated_experiment(drive[p], drive=drive,vals_set = vals_set, noiselevel=noiselevel, MONOMER=MONOMER, 
                                           repeats=1 , verbose = verbose, forceboth=forceboth,labelcounts = False,
                                           noiseless_spectra=noiseless_spectra, noisy_spectra = noisy_spectra, 
                                           return_1D_plot_info = True,
                                           **kwargs
                                           )
                
            
            try: # repeated experiments results
                resultsdf = pd.concat([resultsdf,thisres], ignore_index=True)
            except:
                resultsdf = thisres


    return resultsdf, plot_info_1D