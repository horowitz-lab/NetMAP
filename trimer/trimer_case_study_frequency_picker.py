#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:55:10 2024

@author: lydiabullock
"""
''' Case Study for System 0 from '15 Systems - 10 Freqs NetMAP' 
    Using "ideal" frequencies to test NetMAP. '''

from comparing_curvefit_types import run_trials
from kind_of_trimer_resonatorfrequencypicker import res_freq_numeric


''' Begin Work Here. '''

## System 0 from '15 Systems - 10 Freqs NetMAP'
true_parameters = [1.045, 0.179, 3.852, 1.877, 5.542, 1.956, 3.71, 1, 3.976, 0.656, 3.198]
guessed_parameters = [1.2379, 0.1764, 3.7327, 1.8628, 5.93, 2.1793, 4.2198, 1, 4.3335, 0.7016, 3.0719]

MONOMER = False
forceall = False

best_frequencies_list = res_freq_numeric(true_parameters, MONOMER, forceall)
print(best_frequencies_list)


# avg_e1_list, avg_e2_list, avg_e3_list, avg_e1_bar, avg_e2_bar, avg_e3_bar = run_trials(true_parameters, guessed_parameters, 50, 'Sys0_Freq_Pick.xlsx', 'Sys0_Freq_Pick - Plots')