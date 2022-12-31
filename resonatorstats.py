# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:08:13 2022

@author: vhorowit
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

def syserr(x_found,x_set, absval = True):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        se = 100*(x_found-x_set)/x_set
    if absval:
        return abs(se)
    else:
        return se


def combinedsyserr(syserrs, notdof): # notdof = not degrees of freedom, meaning the count of fixed parameters.
    assert len(syserrs) > 0
    assert notdof > 0
    abssyserrs = [abs(err) for err in syserrs]
    dof = len(syserrs) - notdof # fixed parameters have a systematic err of zero; don't count them as free parameters.
    assert dof > 0
    squared = [(err**2) for err in syserrs]
    rms = np.sqrt(sum(squared) / dof)
    avg = sum(abssyserrs)/ dof
    
    syserrs = [i for _,i in sorted(zip(abssyserrs,syserrs))]
    while notdof >0:
        syserrs.pop(0) # remove small values
        notdof = notdof -1
    assert len(syserrs) == dof
    if False: # never mind. This is too slow and uninteresting.
        try:
            from decimal import Decimal
            Lavg =  np.log10(sum([10**(Decimal(err)) for err in syserrs])/dof)
        except: # probably an Overflow
            Lavg = np.nan
    else:
        Lavg = np.nan
   
    return avg, rms, max(abssyserrs), Lavg

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
