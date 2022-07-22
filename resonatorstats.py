"""
File created 2022-07-22 by Viva Horowitz
"""

import numpy as np
import matplotlib.pyplot as plt

def syserr(x_found,x_set):
    return 100*abs(x_found-x_set)/x_set

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
    Lavg =  np.log10(sum([10**(Decimal(err)) for err in syserrs])/dof)
   
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
	
