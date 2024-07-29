# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:07:09 2022

@author: vhorowit
"""

''' Changes by lydiabullock. Adapting resonatorphysics to work for the Trimer functions and parameters.
    I believe I only changed line 49 for now. '''

from helperfunctions import read_params
import numpy as np
import math


def complexamp(A,phi):
    return A * np.exp(1j*phi)

def amp(a,b):
    return np.sqrt(a**2 + b**2)

def A_from_Z(Z): # calculate amplitude of complex number
    return amp(Z.real, Z.imag)

# For driven, damped oscillator: res_freq = sqrt(k/m - b^2/(2m^2))
# Note: Requires b < sqrt(2mk) to be significantly underdamped
# Otherwise there is no resonant frequency and we get an err from the negative number under the square root
# This works for monomer and for weak coupling. It does not work for strong coupling.
# Uses privilege. See also res_freq_numeric()
def res_freq_weak_coupling(k, m, b):
    try:
        w = math.sqrt(k/m - (b*b)/(2*m*m))
    except:
        w = np.nan
        print('no resonance frequency for k=', k, ', m=', m, ' b=', b)
    return w

## source: https://en.wikipedia.org/wiki/Q_factor#Mechanical_systems
# Does not work for strong coupling.
def approx_Q(k, m, b):
    return math.sqrt(m*k)/b

# Approximate width of Lorentzian peak.
# Does not work for strong coupling.
def approx_width(k, m, b):
    return res_freq_weak_coupling(k, m, b) / approx_Q(k, m, b)

def calcnarrowerW(vals_set, MONOMER):
    [k1_set, k2_set, k3_set, k4_set, b1_set, b2_set, b3_set, F_set, m1_set, m2_set, m3_set] = read_params(vals_set, MONOMER)
    W1=approx_width(k1_set, m1_set, b1_set)
    if MONOMER:
        narrowerW = W1
    else:
        W2=approx_width(k2_set, m2_set, b2_set)
        narrowerW = min(W1,W2)
    return narrowerW



