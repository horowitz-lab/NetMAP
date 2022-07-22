"""
File created 2022-07-22 by Viva Horowitz
"""

import math
import numpy as np

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

def quadratic_formula(a, b, c):
    return (-b + math.sqrt(b*b - 4*a*c))/(2*a), (-b - math.sqrt(b*b - 4*a*c))/(2*a)

"""Zmatrix2resonators(df) will return a matrix for svd for any number of frequency measurements, 
listed in each row of the dataframe measurementdf 
If forceboth is true then both masses receive a force.
parameternames = ['m1', 'm2', 'b1', 'b2', 'k1', 'k2','c12', 'Driving Force']
"""
def Zmatrix2resonators(measurementdf, forceboth,
                       frequencycolumn = 'drive', 
                       complexamplitude1 = 'R1AmpCom', complexamplitude2 = 'R2AmpCom',  dtype=complex):
    Zmatrix = []
    for rowindex in measurementdf.index:
        w = measurementdf[frequencycolumn][rowindex]
        #print(w)
        ZZ1 = measurementdf[complexamplitude1][rowindex]
        ZZ2 = measurementdf[complexamplitude2][rowindex]
        Zmatrix.append([-w**2*np.real(ZZ1), 0, -w*np.imag(ZZ1), 0, np.real(ZZ1), 0, np.real(ZZ1)-np.real(ZZ2), -1])
        Zmatrix.append([-w**2*np.imag(ZZ1), 0, w*np.real(ZZ1), 0, np.imag(ZZ1), 0, np.imag(ZZ1)-np.imag(ZZ2), 0])
        if forceboth:
            Zmatrix.append([0, -w**2*np.real(ZZ2), 0, -w*np.imag(ZZ2), 0, np.real(ZZ2), np.real(ZZ2)-np.real(ZZ1), -1])
        else:
            Zmatrix.append([0, -w**2*np.real(ZZ2), 0, -w*np.imag(ZZ2), 0, np.real(ZZ2), np.real(ZZ2)-np.real(ZZ1), 0])
        Zmatrix.append([0, -w**2*np.imag(ZZ2), 0, w*np.real(ZZ2), 0, np.imag(ZZ2), np.imag(ZZ2)-np.imag(ZZ1), 0])
    #display(Zmatrix)
    return np.array(Zmatrix, dtype=dtype)

"""ZmatrixMONOMER(df) will return a matrix for svd for any number of frequency measurements, 
listed in each row of the dataframe measurementdf 
parameternames = ['m1', 'b1', 'k1', 'Driving Force']
"""
def ZmatrixMONOMER(measurementdf, 
                       frequencycolumn = 'drive', 
                       complexamplitude1 = 'R1AmpCom', dtype=complex):
    Zmatrix = []
    for rowindex in measurementdf.index:
        w = measurementdf[frequencycolumn][rowindex]
        #print(w)
        ZZ1 = measurementdf[complexamplitude1][rowindex]
        Zmatrix.append([-w**2*np.real(ZZ1), -w*np.imag(ZZ1), np.real(ZZ1), -1])
        Zmatrix.append([-w**2*np.imag(ZZ1),  w*np.real(ZZ1), np.imag(ZZ1),  0])
    #display(Zmatrix)
    return np.array(Zmatrix, dtype=dtype)

def Zmat(measurementdf, MONOMER, forceboth,
         frequencycolumn = 'drive', complexamplitude1 = 'R1AmpCom', complexamplitude2 = 'R2AmpCom', dtype=complex, 
         ):
    if MONOMER:
        return ZmatrixMONOMER(measurementdf=measurementdf, 
                       frequencycolumn = frequencycolumn, # can't force both when there aren't two masses
                       complexamplitude1 =complexamplitude1, dtype=dtype)
    else:
        return Zmatrix2resonators(measurementdf=measurementdf, 
                       frequencycolumn = frequencycolumn, forceboth = forceboth,
                       complexamplitude1 = complexamplitude1, complexamplitude2 = complexamplitude2, dtype=dtype)
