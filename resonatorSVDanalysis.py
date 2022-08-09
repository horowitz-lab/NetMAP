# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:50:38 2022

@author: vhorowit
"""


import numpy as np

"""Zmatrix2resonators(df) will return a matrix for svd for any number of frequency measurements, 
listed in each row of the dataframe measurementdf 
If forceboth is true then both masses receive a force.
parameternames = ['m1', 'm2', 'b1', 'b2', 'k1', 'k2','c12', 'Driving Force']
"""
def Zmatrix2resonators(measurementdf, forceboth,
                       frequencycolumn = 'drive', 
                       complexamplitude1 = 'R1AmpCom', complexamplitude2 = 'R2AmpCom', dtype=complex):
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
        return Zmatrix2resonators(measurementdf=measurementdf, forceboth = forceboth,
                       frequencycolumn = frequencycolumn, 
                       complexamplitude1 = complexamplitude1, complexamplitude2 = complexamplitude2, dtype=dtype)

    