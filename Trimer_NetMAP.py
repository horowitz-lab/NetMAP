#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:57:10 2024

@author: samfeldman
"""
import numpy as np
#     return amp(Z.real, Z.imag)
from Trimer_simulator import calculate_spectra

''' THIS IS THE NETMAP PART 
    Note that we do not deal with noise in any way '''

def Zmatrix(force_all, freq, complexamp1, complexamp2, complexamp3):
    Zmatrix = []
    for rowindex in range(len(freq)):
        w = freq[rowindex]
        Z1 = complexamp1[rowindex]
        Z2 = complexamp2[rowindex]
        Z3 = complexamp3[rowindex]
        
        Zmatrix.append([-w**2*np.real(Z1), 0, 0, -w*np.imag(Z1), 0, 0, np.real(Z1), 
                        np.real(Z1)-np.real(Z2), 0, -1])
        Zmatrix.append([-w**2*np.imag(Z1), 0, 0, w*np.real(Z1), 0, 0, np.imag(Z1), 
                          np.imag(Z1) - np.imag(Z2), 0, 0])

        if force_all:
            Zmatrix.append([0, -w**2*np.real(Z2), 0, 0, -w*np.imag(Z2), 0, 0, 
                            np.real(Z2)-np.real(Z1), np.real(Z2) - np.real(Z3), -1])
        else:
            Zmatrix.append([0, -w**2*np.real(Z2), 0, 0, -w*np.imag(Z2), 0, 0, 
                            np.real(Z2)-np.real(Z1), np.real(Z2) - np.real(Z3), 0])
            
        Zmatrix.append([0, -w**2*np.imag(Z2), 0, 0, w*np.real(Z2), 0, 0, 
                        np.imag(Z2)-np.imag(Z1), np.imag(Z2) - np.imag(Z3), 0])
        
        if force_all:
            Zmatrix.append([0, 0, -w**2*np.real(Z3), 0, 0, -w*np.imag(Z3), 0, 0, 
                            np.real(Z3)-np.real(Z2), -1])     
        else:
            Zmatrix.append([0, 0, -w**2*np.real(Z3), 0, 0, -w*np.imag(Z3), 0, 0, 
                            np.real(Z3)-np.real(Z2), 0])
        
        Zmatrix.append([0, 0, -w**2*np.imag(Z3), 0, 0, w*np.real(Z3), 0, 0, 
                        np.imag(Z3)-np.imag(Z2), 0])
        
    return np.array(Zmatrix)

def unnormalizedparameters(Zmatrix):
    U, S, Vh = np.linalg.svd(Zmatrix)
    V = Vh.conj().T
    return V[:,-1] #Will it always be the last column of V??

def normalize_parameters_1d_by_force(unnormalizedparameters, F_set):
    # parameters vector: 'm1', 'm2', 'b1', 'b2', 'k1', 'k2','k12', 'Driving Force'
    c = F_set / unnormalizedparameters[-1]
    parameters = [c*unnormalizedparameters[k] for k in range(len(unnormalizedparameters)) ]
    return parameters

#This is the data for NetMAP to work with. Using the same data as Sam in thesis

f1 = 1.7
f2 = 2.3
m1 = 3
m2 = 3
m3 = 3
b1 = 0.1
b2 = 0.1
b3 = 0.1
k1 = 5
k2 = 5
k3 = 5
k4 = 0 #no fourth spring connecting mass 4 to wall in this
F = 1

# getting the complex amplitudes with a function from Trimer_simulator
Z11, Z21, Z31 = calculate_spectra(f1, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3, 0, False)
Z12, Z22, Z32 = calculate_spectra(f2, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3, 0, False)

frequencies = [f1, f2]
comamps1 = [Z11, Z12]
comamps2 = [Z21, Z22]
comamps3 = [Z31, Z32]
#these are good

#Create the Zmatrix:
trizmatrix = Zmatrix(False, frequencies, comamps1, comamps2, comamps3)

#Get the unnormalized parameters:
notnormparam_tri = unnormalizedparameters(trizmatrix)

#Normalize the parameters
final_tri = normalize_parameters_1d_by_force(notnormparam_tri, 1)

print(final_tri)
# it works! finally!
