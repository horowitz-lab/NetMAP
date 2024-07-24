#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:57:10 2024

@author: samfeldman & lydiabullock
"""
import numpy as np
#     return amp(Z.real, Z.imag)
from Trimer_simulator import calculate_spectra

''' THIS IS THE NETMAP PART '''

def Zmatrix(freq, complexamp1, complexamp2, complexamp3, force_all):
    Zmatrix = []
    for rowindex in range(len(freq)):
        w = freq[rowindex]
        Z1 = complexamp1[rowindex]
        Z2 = complexamp2[rowindex]
        Z3 = complexamp3[rowindex]
        
        Zmatrix.append([-w**2*np.real(Z1), 0, 0, -w*np.imag(Z1), 0, 0, np.real(Z1), 
                        np.real(Z1)-np.real(Z2), 0, 0, -1])
        Zmatrix.append([-w**2*np.imag(Z1), 0, 0, w*np.real(Z1), 0, 0, np.imag(Z1), 
                          np.imag(Z1) - np.imag(Z2), 0, 0, 0])

        if force_all:
            Zmatrix.append([0, -w**2*np.real(Z2), 0, 0, -w*np.imag(Z2), 0, 0, 
                            np.real(Z2)-np.real(Z1), np.real(Z2) - np.real(Z3), 0, -1])
        else:
            Zmatrix.append([0, -w**2*np.real(Z2), 0, 0, -w*np.imag(Z2), 0, 0, 
                            np.real(Z2)-np.real(Z1), np.real(Z2) - np.real(Z3), 0, 0])
            
        Zmatrix.append([0, -w**2*np.imag(Z2), 0, 0, w*np.real(Z2), 0, 0, 
                        np.imag(Z2)-np.imag(Z1), np.imag(Z2) - np.imag(Z3), 0, 0])
        
        if force_all:
            Zmatrix.append([0, 0, -w**2*np.real(Z3), 0, 0, -w*np.imag(Z3), 0, 0, 
                            np.real(Z3)-np.real(Z2), np.real(Z3), -1])     
        else:
            Zmatrix.append([0, 0, -w**2*np.real(Z3), 0, 0, -w*np.imag(Z3), 0, 0, 
                            np.real(Z3)-np.real(Z2), np.real(Z3), 0])
        
        Zmatrix.append([0, 0, -w**2*np.imag(Z3), 0, 0, w*np.real(Z3), 0, 0, 
                        np.imag(Z3)-np.imag(Z2), np.imag(Z3), 0])
        
    return np.array(Zmatrix)

def unnormalizedparameters(Zmatrix):
    U, S, Vh = np.linalg.svd(Zmatrix)
    V = Vh.conj().T
    return V[:,-1] #Will it always be the last column of V??

def normalize_parameters_1d_by_force(unnormalizedparameters, F_set):
    # parameters vector: 'm1', 'm2', 'm3', 'b1', 'b2', 'b3', 'k1', 'k2', 'k3', 'k4', 'Driving Force'
    c = F_set / unnormalizedparameters[-1]
    parameters = [c*unnormalizedparameters[k] for k in range(len(unnormalizedparameters)) ]
    return parameters


''' Example work begins here. '''

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
k4 = 1 #no fourth spring connecting mass 4 to wall in this
F = 1

#create some noise
from resonatorsimulator import complex_noise
e = complex_noise(2, 2) #number of frequencies, noise level
frequencies = [f1, f2]

# getting the complex amplitudes with a function from Trimer_simulator
comamps1, comamps2, comamps3 = calculate_spectra(frequencies, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3, e, False)


#Create the Zmatrix:
trizmatrix = Zmatrix(frequencies, comamps1, comamps2, comamps3, False)

#Get the unnormalized parameters:
notnormparam_tri = unnormalizedparameters(trizmatrix)

#Normalize the parameters
final_tri = normalize_parameters_1d_by_force(notnormparam_tri, 1)

print(final_tri)
# it works! finally!
