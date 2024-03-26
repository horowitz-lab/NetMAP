#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:57:10 2024

@author: samfeldman
"""
import numpy as np

def Zmatrix(array, force_all, freq, complexamp1, complexamp2, complexamp3):
    result = []
    for rowindex in range(len(array)):
        w = freq[rowindex]
        Z1 = complexamp1[rowindex]
        Z2 = complexamp2[rowindex]
        Z3 = complexamp3[rowindex]
        
        Z1R = np.array([-w**2*np.real(Z1), 0, 0, -w*np.imag(Z1), 0, 0, np.real(Z1), 
                        0, np.real(Z1)-np.real(Z2), 0, np.real(Z1)-np.real(Z3), -1])
        Z1I = np.array([-w**2*np.imag(Z1), 0, 0, w*np.real(Z1), 0, 0, np.imag(Z1), 
                         np.imag(Z1) - np.imag(Z2), 0, np.imag(Z1)-np.imag(Z3), 0])
        
        if force_all:
            Z2R = np.array([0, -w**2*np.real(Z2), 0, 0, -w*np.imag(Z2), 0, 0, 
                            np.real(Z2)-np.real(Z1), np.real(Z2) - np.real(Z3), 0, -1])        
        else:
            Z2R = np.array([0, -w**2*np.real(Z1), 0, 0, -w*np.imag(Z2), 0, 0, 
                            np.real(Z2)-np.real(Z1), np.real(Z2) - np.real(Z3), 0, 0])
        Z2I = np.array([0, -w**2*np.imag(Z2), 0, 0, w*np.real(Z2), 0, 0, 
                        np.imag(Z2)-np.imag(Z1), np.imag(Z2) - np.imag(Z3), 0, 0])
        
        if force_all:
            Z3R = np.array([0, 0, -w**2*np.real(Z3), 0, 0, -w*np.imag(Z3), 0, 0, 
                            np.real(Z3)-np.real(Z2), np.real(Z3) - np.real(Z1), -1])        
        else:
            Z3R = np.array([0, 0, -w**2*np.real(Z3), 0, 0, -w*np.imag(Z3), 0, 0, 
                            np.real(Z3)-np.real(Z2), np.real(Z3) - np.real(Z1), 0])
        Z3I = np.array([0, 0, -w**2*np.imag(Z3), 0, 0, w*np.real(Z3), 0, 0, 
                        np.imag(Z3)-np.imag(Z2), np.imag(Z3) - np.imag(Z1), -1])
        
        result.append(np.concatenate([Z1R, Z1I, Z2R, Z2I, Z3R, Z3I]))
    
    Zmatrix = np.array(result)
    return Zmatrix



