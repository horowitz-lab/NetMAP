# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Create code that simulates spectrum response for trimer
# See if we can recover the parameters

import numpy as np
import sympy as sp
# from helperfunctions import read_params, listlength
# from resonatorphysics import amp, complexamp
# import matplotlib.pyplot as plt
# from resonatorstats import rsqrd
# from resonatorphysics import A_from_Z

# defaults
usenoise = True
use_complexnoise = True

#Define all variables for sympy

#individual springs that correspond to individual masses
k1 = sp.symbols('k_1', real = True)

#springs that connect two masses
k2 = sp.symbols('k_2', real = True)
k3 = sp.symbols('k_3', real = True)
k4 = sp.symbols('k_4', real = True)

#damping coefficients
b1 = sp.symbols('b1', real = True)
b2 = sp.symbols('b2', real = True)
b3 = sp.symbols('b3', real = True)
 
#masses
m1 = sp.symbols('m1', real = True)
m2 = sp.symbols('m2', real = True)
m3 = sp.symbols('m3', real = True)

#Driving force amplitude
F = sp.symbols('F', real = True)

#driving frequency (leave as variable)
wd = sp.symbols('\omega_d', real = True)

#symbolically Solve for driving amplitudes and phase using sympy

### Dimer
#Matrix for complex equations of motion, Matrix . Zvec = Fvec
unknownsmatrix = sp.Matrix([[-wd**2*m1 + 1j*wd*b1 + k1 + k2 + k4, -k2, -k4], 
                            [-k2, -wd**2*m2 + 1j*wd*b2 + k2 + k3, -k3],
                            [-k4, -k3, -wd**2*m3 + 1j*wd*b3 + k3 + k4]])

#Matrices for Cramer's Rule: substitute force vector Fvec=[F,0] for each column in turn (m1 is driven, m2 is not)
unknownsmatrix1 = sp.Matrix([[F, -k2, -k4], 
                             [0, -wd**2*m2 + 1j*wd*b2 + k2 + k3, -k3],
                             [0, -k3, -wd**2*m3 + 1j*wd*b3 + k3 + k4]])
unknownsmatrix2 = sp.Matrix([[-wd**2*m1 + 1j*wd*b1 + k1 + k2 + k4, F, -k4], 
                             [-k2, 0, -k3],
                             [-k4, 0, -wd**2*m3 + 1j*wd*b3 + k3 + k4]])
unknownsmatrix3 = sp.Matrix([[-wd**2*m1 + 1j*wd*b1 + k1 + k2 + k4, -k3, F], 
                             [-k2, -wd**2*m2 + 1j*wd*b2 + k2 + k3, 0],
                             [-k4, -k3, 0]])

#Apply Cramer's Rule to solve for Zvec
complexamp1, complexamp2, complexamp3 = (unknownsmatrix1.det()/unknownsmatrix.det(), 
                                         unknownsmatrix2.det()/unknownsmatrix.det(),
                                         unknownsmatrix3.det()/unknownsmatrix.det())

#Solve for phases for each mass
delta1 = sp.arg(complexamp1) # Returns the argument (phase angle in radians) of a complex number. 
delta2 = sp.arg(complexamp2) # sp.re(complexamp2)/sp.cos(delta2) (this is the same thing)
delta3 = sp.arg(complexamp3)


### What if we apply the same force to all three masses of dimer?
#Matrices for Cramer's Rule: substitute force vector Fvec=[F,0] for each column in turn (m1 is driven, m2 is not)
unknownsmatrix1FFF = sp.Matrix([[F, -k2, -k4], 
                             [F, -wd**2*m2 + 1j*wd*b2 + k2 + k3, -k3],
                             [F, -k3, -wd**2*m3 + 1j*wd*b3 + k3 + k4]])
unknownsmatrix2FFF = sp.Matrix([[-wd**2*m1 + 1j*wd*b1 + k1 + k2 + k4, F, -k4], 
                             [-k2, F, -k3],
                             [-k4, F, -wd**2*m3 + 1j*wd*b3 + k3 + k4]])
unknownsmatrix3FFF = sp.Matrix([[-wd**2*m1 + 1j*wd*b1 + k1 + k2 + k4, -k3, F], 
                             [-k2, -wd**2*m2 + 1j*wd*b2 + k2 + k3,F],
                             [-k4, -k3, F]])
#Apply Cramer's Rule to solve for Zvec
complexamp1FFF, complexamp2FFF, complexamp3FFF = (unknownsmatrix1FFF.det()/unknownsmatrix.det(), 
                                unknownsmatrix2FFF.det()/unknownsmatrix.det(),
                                unknownsmatrix3FFF.det()/unknownsmatrix.det())
#Solve for phases for each mass
delta1FFF = sp.arg(complexamp1FFF) # Returns the argument (phase angle in radians) of a complex number. 
delta2FFF = sp.arg(complexamp2FFF) # sp.re(complexamp2)/sp.cos(delta2) (this is the same thing)
delta3FFF = sp.arg(complexamp3FFF)

### Ampolitude and phase
#Wrap phases for plots

wrap1 = (delta1)%(2*sp.pi)
wrap2 = (delta2)%(2*sp.pi)
wrap3 = (delta3)%(2*sp.pi)
wrap1FFF = (delta1FFF)%(2*sp.pi)
wrap2FFF = (delta2FFF)%(2*sp.pi)
wrap3FFF = (delta3FFF)%(2*sp.pi)

#Solve for amplitude coefficients
amp1 = sp.Abs(complexamp1)
amp2 = sp.Abs(complexamp2)
amp3 = sp.Abs(complexamp3)
amp1FFF = sp.Abs(complexamp1FFF)
amp2FFF = sp.Abs(complexamp2FFF)
amp3FFF = sp.Abs(complexamp3FFF)

#lambdify curves using sympy

c1 = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), amp1)
t1 = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), wrap1)

c2 = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), amp2)
t2 = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), wrap2)

c3 = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), amp3)
t3 = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), wrap3)

re1 = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), sp.re(complexamp1))
im1 = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), sp.im(complexamp1))
re2 = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), sp.re(complexamp2))
im2 = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), sp.im(complexamp2))
re3 = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), sp.re(complexamp3))
im3 = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), sp.im(complexamp3))


c1FFF = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), amp1FFF)
t1FFF = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), wrap1FFF)

c2FFF = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), amp2FFF)
t2FFF = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), wrap2FFF)

c3FFF = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), amp3FFF)
t3FFF = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), wrap3FFF)

re1FFF = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), sp.re(complexamp1FFF))
im1FFF = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), sp.im(complexamp1FFF))
re2FFF = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), sp.re(complexamp2FFF))
im2FFF = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), sp.im(complexamp2FFF))
re3FFF = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), sp.re(complexamp3FFF))
im3FFF = sp.lambdify((wd, k1, k2, k3, k4, b1, b2, b3, F, m1, m2, m3), sp.im(complexamp3FFF))

#define functions

#curve = amplitude, theta = phase, e = err (i.e. noise)
def curve1(w, k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3, e, force_all):
    with np.errstate(divide='ignore'):
        if force_all:
            return c1FFF(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) + e
        else: #force just m1
            return c1(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) + e
    
def theta1(w, k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3, e, force_all):
    with np.errstate(divide='ignore'):
        if force_all:
            return t1FFF(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) - 2*np.pi + e
        else: #force just m1
            return t1(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) - 2*np.pi + e

def curve2(w, k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3, e, force_all):
    with np.errstate(divide='ignore'):
        if force_all:
            return c2FFF(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) + e
        else: #force just m1
            return c2(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) + e
    
def theta2(w, k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3, e, force_all):
    with np.errstate(divide='ignore'):
        if force_all:
            return t2FFF(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) - 2*np.pi + e
        else: #force just m1
            return t2(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) - 2*np.pi + e

def curve3(w, k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3, e, force_all):
    with np.errstate(divide='ignore'):
        if force_all:
            return c3FFF(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) + e
        else: #force just m1
            return c3(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) + e
    
def theta3(w, k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3, e, force_all):
    with np.errstate(divide='ignore'):
        if force_all:
            return t3FFF(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) - 2*np.pi + e
        else: #force just m1
            return t3(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) - 2*np.pi + e

def realamp1(w, k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3, e, force_all):
    with np.errstate(divide='ignore'):
        if force_all:
            return re1FFF(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) + e
        else: #force just m1
            return re1(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) + e
    
def imamp1(w, k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3, e, force_all):
    with np.errstate(divide='ignore'):
        if force_all:
            return im1FFF(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) - 2*np.pi + e
        else: #force just m1
            return im1(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) - 2*np.pi + e

def realamp2(w, k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3, e, force_all):
    with np.errstate(divide='ignore'):
        if force_all:
            return re2FFF(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) + e
        else: #force just m1
            return re2(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) + e
    
def imamp2(w, k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3, e, force_all):
    with np.errstate(divide='ignore'):
        if force_all:
            return im2FFF(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) - 2*np.pi + e
        else: #force just m1
            return im2(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) - 2*np.pi + e

def realamp3(w, k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3, e, force_all):
    with np.errstate(divide='ignore'):
        if force_all:
            return re3FFF(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) + e
        else: #force just m1
            return re3(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) + e
    
def imamp3(w, k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3, e, force_all):
    with np.errstate(divide='ignore'):
        if force_all:
            return im3FFF(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) - 2*np.pi + e
        else: #force just m1
            return im3(np.array(w), k_1, k_2, k_3, k_4, b1_, b2_, b_3, F_, m_1, m_2, m_3) - 2*np.pi + e


def complexamp(A,phi):
    return A * np.exp(1j*phi)

def amp(a,b):
    return np.sqrt(a**2 + b**2)

def A_from_Z(Z): # calculate amplitude of complex number
    return amp(Z.real, Z.imag)

freq = np.linspace(.01,5,500)
Z1 = (complexamp(curve1(freq, 1,2,3,4,.5,.5,.5, 1, 2, 3, 4, 0 , False), theta1(freq, 1,2,3,4,.5,.5,.5, 1, 2, 3, 4, 0 , False)))
Z2 = (complexamp(curve2(freq, 1,2,3,4,.5,.5,.5, 1, 2, 3, 4, 0 , False), theta2(freq, 1,2,3,4,.5,.5,.5, 1, 2, 3, 4, 0 , False)))
Z3 = (complexamp(curve3(freq, 1,2,3,4,.5,.5,.5, 1, 2, 3, 4, 0 , False), theta3(freq, 1,2,3,4,.5,.5,.5, 1, 2, 3, 4, 0 , False)))

def Zmatrix(res_list, force_all, freq, complexamp1, complexamp2, complexamp3):
    for i in range(res_list):
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


