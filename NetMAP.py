# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:50:38 2022

NetMAP: Create script-Z matrix and find its kernel, or null-space.

@author: vhorowit
"""

import numpy as np
import math
from resonatorphysics import res_freq_weak_coupling
from helperfunctions import read_params
from resonatorfrequencypicker import res_freq_numeric

"""Zmatrix2resonators(df,...) will return a matrix for svd for any number of frequency measurements, 
listed in each row of the dataframe measurementdf 
If forceboth is true then both masses receive a force.
parameternames = ['m1', 'm2', 'b1', 'b2', 'k1', 'k2','k12', 'Driving Force']
"""

def Zmatrix2resonators(measurementdf, forceboth,
                       frequencycolumn = 'drive', 
                       complexamplitude1 = 'R1AmpCom', complexamplitude2 = 'R2AmpCom', dtype=complex):

    ## Are both masses being pushed? or just the first?
    if forceboth:
        ff = -1
    else:
        ff = 0
        
    Zmatrix = []   # this would likely be more efficient as a numpy array.
    for rowindex in measurementdf.index:
        w = measurementdf[frequencycolumn][rowindex]
        #print(w)
        ZZ1 = measurementdf[complexamplitude1][rowindex]
        ZZ2 = measurementdf[complexamplitude2][rowindex]
        # Matrix columns: m1, m2, b1, b2, k1, k2, k12, F1
        Zmatrix.append([-w**2*np.real(ZZ1), 0, -w*np.imag(ZZ1), 0, np.real(ZZ1), 0, np.real(ZZ1)-np.real(ZZ2), -1])
        Zmatrix.append([-w**2*np.imag(ZZ1), 0, w*np.real(ZZ1), 0, np.imag(ZZ1), 0, np.imag(ZZ1)-np.imag(ZZ2), 0])
        Zmatrix.append([0, -w**2*np.real(ZZ2), 0, -w*np.imag(ZZ2), 0, np.real(ZZ2), np.real(ZZ2)-np.real(ZZ1), ff])
        Zmatrix.append([0, -w**2*np.imag(ZZ2), 0, w*np.real(ZZ2), 0, np.imag(ZZ2), np.imag(ZZ2)-np.imag(ZZ1), 0])
    #display(Zmatrix)
    return np.matrix(Zmatrix, dtype=dtype)

"""ZmatrixMONOMER(df,...) will return a matrix for svd for any number of frequency measurements, 
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
        # Matrix columns: m, b, k, F
        Zmatrix.append([-w**2*np.real(ZZ1), -w*np.imag(ZZ1), np.real(ZZ1), -1])
        Zmatrix.append([-w**2*np.imag(ZZ1),  w*np.real(ZZ1), np.imag(ZZ1),  0])
    #display(Zmatrix)
    return np.matrix(Zmatrix, dtype=dtype)

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

### Normalizations

""" 1d nullspace normalization """
def normalize_parameters_1d_by_force(unnormalizedparameters, F_set):
    # parameters vector: 'm1', 'm2', 'b1', 'b2', 'k1', 'k2','k12', 'Driving Force'
    c = F_set / unnormalizedparameters[-1]
    parameters = [c*unnormalizedparameters[k] for k in range(len(unnormalizedparameters)) ]
    return parameters

def quadratic_formula(a, b, c):
    return (-b + math.sqrt(b*b - 4*a*c))/(2*a), (-b - math.sqrt(b*b - 4*a*c))/(2*a)

"""2D normalizations"""

def normalize_parameters_to_res1_and_F_2d(vh, vals_set, MONOMER, privilege = False):
    
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
    
    # parameters vector: 'm1', 'm2', 'b1', 'b2', 'k1', 'k2','c12', 'Driving Force'
    vect1 = vh[-1]
    m1_1, m2_1, b1_1, b2_1, k1_1, k2_1, c12_1, F_1 = read_params(vect1, MONOMER)
    vect2 = vh[-2]
    m1_2, m2_2, b1_2, b2_2, k1_2, k2_2, c12_2, F_2 = read_params(vect2, MONOMER)
    
    # Assume we know the resonant frequency of one of the driven, damped oscillators
    # We seem to get much more accurate results knowing the first one, perhaps because that one is being driven
    if privilege:
        res_freq1 = res_freq_weak_coupling(k1_set, m1_set, b1_set) ## This uses privileged information.
    else:
        res_freq1 = res_freq_numeric(numtoreturn = 1,  verboseplot = True, verbose=True)
    #res_freq2 = res_freq_weak_coupling(k2_set, m2_set, b2_set)
    
    # Subscript by 1 and 2 for the two null space vectors
    # Then in res_freq formula, substitute k -> k_1 + Rk_2, m -> m_1 + Rm_2, b -> b_1 + Rb_2
    # Solve for R, the weight of null vector 2 relative to null vector 1 (formula found using Mathematica)
    # The formula is quadratic, so we get two values of R for each oscillator
    # Pick the one that gives the correct (or closer to correct) resonating frequency
    # For simplicity, first solve for A, B, C, the coefficients of the quadratic equation
    
    osc1_A = -b1_2**2 + 2 * k1_2 * m1_2 - 2 * m1_2**2 * res_freq1**2
    osc1_B = -2 * b1_1 * b1_2 + 2 * k1_2 * m1_1 + 2 * k1_1 * m1_2 - 4 * m1_1 * m1_2 * res_freq1**2
    osc1_C = -b1_1**2 + 2 * k1_1 * m1_1 - 2 * m1_1**2 * res_freq1**2
    # If there's a ValueError, just do 1D
    osc1_R1, osc1_R2 = 0, 0
    try:
        osc1_R1, osc1_R2 = quadratic_formula(osc1_A, osc1_B, osc1_C)
    except ValueError:
        pass
    
    # There may be ValueErrors if there exists no resonating frequency for the incorrect R
    # In that case, we make the difference infinity so that R isn't chosen
    osc1_R1_diff = float('inf')
    osc1_R2_diff = float('inf')
    try:
        osc1_R1_diff = abs(res_freq_weak_coupling(k1_1 + osc1_R1 * k1_2, m1_1 + osc1_R1 * m1_2, b1_1 + osc1_R1 * b1_2) - res_freq1)
    except ValueError:
        pass
    try:
        osc1_R2_diff = abs(res_freq_weak_coupling(k1_1 + osc1_R2 * k1_2, m1_1 + osc1_R2 * m1_2, b1_1 + osc1_R2 * b1_2) - res_freq1)
    except ValueError:
        pass
    osc1_R = osc1_R1 if osc1_R1_diff < osc1_R2_diff else osc1_R2
    
    #osc2_A = -b2_2**2 + 2 * k2_2 * m2_2 - 2 * m2_2**2 * res_freq2**2
    #osc2_B = -2 * b2_1 * b2_2 + 2 * k2_2 * m2_1 + 2 * k2_1 * m2_2 - 4 * m2_1 * m2_2 * res_freq2**2
    #osc2_C = -b2_1**2 + 2 * k2_1 * m2_1 - 2 * m2_1**2 * res_freq2**2
    #osc2_R1, osc2_R2 = quadratic_formula(osc2_A, osc2_B, osc2_C)
    
    #osc2_R1_diff = float('inf')
    #osc2_R2_diff = float('inf')
    #try:
    #    osc2_R1_diff = abs(res_freq_weak_coupling(k2_1 + osc2_R1 * k2_2, m2_1 + osc2_R1 * m2_2, b2_1 + osc2_R1 * b2_2) - res_freq2)
    #except ValueError:
    #    pass
    #try:
    #    osc2_R2_diff = abs(res_freq_weak_coupling(k2_1 + osc2_R2 * k2_2, m2_1 + osc2_R2 * m2_2, b2_1 + osc2_R2 * b2_2) - res_freq2)
    #except ValueError:
    #    pass
    #osc2_R = osc2_R1 if osc2_R1_diff < osc2_R2_diff else osc2_R2
    
    # For testing purposes
    #calc_res_freq_1with1 = res_freq_weak_coupling(k1_1 + osc1_R * k1_2, m1_1 + osc1_R * m1_2, b1_1 + osc1_R * b1_2)
    #calc_res_freq_1with2 = res_freq_weak_coupling(k1_1 + osc2_R * k1_2, m1_1 + osc2_R * m1_2, b1_1 + osc2_R * b1_2)
    #calc_res_freq_2with1 = res_freq_weak_coupling(k2_1 + osc1_R * k2_2, m2_1 + osc1_R * m2_2, b2_1 + osc1_R * b2_2)
    #calc_res_freq_2with2 = res_freq_weak_coupling(k2_1 + osc2_R * k2_2, m2_1 + osc2_R * m2_2, b2_1 + osc2_R * b2_2)
    #print("Actual Oscillator 1 Resonant Frequency: " + str(res_freq1))
    #print("Calculated Oscillator 1with1 Resonant Frequency: " + str(calc_res_freq_1with1))
    #print("Calculated Oscillator 1with2 Resonant Frequency: " + str(calc_res_freq_1with2))
    #print("Actual Oscillator 2 Resonant Frequency: " + str(res_freq2))
    #print("Calculated Oscillator 2with1 Resonant Frequency: " + str(calc_res_freq_2with1))
    #print("Calculated Oscillator 2with2 Resonant Frequency: " + str(calc_res_freq_2with2))
    #print("Oscillator 1 Null Vector 1 Resonant Frequency: " + str(res_freq_weak_coupling(k1_1, m1_1, b1_1)))
    #print("Oscillator 1 Null Vector 2 Resonant Frequency: " + str(res_freq_weak_coupling(k1_2, m1_2, b1_2)))
    #print("Oscillator 2 Null Vector 1 Resonant Frequency: " + str(res_freq_weak_coupling(k2_1, m2_1, b2_1)))
    #print("Oscillator 2 Null Vector 2 Resonant Frequency: " + str(res_freq_weak_coupling(k2_2, m2_2, b2_2)))
    
    #print("Weight Ratio from Oscillator 1: " + str(osc1_R))
    #print("Weight Ratio from Oscillator 2: " + str(osc2_R))
    # The R from oscillator 1 seems to work much better, perhaps because it's the one being driven
    R = osc1_R

    # To find the overall weight, we just use the 1D case assuming we know the force
    vect1 = vh[-1]
    vect2 = vh[-2]
    parameters = [vect1[k] + R*vect2[k] for k in range(len(vect1))]
    return normalize_parameters_1d_by_force(parameters, F_set) # does not return the two coefficients


""" mass 1 and mass 2 normalization, 2D nullspace assumption """
# not great for monomer
def normalize_parameters_to_m1_m2_assuming_2d(vh,verbose, m1_set, m2_set):
    if verbose:
        print('Running normalize_parameters_to_m1_m2_assuming_2d()')
    
    # parameters vector: 'm1', 'm2', 'b1', 'b2', 'k1', 'k2','c12', 'Driving Force'
    vect1 = vh[-1]
    vect2 = vh[-2]
    
    if verbose:
        print("If the null-space is 2D, we must be able to independently determine two parameters; say it's m1 and m2.")

    # find linear combination such that:
    # a * vect1[0] + b * vect2[0] = m1_set   and
    # a * vect1[1] + b * vect2[1] = m2_set
    ## But this rearranges to:

    coefa = ( vect2[1] * m1_set - m2_set * vect2[0] ) / (vect2[1]*vect1[0] - vect1[1]*vect2[0] )
    coefb = (vect1[1]*m1_set - m2_set *vect1[0] ) /(vect1[1]*vect2[0] - vect2[1]*vect1[0] )

    parameters = [coefa*vect1[k]+coefb*vect2[k]  for k in range(len(vect1)) ]
    return parameters, coefa, coefb

def normalize_parameters_to_m1_set_k1_set_assuming_2d(vh, verbose, m1_set, k1_set, MONOMER):
    
    if verbose:
        print('Running normalize_parameters_to_m1_set_k1_set_assuming_2d()')
    
    # parameters vector: 'm1', 'm2', 'b1', 'b2', 'k1', 'k2','c12', 'Driving Force'
    vect1 = vh[-1]
    vect2 = vh[-2]
    
    if verbose:
        print("If the null-space is 2D, we must be able to independently determine two parameters; say it's m1 and k1.")

    indexm1 = 0
    if MONOMER:
        indexk1 = 2
    else:
        indexk1 = 4
        
    # find linear combination such that:
    # a * vect1[0] + b * vect2[0] = m1_set   and
    # a * vect1[4] + b * vect2[4] = k1_set
    ## But this rearranges to:

    coefa = ( vect2[indexk1] * m1_set - k1_set * vect2[indexm1] ) / \
        (vect2[indexk1]*vect1[indexm1] - vect1[indexk1]*vect2[indexm1] )
    coefb = ( vect1[indexk1] * m1_set - k1_set * vect1[indexm1] ) / \
        (vect1[indexk1]*vect2[indexm1] - vect2[indexk1]*vect1[indexm1] )
    
    if verbose:
        print(str(coefa) + ' of last singular vector and ' + str(coefb) + ' of second to last singular vector.')

    parameters = [coefa*vect1[k]+coefb*vect2[k]  for k in range(len(vect1)) ]
    return parameters, coefa, coefb

def normalize_parameters_to_m1_F_set_assuming_2d(vh, MONOMER, verbose, m1_set, F_set):
    if verbose:
        print('Running normalize_parameters_to_m1_F_set_assuming_2d()')
    
    # parameters vector: 'm1', 'm2', 'b1', 'b2', 'k1', 'k2','c12', 'Driving Force'
    # parameters vector: 'm1', 'b1',  'k1', 'Driving Force'
    vect1 = vh[-1]
    vect2 = vh[-2]
    
    if verbose:
        print("If the null-space is 2D, we must be able to independently determine two parameters; say it's m1 and driving force.")

    indexm1 = 0
    if MONOMER:
        indexF = 3
    else:
        indexF = 7

    # find linear combination such that:
    # a * vect1[0] + b * vect2[0] = m1_set   and
    # a * vect1[7] + b * vect2[7] = F_set
    ## But this rearranges to: 

    coefa = ( vect2[indexF] * m1_set - F_set * vect2[indexm1] ) / \
        (vect2[indexF]*vect1[indexm1] - vect1[indexF]*vect2[indexm1] )
    coefb = ( vect1[indexF] * m1_set - F_set * vect1[indexm1] ) / \
        (vect1[indexF]*vect2[indexm1] - vect2[indexF]*vect1[indexm1] )
    
    if verbose:
        print(str(coefa) + ' of last singular vector and ' + str(coefb) + ' of second to last singular vector.')

    parameters = [coefa*vect1[k]+coefb*vect2[k]  for k in range(len(vect1)) ]
    return parameters, coefa, coefb


""" Force, mass 1 and mass 2 normalization, 3D nullspace assumption
    Numerics corresponding to m1, m2, F in the parameters vector:
    known1 = 0
    known2 = 1
    known3 = 7.
    For monomer, recommend choosing 0,2,3 because damping is the hardest to know experimentally """
def normalize_parameters_assuming_3d(vh, vals_set, MONOMER, known1 = None, known2 = None, known3 = None, verbose = False):
   
    if verbose:
        print('Running normalize_parameters_assuming_3d()')
        
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
    
    ## The knowns indicate which elements of the parameters vector are known. We need three for the 3D nullspace.
    # parameters vector: 'm1', 'm2', 'b1', 'b2', 'k1', 'k2','c12', 'Driving Force'
    # MONOMER parameters vector:  'm1',  'b1',  'k1',  'Driving Force'
    if known1 is None:
        known1 = 0     # m1
    if known2 is None:
        if MONOMER:
            known2 = 2 # k
        else:
            known2 = 1 # m2
    if known3 is None:
        if MONOMER:
            known3 = 3 # F
        else:
            known3 = 7 # F

    
    vect1 = vh[-1]
    #[m1_1, m2_1, b1_1, b2_1, k1_1, k2_1, c12_1, F_1] = vect1
    vect2 = vh[-2]
    #[m1_2, m2_2, b1_2, b2_2, k1_2, k2_2, c12_2, F_2] = vect2
    vect3 = vh[-3]
    
    if len(vect1) <= 0:
        print('Warning: vh[-1] is ' + str(vh[-1]))
    if len(vect1) <= known3:
        print('Warning: vect1 has length ' + str(len(vect1)) + ' so we cannot access element ' + str(known3))
    
    if verbose:
        print("If the null-space is 3D, we must be able to independently determine two parameters; say it's m1, m2, and F.")

    
    # find linear combination such that:
    # a * vect1[0] + b * vect2[0] + c * vect3[0] = m1_set   and
    # a * vect1[1] + b * vect2[1] + c * vect3[1] = m2_set   and
    # a * vect1[7] + b * vect2[7] + c * vect3[7] = F_set
    ## But this rearranges to:
    
    denom = (vect3[known1] * vect2[known2] * vect1[known3]  - 
             vect2[known1] * vect3[known2] * vect1[known3]  - 
             vect3[known1] * vect1[known2] * vect2[known3]  + 
             vect1[known1] * vect3[known2] * vect2[known3]  + 
             vect2[known1] * vect1[known2] * vect3[known3]  - 
             vect1[known1] * vect2[known2] * vect3[known3])

    coefa = -(-vals_set[known3] * vect3[known1] * vect2[known2]  + 
              vals_set[known3] * vect2[known1] * vect3[known2]  - 
              vals_set[known1] * vect3[known2] * vect2[known3]  + 
              vect3[known1] * vals_set[known2] * vect2[known3]  + 
              vals_set[known1] * vect2[known2] * vect3[known3]  - 
              vect2[known1] * vals_set[known2] * vect3[known3])/ denom
    coefb = -(vals_set[known3] * vect3[known1] * vect1[known2]  - 
              vals_set[known3] * vect1[known1] * vect3[known2]  + 
              vals_set[known1] * vect3[known2] * vect1[known3]  - 
              vect3[known1] * vals_set[known2] * vect1[known3]  - 
              vals_set[known1] * vect1[known2] * vect3[known3]  + 
              vect1[known1] * vals_set[known2] * vect3[known3])/ denom
    coefc = -(-vals_set[known3] * vect2[known1] * vect1[known2]  + 
              vals_set[known3] * vect1[known1] * vect2[known2]  - 
              vals_set[known1] * vect2[known2] * vect1[known3]  + 
              vect2[known1] * vals_set[known2] * vect1[known3]  + 
              vals_set[known1] * vect1[known2] * vect2[known3]  - 
              vect1[known1] * vals_set[known2] * vect2[known3])/ denom

    parameters = [coefa*vect1[k]+coefb*vect2[k]+coefc*vect3[k]  for k in range(len(vect1)) ]
    
    if verbose:
        print('Parameters 3D: ')
        print(parameters)
    return parameters, coefa, coefb, coefc



'''Below, I (Lydia) am practicing using the code for monomer and dimer.
I used the simulation to find complex amplitudes from specific frequencies. 
'''

#MONOMER
#parameters used: m=4, b=0.01, k=16 f=1
#what is measurementdf
#created my own dataframe, but you can import a file for this 

import pandas as pd
datamono = {
    'drive' : [2, 2.0025],
    'R1AmpCom': [4.014494396865636+ (1j*-49.678113600020566), -19.98251374031844 + (1j*-9.997497505306109)],
    'R2AmpCom': [0,0]
} 
measurementdfmono = pd.DataFrame(datamono)

#Calculate Zmatrix
#Doesn't matter which kind of system you have, this function will know (monomer or dimer)
monozmatrix = Zmat(measurementdfmono, True, False,
          frequencycolumn = 'drive', complexamplitude1 = 'R1AmpCom', complexamplitude2 = 'R2AmpCom', dtype=complex)

#perform SVD and extract the unnormalized parameters 
#U - left singular vectors
#S - singular values
#V - right singular vectors
#Vh - transpose of right singular vectors
U_m, S_m, Vh_m = np.linalg.svd(monozmatrix)
V_m = Vh_m.conj().T

#need to figure out how to get the smallest value from S and match it to a column of V
#but I just did it manually for now by looking at S
notnormparam_mono = V_m[:,-1]

#Now, let's normalize them so we can get the real values!

#setting force equal to force we used in simulation (in this case 1N)
final_mono = normalize_parameters_1d_by_force(notnormparam_mono, 1)

#[m_1, b_1, k_1, Force]
#and it checks out!!

#DIMER
#parameters: [k_1, k_2, k_12, b_1, b_2, F, m_1, m_2]
#          : [1, 10, 1, 0.1, 0.1, 10, 1, 10]

datadi = {
    'drive' : [1.9975, 2.0025],
    'R1AmpCom': [-5.06108785444674 + 1j*-0.5176227871244886, -5.010047982899977 + 1j*-0.508414957225015],
    'R2AmpCom': [0.17499161387621584 + 1j*0.019120282594415245, 0.17203785361315663 + 1j*0.018655133040042425]
} 
measurementdfdi = pd.DataFrame(datadi)
dizmatrix = Zmat(measurementdfdi, False, False, frequencycolumn='drive', complexamplitude1 = 'R1AmpCom', complexamplitude2 = 'R2AmpCom', dtype=complex,)
U_d, S_d, Vh_d = np.linalg.svd(dizmatrix)
V_d = Vh_d.conj().T
notnormparam_di = V_d[:,-1]
final_di = normalize_parameters_1d_by_force(notnormparam_di,10)
#print(final_di)
#['m1', 'm2', 'b1', 'b2', 'k1', 'k2','k12', 'Driving Force']
#checks out!!


