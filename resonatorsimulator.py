# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:42:36 2022

Solve equations of motion using Cramer's rule in order to obtain amplitude and phase of each resonator in the network.

@author: vhorowit
"""

import numpy as np
import sympy as sp
from helperfunctions import read_params, listlength
from resonatorphysics import amp, complexamp
import matplotlib.pyplot as plt
from resonatorstats import rsqrd
from resonatorphysics import A_from_Z

# defaults
usenoise = True
use_complexnoise = True

#Define all variables for sympy

#individual springs that correspond to individual masses
k1 = sp.symbols('k_1', real = True)
k2 = sp.symbols('k_2', real = True)

#springs that connect two masses
k12 = sp.symbols('k_12', real = True)

#damping coefficients
b1 = sp.symbols('b1', real = True)
b2 = sp.symbols('b2', real = True)
 
#masses
m1 = sp.symbols('m1', real = True)
m2 = sp.symbols('m2', real = True)

#Driving force amplitude
F = sp.symbols('F', real = True)

#driving frequency (leave as variable)
wd = sp.symbols(r'\omega_d', real = True)

#symbolically Solve for driving amplitudes and phase using sympy

### Dimer
#Matrix for complex equations of motion, Matrix . Zvec = Fvec
unknownsmatrix = sp.Matrix([[-wd**2*m1 + 1j*wd*b1 + k1 + k12, -k12], 
                            [-k12, -wd**2*m2 + 1j*wd*b2 + k2 + k12]])

#Matrices for Cramer's Rule: substitute force vector Fvec=[F,0] for each column in turn (m1 is driven, m2 is not)
unknownsmatrix1 = sp.Matrix([[F, -k12], [0, -wd**2*m2 + 1j*wd*b2 + k2 + k12]])
unknownsmatrix2 = sp.Matrix([[-wd**2*m1 + 1j*wd*b1 + k1 + k12, F], [-k12, 0]])

#Apply Cramer's Rule to solve for Zvec
complexamp1, complexamp2 = (unknownsmatrix1.det()/unknownsmatrix.det(), unknownsmatrix2.det()/unknownsmatrix.det())

#Solve for phases for each mass
delta1 = sp.arg(complexamp1) # Returns the argument (phase angle in radians) of a complex number. 
delta2 = sp.arg(complexamp2) # sp.re(complexamp2)/sp.cos(delta2) (this is the same thing)

### What if we apply the same force to both masses of dimer?
#Matrices for Cramer's Rule: substitute force vector Fvec=[F,0] for each column in turn (m1 is driven, m2 is not)
unknownsmatrix1FF = sp.Matrix([[F, -k12], [F, -wd**2*m2 + 1j*wd*b2 + k2 + k12]])
unknownsmatrix2FF = sp.Matrix([[-wd**2*m1 + 1j*wd*b1 + k1 + k12, F], [-k12, F]])

#Apply Cramer's Rule to solve for Zvec
complexamp1FF, complexamp2FF = (unknownsmatrix1FF.det()/unknownsmatrix.det(), unknownsmatrix2FF.det()/unknownsmatrix.det())

#Solve for phases for each mass
delta1FF = sp.arg(complexamp1FF) # Returns the argument (phase angle in radians) of a complex number. 
delta2FF = sp.arg(complexamp2FF) # sp.re(complexamp2)/sp.cos(delta2) (this is the same thing)

### Monomer
complexamp1monomer = F/(-wd**2*m1 + 1j*wd*b1 + k1) # Don't need Cramer's rule for monomer.
deltamono = sp.arg(complexamp1monomer)

### Amplitude and phase
#Wrap phases for plots

wrap1 = (delta1)%(2*sp.pi)
wrap2 = (delta2)%(2*sp.pi)
wrapmono = (deltamono)%(2*sp.pi)
wrap1FF = (delta1FF)%(2*sp.pi)
wrap2FF = (delta2FF)%(2*sp.pi)

#Solve for amplitude coefficients
amp1 = sp.Abs(complexamp1)
amp2 = sp.Abs(complexamp2)
ampmono = sp.Abs(complexamp1monomer)
amp1FF = sp.Abs(complexamp1FF)
amp2FF = sp.Abs(complexamp2FF)

"""complexamp1 = amp1 * sp.exp(sp.I * sp.pi * delta1)
complexamp2 = amp2 * sp.exp(sp.I * sp.pi * delta2) """

"""
if verbose: # display symbolic solutions
    display(r"R1 Amplitude:")
    display(amp1)

    display(r"R2 Amplitude:")
    display(amp2)

    display(r"R1 complex amplitude:")
    display(complexamp1)
    
    display(r"R2 complex amplitude:")
    display(complexamp2)
    
    display("R1 Real amplitude:")
    display(sp.re(complexamp1))
    
    display("R1 Imaginary amplitude:")
    display(sp.im(complexamp1))
    
    display("R2 Real amplitude:")
    display(sp.re(complexamp2))
    
    display("R2 Imaginary amplitude:")
    display(sp.im(complexamp2))
    """

#lambdify curves using sympy
#c = amplitude (not complex), t = phase
#re and im are the real and imaginary parts of complex number

c1 = sp.lambdify((wd, k1, k2, k12, b1, b2, F, m1, m2), amp1)
t1 = sp.lambdify((wd, k1, k2, k12, b1, b2, F,  m1, m2), wrap1)

c2 = sp.lambdify((wd, k1, k2, k12, b1, b2, F, m1, m2), amp2)
t2 = sp.lambdify((wd, k1, k2, k12, b1, b2, F, m1, m2), wrap2)

re1 = sp.lambdify((wd, k1, k2, k12, b1, b2, F, m1, m2), sp.re(complexamp1))
im1 = sp.lambdify((wd, k1, k2, k12, b1, b2, F, m1, m2), sp.im(complexamp1))
re2 = sp.lambdify((wd, k1, k2, k12, b1, b2, F, m1, m2), sp.re(complexamp2))
im2 = sp.lambdify((wd, k1, k2, k12, b1, b2, F, m1, m2), sp.im(complexamp2))

c1mono  = sp.lambdify((wd, k1, b1, F, m1), ampmono)
t1mono  = sp.lambdify((wd, k1, b1, F, m1), wrapmono)
re_mono = sp.lambdify((wd, k1, b1, F, m1), sp.re(complexamp1monomer))
im_mono = sp.lambdify((wd, k1, b1, F, m1), sp.im(complexamp1monomer))

c1FF  = sp.lambdify((wd, k1, k2, k12, b1, b2, F, m1, m2), amp1FF)
t1FF  = sp.lambdify((wd, k1, k2, k12, b1, b2, F, m1, m2), wrap1FF)
c2FF  = sp.lambdify((wd, k1, k2, k12, b1, b2, F, m1, m2), amp2FF)
t2FF  = sp.lambdify((wd, k1, k2, k12, b1, b2, F, m1, m2), wrap2FF)
re1FF = sp.lambdify((wd, k1, k2, k12, b1, b2, F, m1, m2), sp.re(complexamp1FF))
im1FF = sp.lambdify((wd, k1, k2, k12, b1, b2, F, m1, m2), sp.im(complexamp1FF))
re2FF = sp.lambdify((wd, k1, k2, k12, b1, b2, F, m1, m2), sp.re(complexamp2FF))
im2FF = sp.lambdify((wd, k1, k2, k12, b1, b2, F, m1, m2), sp.im(complexamp2FF))

#define functions

#curve = amplitude, theta = phase, e = error (i.e. noise)
#realamp and imamp refer to the real and imaginary parts of the complex amplitude
#for MONOMER and forceboth, you would enter True or False
#forceboth means there are forces on both masses of the dimer
#to do a trimer, code needs to be added ofc. And you could forceone, forceboth or forcethree
#w takes in a list of frequencies

def curve1(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2, e, MONOMER, forceboth): 
    with np.errstate(divide='ignore'):
        if MONOMER:
            return c1mono(np.array(w), k_1, b1_, F_, m_1) + e #why np.array(w)
        else: # dimer
            if forceboth:
                return c1FF(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
            else: #force just m1
                return c1(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
    
def theta1(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2, e, MONOMER, forceboth):
    with np.errstate(divide='ignore'):
        if MONOMER:
            return t1mono(np.array(w), k_1, b1_, F_, m_1) - 2*np.pi + e
        else: # dimer
            if forceboth:
                return t1FF(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) - 2*np.pi + e
            else: #force just m1
                return t1(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) - 2*np.pi + e
    
def curve2(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2, e, forceboth, MONOMER = False):
    with np.errstate(divide='ignore'):
        if forceboth:
            return c2FF(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
        else: #force just m1
            return c2(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
    
def theta2(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2, e, forceboth, MONOMER = False):
    with np.errstate(divide='ignore'):
        if forceboth:
            return t2FF(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) - 2*np.pi + e
        else: #force just m1
            return t2(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) - 2*np.pi + e
    
def realamp1(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2, e, MONOMER, forceboth):
    with np.errstate(divide='ignore'):
        if MONOMER:
            return re_mono(np.array(w), k_1, b1_, F_, m_1) + e
        else: #dimer
            if forceboth:
                return re1FF(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
            else:
                return re1(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
    
def imamp1(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2, e, MONOMER, forceboth):
    with np.errstate(divide='ignore'):
        if MONOMER: 
            return im_mono(np.array(w), k_1, b1_, F_, m_1) + e
        else:
            if forceboth:
                return im1FF(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e   
            else: 
                return im1(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
    
#MONOMER = False here because there would only be one complex # and thus one re and one im for a monomer 
def realamp2(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2, e, forceboth, MONOMER = False):
    with np.errstate(divide='ignore'):
        if forceboth:
            return re2FF(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
        else:
            return re2(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
    
def imamp2(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2, e, forceboth, MONOMER = False):
    with np.errstate(divide='ignore'):
        if forceboth:
            return im2FF(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
        else:
            return im2(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
    
## Monomer:
    # Could use this or could use functions above and just specify MONOMER = True. 
    # Note: you would just put something like 0 for all the known parameters that 
    # don't apply to a monomer (like m_2, k_2)
    
def curvemono(w, k_1, b1_, F_, m_1, e):
    with np.errstate(divide='ignore'):
         return c1mono(np.array(w), k_1, b1_, F_, m_1) + e
    
def thetamono(w, k_1, b1_, F_, m_1, e):
    with np.errstate(divide='ignore'):
         return t1mono(np.array(w), k_1, b1_, F_, m_1) - 2*np.pi + e

def realampmono(w, k_1, b1_, F_, m_1, e):
    with np.errstate(divide='ignore'):
         return re_mono(np.array(w), k_1, b1_, F_, m_1) + e
    
def imampmono(w, k_1, b1_, F_, m_1, e):
    with np.errstate(divide='ignore'):
         return im_mono(np.array(w), k_1, b1_, F_, m_1) + e
     
     
""" calculate rsqrd in polar and cartesian
    using either the vals_set (privileged rsqrd) or the parameters from SVD (experimental rsqrd) 
    rsqrd is the Coefficient of Determination.
    """
def rsqrdlist(R1_amp, R1_phase, R2_amp, R2_phase, R1_real_amp, R1_im_amp, R2_real_amp, R2_im_amp,
              drive, k1, k2, k12, b1, b2, F, m1, m2, MONOMER, forceboth):
    R1_amp_rsqrd = rsqrd(model = curve1(drive, k1, k2, k12, b1, b2, F, m1, m2,0 , MONOMER, forceboth = forceboth), 
                       data = R1_amp)
    R1_phase_rsqrd = rsqrd(model = theta1(drive, k1, k2, k12, b1, b2, F, m1, m2,0 , MONOMER, forceboth = forceboth), 
                       data = R1_phase)
    if MONOMER: #np.nan - not a number (b/c a monomer only has one complex amplitude)
        R2_amp_rsqrd = np.nan
        R2_phase_rsqrd = np.nan
    else:
        R2_amp_rsqrd = rsqrd(model = curve2(drive, k1, k2, k12, b1, b2, F, m1, m2,0, forceboth = forceboth ), 
                           data = R2_amp)
        R2_phase_rsqrd = rsqrd(model = theta2(drive, k1, k2, k12, b1, b2, F, m1, m2,0 , forceboth = forceboth), 
                           data = R2_phase)
    R1_real_amp_rsqrd = rsqrd(model = realamp1(drive, k1, k2, k12, b1, b2, F, m1, m2,0, MONOMER,forceboth = forceboth ), 
                       data = R1_real_amp)
    R1_im_amp_rsqrd = rsqrd(model = imamp1(drive, k1, k2, k12, b1, b2, F, m1, m2,0 , MONOMER,forceboth = forceboth), 
                       data = R1_im_amp)
    if MONOMER:
        R2_real_amp_rsqrd = np.nan
        R2_im_amp_rsqrd = np.nan
    else:
        R2_real_amp_rsqrd = rsqrd(model = realamp2(drive, k1, k2, k12, b1, b2, F, m1, m2,0,forceboth = forceboth ), 
                           data = R2_real_amp)
        R2_im_amp_rsqrd = rsqrd(model = imamp2(drive, k1, k2, k12, b1, b2, F, m1, m2,0,forceboth = forceboth ), 
                           data = R2_im_amp)
    rsqrdlist = [R1_amp_rsqrd,R1_phase_rsqrd,R2_amp_rsqrd,R2_phase_rsqrd,R1_real_amp_rsqrd,R1_im_amp_rsqrd, R2_real_amp_rsqrd, R2_im_amp_rsqrd]
    return rsqrdlist

"""
maxamp is the maximum amplitude, probably the amplitude at the resonance peak.
Returns arclength in same units as amplitude.
Not used.
"""
def arclength_between_pair(maxamp, Z1, Z2):
    radius = maxamp/2 # radius of twirl, approximating it as a circle 
    x1 = Z1.real
    y1 = Z1.imag
    x2 = Z2.real
    y2 = Z2.imag
    
    ## If one is above the origin and the other is below the origin, then this code won't work at all.
    if y1*y2 < 0:
        return np.nan,np.nan,np.nan
    
    ## Convert to prime coordinates with origin at the center of the twirl
    if y1<0:
        # twirl is below the origin.
        y1p = y1 + radius
        y2p = y2 + radius
    else:
        # twirl is above the origin
        y1p = y1 - radius
        y2p = y2 - radius
    Z1p = complex(x1, y1p)
    Z2p = complex(x2, y2p)
    
    ## Calculate prime coords amplitude and phase
    angle1p = np.angle(Z1p)
    angle2p = np.angle(Z2p)
    A1p = A_from_Z(Z1p) # hopefully these two amplitudes A1p and A2p are the same as each other and as radius
    A2p = A_from_Z(Z2p)
    
    r = (A1p + A2p)/2 # update radius estimate to average
    #print('Radius: ' + str(radius) + ', radius2: ' + str(A1p) + ', radius3: ' + str(A2p) )
    theta = (angle2p-angle1p) # if I don't calculate abs() then I get signed arclength
    theta = (theta + np.pi) % (2 * np.pi) - np.pi ## Force angle to be between -pi and pi.
    
    # calculate signed arclength
    s = r*theta
    return s, theta, r     
     
#define noise (randn(n,) gives a array of normally-distributed random numbers of size n)
# legacy values from before I implemented use_complexnoise. Hold on to them; Brittany was thoughtful about choosing these.
def amp1_noise(n, noiselevel): 
    global amplitudenoisefactor1
    amplitudenoisefactor1 = 0.005 
    return noiselevel* amplitudenoisefactor1 * np.random.randn(n,)
def phase1_noise(n, noiselevel):
    global phasenoisefactor1
    phasenoisefactor1 = 0.1
    return noiselevel* phasenoisefactor1 * np.random.randn(n,)
def amp2_noise(n, noiselevel):
    global amplitudenoisefactor2
    amplitudenoisefactor2 = 0.0005
    return noiselevel* amplitudenoisefactor2 * np.random.randn(n,)
def phase2_noise(n, noiselevel):
    global phasenoisefactor2
    phasenoisefactor2 = 0.2
    return noiselevel* phasenoisefactor2 * np.random.randn(n,)

""" This is the one I'm actually using """

def complex_noise(n, noiselevel):
    global complexamplitudenoisefactor
    complexamplitudenoisefactor = 0.0005
    return noiselevel* complexamplitudenoisefactor * np.random.randn(n,)

## Calculate the amplitude and phase as spectra, possibly adding noise
def calculate_spectra(drive, vals_set, noiselevel, MONOMER, forceboth):
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
      
    try:
        n = len(drive)
    except TypeError:
        n = drive.size
    
    #usenoise and use_complexnoise are already set to True at the beginning of the code
    if usenoise: # add a random vector of positive and negative numbers to the curve.

        if use_complexnoise: # apply noise in cartesian coordinates
            R1_real_amp = realamp1(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                   complex_noise(n,noiselevel), MONOMER, forceboth = forceboth)
            R1_im_amp   = imamp1  (drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                   complex_noise(n,noiselevel), MONOMER, forceboth = forceboth)
            try:
                R1_amp   = amp(R1_real_amp, R1_im_amp)
            except:
                print('R1_real_amp:',R1_real_amp)
                print('R1_im_amp:',R1_im_amp)
                amp(R1_real_amp, R1_im_amp)
            R1_phase = np.unwrap(np.angle(R1_real_amp + R1_im_amp*1j))
            if MONOMER:
                R2_real_amp = np.zeros_like(R1_real_amp)
                R2_im_amp = R2_real_amp
                R2_amp = np.zeros_like(R1_amp)
                R2_phase = R2_amp
            else:
                R2_real_amp = realamp2(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                   complex_noise(n,noiselevel), forceboth = forceboth)
                R2_im_amp   = imamp2  (drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                   complex_noise(n,noiselevel), forceboth = forceboth)
                R2_amp   = amp(R2_real_amp, R2_im_amp)
                R2_phase = np.unwrap(np.angle(R2_real_amp + R2_im_amp*1j))

        else: # apply noise in polar coordinates
            R1_amp   = curve1(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                              amp1_noise(n,noiselevel), MONOMER, forceboth = forceboth)
            R1_phase = theta1(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                              phase1_noise(n,noiselevel), MONOMER, forceboth = forceboth)
            if MONOMER:
                R2_amp = np.zeros_like (R1_amp)
                R2_phase = np.zeros_like (R1_phase)
            else:
                R2_amp   = curve2(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                  amp2_noise(n,noiselevel), forceboth = forceboth)
                R2_phase = theta2(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                  phase2_noise(n,noiselevel), forceboth = forceboth)

            R1_complexamp = complexamp(R1_amp, R1_phase)
            R2_complexamp = complexamp(R2_amp, R2_phase)

            R1_real_amp = np.real(R1_complexamp)
            R1_im_amp   = np.imag(R1_complexamp)
            R2_real_amp = np.real(R2_complexamp)
            R2_im_amp   = np.imag(R2_complexamp)

    #this is for no noise
    else: ## This won't work later when I expand the drive list but I use it as a sanity check.
        R1_amp_noiseless = curve1(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER, forceboth=forceboth)
        R1_phase_noiseless = theta1(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER, forceboth=forceboth)
        R2_amp_noiseless = curve2(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)
        R2_phase_noiseless = theta2(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)
        
        R1_real_amp_noiseless = realamp1(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER, forceboth=forceboth)
        R1_im_amp_noiseless = imamp1(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER, forceboth=forceboth)
        R2_real_amp_noiseless = realamp2(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)
        R2_im_amp_noiseless = imamp2(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)

        R1_amp = R1_amp_noiseless
        R1_phase = R1_phase_noiseless
        R2_amp = R2_amp_noiseless
        R2_phase = R2_phase_noiseless
        R1_real_amp = R1_real_amp_noiseless
        R1_im_amp = R1_im_amp_noiseless
        R2_real_amp = R2_real_amp_noiseless
        R2_im_amp = R2_im_amp_noiseless
        
    ## calculate privileged rsqrd
    privilegedrsqrd = rsqrdlist(R1_amp, R1_phase, R2_amp, R2_phase, R1_real_amp, R1_im_amp, R2_real_amp, R2_im_amp,\
             drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set,\
             MONOMER=MONOMER, forceboth=forceboth
             )
        
    return R1_amp, R1_phase, R2_amp, R2_phase, R1_real_amp, R1_im_amp, R2_real_amp, R2_im_amp, privilegedrsqrd


## Calculate the amplitude and phase as individual measurements, possibly adding noise
if use_complexnoise:
    # Unfortunately doesn't yet work with multiple drive frequencies
    def noisyR1ampphase(drive, vals_set, noiselevel, MONOMER, forceboth):
        [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set,MONOMER=MONOMER)
        n = listlength(drive)
        ## calculate fresh the noise of amplitude 1. This is an independent noise calculation.
        R1_real_amp = realamp1(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                               noiselevel* complexamplitudenoisefactor * np.random.randn(n,), MONOMER, forceboth=forceboth)
        R1_im_amp   = imamp1  (drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                               noiselevel* complexamplitudenoisefactor * np.random.randn(n,), MONOMER, forceboth=forceboth)
        R1complexamp = R1_real_amp + 1j * R1_im_amp
        return amp(R1_real_amp,R1_im_amp), np.angle(R1_real_amp + R1_im_amp*1j), R1complexamp

    def noisyR2ampphase(drive, vals_set, noiselevel, MONOMER, forceboth):
        [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
        n = listlength(drive)
        ## calculate fresh the noise of amplitude. This is an independent noise calculation.
        R2_real_amp = realamp2(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                               noiselevel* complexamplitudenoisefactor * np.random.randn(n,), forceboth=forceboth)
        R2_im_amp   = imamp2  (drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                               noiselevel* complexamplitudenoisefactor * np.random.randn(n,), forceboth=forceboth)
        R2complexamp = R2_real_amp + 1j * R2_im_amp
        return amp(R2_real_amp,R2_im_amp), np.angle(R2_real_amp + R2_im_amp*1j), R2complexamp

else:
    def noisyR1ampphase(drive, vals_set, noiselevel, MONOMER, forceboth):
        [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
        n = listlength(drive)
        a = curve1(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                   noiselevel* amplitudenoisefactor1 * np.random.randn(n,), MONOMER, forceboth=forceboth)
        p = theta1(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                   noiselevel* phasenoisefactor1 * np.random.randn(n,), MONOMER, forceboth=forceboth)
        return a,p, complexamp(a,p)

    def noisyR2ampphase(drive, vals_set, noiselevel, MONOMER, forceboth):
        [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
        n = listlength(drive)
        a = curve2(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, noiselevel* amplitudenoisefactor2 * np.random.randn(n,), forceboth=forceboth)
        p = theta2(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, noiselevel* phasenoisefactor2 * np.random.randn(n,), forceboth=forceboth)
        return a,p, complexamp(a,p)
    
    
""" Simulator privilege to determine SNR. 
    Only one (first) frequency will be used.
"""
def SNRknown(freq,vals_set, noiselevel, MONOMER, forceboth, use_complexnoise=use_complexnoise, 
             detailed = False):
    A1,_,_= noisyR1ampphase(freq, vals_set=vals_set,noiselevel = 0,MONOMER=MONOMER,forceboth=forceboth) # privilege! no noise!
    A1 = A1[0]
    if MONOMER:
        A2 = np.nan
    else:
        A2,_,_= noisyR2ampphase(freq, vals_set=vals_set,noiselevel = 0,MONOMER=MONOMER,forceboth=forceboth)
        A2 = A2[0]
    if use_complexnoise:
        STD1 = noiselevel* complexamplitudenoisefactor
        STD2 = STD1
    else: # legacy code
        STD1 = noiselevel* amplitudenoisefactor1
        STD2 = noiselevel* amplitudenoisefactor2
        
    SNR_R1 = A1 / STD1
    SNR_R2 = A2 / STD2

    if detailed:
        # SNR, SNR, signal, noise, signal, noise
        return SNR_R1,SNR_R2, A1, STD1, A2, STD2 # R1 and R2 for each quantity
    else:
        return SNR_R1,SNR_R2

def SNRs(freqs,vals_set, noiselevel, MONOMER, forceboth, use_complexnoise=use_complexnoise,
         privilege=True, detailed = False):
    SNR_R1_list = []
    SNR_R2_list = []
    A1list = []
    STD1list = []
    A2list= []
    STD2list = []
    
    for freq in freqs:
        if detailed:
            if privilege:
                SNR_R1,SNR_R2, A1, STD1, A2, STD2 = SNRknown(
                    freq,vals_set, noiselevel=noiselevel,MONOMER=MONOMER, forceboth=forceboth,
                    use_complexnoise=use_complexnoise, detailed = True)
            else:
                SNR_R1,SNR_R2, A1, STD1, A2, STD2 = SNRcalc(
                    freq,vals_set=vals_set, noiselevel = noiselevel, MONOMER=MONOMER, forceboth=forceboth,
                    detailed = True)
            A1list.append(A1)
            STD1list.append(STD1)
            A2list.append(A2)
            STD2list.append(STD2)
        else:    # detailed = False
            if privilege:
                SNR_R1,SNR_R2 = SNRknown(freq,vals_set, noiselevel=noiselevel, forceboth=forceboth,
                                         use_complexnoise=use_complexnoise,MONOMER=MONOMER, detailed = False)
            else:
                SNR_R1,SNR_R2 = SNRcalc(freq,vals_set=vals_set, 
                                        noiselevel = noiselevel, MONOMER=MONOMER, 
                                        forceboth=forceboth,
                                        detailed = False)
        SNR_R1_list.append(SNR_R1) # list is in same order as frequencies
        SNR_R2_list.append(SNR_R2)
    
    if False: # especially verbose
        print('SNR_R1_list', SNR_R1_list)
        print('SNR_R2_list', SNR_R2_list)
        print('mean(SNR_R1_list)', np.mean(SNR_R1_list))
        print('mean(SNR_R2_list)', np.mean(SNR_R2_list))
        
    
    if detailed:
        return max(SNR_R1_list),max(SNR_R2_list),min(SNR_R1_list),min(SNR_R2_list), \
            np.mean(SNR_R1_list),np.mean(SNR_R2_list), SNR_R1_list, SNR_R2_list, \
            np.mean(A1list), np.mean(STD1list), np.mean(A2list), np.mean(STD2list)
            
    else:
        return max(SNR_R1_list),max(SNR_R2_list),min(SNR_R1_list),min(SNR_R2_list), \
            np.mean(SNR_R1_list),np.mean(SNR_R2_list), SNR_R1_list, SNR_R2_list 

""" Experimentalist style to determine SNR, not used because I have a priori privilege """
def SNRcalc(freq,vals_set, noiselevel, MONOMER, forceboth, plot = False, ax = None, detailed = False):
    n = 50 # number of randomized values to calculate
    amps1 = np.zeros(n)
    zs1 = np.zeros(n ,dtype=complex)
    amps2 = np.zeros(n)
    zs2 = np.zeros(n ,dtype=complex)
    for j in range(n):
        thisamp1, _, thisz1 = noisyR1ampphase(freq, vals_set, noiselevel, MONOMER)
        amps1[j] = thisamp1
        zs1[j] = thisz1[0] # multiple simulated measurements of complex amplitude Z1 (of R1)
        thisamp2, _, thisz2 = noisyR2ampphase(freq, vals_set, noiselevel, MONOMER)
        amps2[j] = thisamp2
        zs2[j] = thisz2[0] # multiple simulated measurements of complex amplitude Z2 (of R2)
    SNR_R1 = np.mean(amps1) / np.std(amps1)
    SNR_R2 = np.mean(amps2) / np.std(amps2)

    if plot:
        if ax is not None:
            plt.sca(ax)
        plt.plot(np.real(zs1), np.imag(zs1), '.', alpha = .2) 
        plt.plot(np.real(zs2), np.imag(zs2), '.', alpha = .2) 
        plt.plot(0,0, 'o')
        plt.gca().axis('equal');
        plt.title('Freq: ' + str(freq) +   
                  ', SNR R1: ' ,SNR_R1, 
                  ', SNR R2: ' ,SNR_R2)
    if detailed:
        # SNR, SNR, signal, noise, signal, noise
        return SNR_R1,SNR_R2, np.mean(amps1), np.std(amps1),  np.mean(amps2), np.std(amps2)
    else:
        return SNR_R1,SNR_R2


""" Below, I (Lydia) am practicing using the data to make graphs. 
    This is a helpful teaching tool.
    Comment and uncomment sections to see the graph produced
"""

#Making graphs for the monomer! Used example m, k, b, and f
#Note, the range of frequency matters
# freqs = np.linspace(1.99, 2.01, num=100)
# amps1 = curve1(freqs, 16, 0, 0, 0.01, 0, 1, 4, 0, 0, True, False)
# phase1 = theta1(freqs, 16, 0, 0, 0.01, 0, 1, 4, 0, 0, True, False)
# realpart = realamp1(freqs, 16, 0, 0, 0.01, 0, 1, 4, 0, 0, True, False)
# impart = imamp1(freqs, 16, 0, 0, 0.01, 0, 1, 4, 0, 0, True, False)

#finding the maximum amplitude and the complex amplitude associated with it
# maxamp = max(amps1)
# maxamp_index = np.argmax(amps1)
# corresponding_freq = freqs[maxamp_index]
# real_atmax1 = realamp1(corresponding_freq, 16, 0, 0, 0.01, 0, 1, 4, 0, 0, True, False)
# im_atmax1 = imamp1(corresponding_freq, 16, 0, 0, 0.01, 0, 1, 4, 0, 0, True, False)

#Create one plot for both amps vs freqs and phases vs freqs
# fig, ax1 = plt.subplots()
# ax1.plot(freqs, amps1,'r-', label='Amplitude')
# ax1.set_xlabel('Frequency')
# ax1.set_ylabel('Amplitude')
# ax2 = ax1.twinx()
# ax2.plot(freqs, phase1,'b-', label='Phase')
# ax2.set_ylabel('Phase')
# ax1.legend(loc='upper right')
# ax2.legend(loc='center right')

#plot on complex plane
# plt.plot(realpart, impart, 'g-')
# plt.xlabel('Re(Z)')
# plt.ylabel('Im(Z)')
# plt.axis('equal')

#Making graphs for the dimer! Used example m, k, b, and f
# freqs = np.linspace(0.5, 2, num=500)
# amps1 = curve1(freqs, 1, 10, 1, 0.1, 0.1, 10, 1, 10, 0, False, False)
# amps2 = curve2(freqs, 1, 10, 1, 0.1, 0.1, 10, 1, 10, 0, False)
# phase1 = theta1(freqs, 1, 10, 1, 0.1, 0.1, 10, 1, 10, 0, False, False)
# phase2 = theta2(freqs, 1, 10, 1, 0.1, 0.1, 10, 1, 10, 0, False)
# realpart1 = realamp1(freqs, 1, 10, 1, 0.1, 0.1, 10, 1, 10, 0, False, False)
# impart1 = imamp1(freqs, 1, 10, 1, 0.1, 0.1, 10, 1, 10, 0, False, False)
# realpart2 = realamp2(freqs, 1, 10, 1, 0.1, 0.1, 10, 1, 10, 0, False)
# impart2 = imamp2(freqs, 1, 10, 1, 0.1, 0.1, 10, 1, 10, 0, False)

#getting info to use in NetMAP
# w_1 = 1.9975
# w_2 = 2.0025
# realpart11 = realamp1(w_1, 1, 10, 1, 0.1, 0.1, 10, 1, 10, 0, False, False)
# impart11 = imamp1(w_1, 1, 10, 1, 0.1, 0.1, 10, 1, 10, 0, False, False)
# realpart12 = realamp2(w_1, 1, 10, 1, 0.1, 0.1, 10, 1, 10, 0, False)
# impart12 = imamp2(w_1, 1, 10, 1, 0.1, 0.1, 10, 1, 10, 0, False)

# realpart21 = realamp1(w_2, 1, 10, 1, 0.1, 0.1, 10, 1, 10, 0, False, False)
# impart21 = imamp1(w_2, 1, 10, 1, 0.1, 0.1, 10, 1, 10, 0, False, False)
# realpart22 = realamp2(w_2, 1, 10, 1, 0.1, 0.1, 10, 1, 10, 0, False)
# impart22 = imamp2(w_2, 1, 10, 1, 0.1, 0.1, 10, 1, 10, 0, False)

#print(realpart11, impart11, realpart12, impart12, realpart21, impart21, realpart22, impart22)

#Z_1 - amplitude and phase vs frequency
# fig, ax1 = plt.subplots()
# ax1.plot(freqs, amps1,'r-', label='Amplitude')
# ax1.set_xlabel('Frequency')
# ax1.set_ylabel('Amplitude')
# ax2 = ax1.twinx()
# ax2.plot(freqs, phase1,'b-', label='Phase')
# ax2.set_ylabel('Phase')
# ax1.legend(loc='upper right')
# ax2.legend(loc='center right')
# ax1.set_title('$Z_1(w)$')

#Z_1 - complex plane
# plt.plot(realpart1, impart1, 'go', linestyle='dashed')
# plt.xlabel('Re(Z)')
# plt.ylabel('Im(Z)')
# plt.title('$Z_1(w)$')

#Z_2 - amplitude and phase vs frequency
# fig, ax1 = plt.subplots()
# ax1.plot(freqs, amps2,'r-', label='Amplitude')
# ax1.set_xlabel('Frequency')
# ax1.set_ylabel('Amplitude')
# ax2 = ax1.twinx()
# ax2.plot(freqs, phase2,'b-', label='Phase')
# ax2.set_ylabel('Phase')
# ax1.legend(loc='upper right')
# ax2.legend(loc='center right')
# ax1.set_title('$Z_2(w)$')

#Z_2 - complex plane
# plt.plot(realpart2, impart2, 'go', linestyle='dashed')
# plt.xlabel('Re(Z)')
# plt.ylabel('Im(Z)')
# plt.title('$Z_2(w)$')



