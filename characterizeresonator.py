# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:32:29 2022

@author: vhorowit
"""

import numpy as np
from resonatorstats import rsqrd
from resonatorsimulator import \
    curve1, theta1, curve2, theta2, realamp1, imamp1, realamp2, imamp2
from resonatorphysics import A_from_Z

""" calculate rsqrd in polar and cartesian
    using either the vals_set (privileged rsqrd) or the parameters from SVD (experimental rsqrd) """
def rsqrdlist(R1_amp, R1_phase, R2_amp, R2_phase, R1_real_amp, R1_im_amp, R2_real_amp, R2_im_amp,
              drive, k1, k2, k12, b1, b2, F, m1, m2, MONOMER, forceboth):
    R1_amp_rsqrd = rsqrd(model = curve1(drive, k1, k2, k12, b1, b2, F, m1, m2,0 , MONOMER, forceboth = forceboth), 
                       data = R1_amp)
    R1_phase_rsqrd = rsqrd(model = theta1(drive, k1, k2, k12, b1, b2, F, m1, m2,0 , MONOMER, forceboth = forceboth), 
                       data = R1_phase)
    if MONOMER:
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