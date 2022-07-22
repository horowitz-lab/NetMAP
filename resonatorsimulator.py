import sympy as sp
from helperfunctions import read_params
import numpy as np

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
wd = sp.symbols('\omega_d', real = True)

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

### Ampolitude and phase
#Wrap phases for plots

wrap1 = (delta1)%(2*sp.pi)
wrap2 = (delta2)%(2*sp.pi)
wrapmono = deltamono%(2*sp.pi)
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



#lambdify curves using sympy

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

#define spectrum functions with error

#curve = amplitude, theta = phase, e = err (i.e. noise)
def curve1(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2, e, MONOMER, forceboth):
    if MONOMER:
        return c1mono(np.array(w), k_1, b1_, F_, m_1) + e
    else: # dimer
        if forceboth:
            return c1FF(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
        else: #force just m1
            return c1(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
    
def theta1(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2, e, MONOMER, forceboth):
    if MONOMER:
        return t1mono(np.array(w), k_1, b1_, F_, m_1) - 2*np.pi + e
    else: # dimer
        if forceboth:
            return t1FF(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) - 2*np.pi + e
        else: #force just m1
            return t1(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) - 2*np.pi + e
    
def curve2(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2, e, forceboth):
    if forceboth:
        return c2FF(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
    else: #force just m1
        return c2(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
    
def theta2(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2, e, forceboth):
    if forceboth:
        return t2FF(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) - 2*np.pi + e
    else: #force just m1
        return t2(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) - 2*np.pi + e
    
def realamp1(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2, e, MONOMER, forceboth):
    if MONOMER:
        return re_mono(np.array(w), k_1, b1_, F_, m_1) + e
    else: #dimer
        if forceboth:
            return re1FF(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
        else:
            return re1(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
    
def imamp1(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2, e, MONOMER, forceboth):
    if MONOMER: 
        return im_mono(np.array(w), k_1, b1_, F_, m_1) + e
    else:
        if forceboth:
            return im1FF(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e   
        else: 
            return im1(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
    
def realamp2(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2, e, forceboth):
    if forceboth:
        return re2FF(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
    else:
        return re2(np.array(w), k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
    
def imamp2(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2, e, forceboth):
    if forceboth:
        return im2FF(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
    else:
        return im2(w, k_1, k_2, k_12, b1_, b2_, F_, m_1, m_2) + e
    
## Monomer:    
def curvemono(w, k_1, b1_, F_, m_1, e):
     return c1mono(np.array(w), k_1, b1_, F_, m_1) + e
    
def thetamono(w, k_1, b1_, F_, m_1, e):
     return t1mono(np.array(w), k_1, b1_, F_, m_1) - 2*np.pi + e

def realampmono(w, k_1, b1_, F_, m_1, e):
     return re_mono(np.array(w), k_1, b1_, F_, m_1) + e
    
def imampmono(w, k_1, b1_, F_, m_1, e):
     return im_mono(np.array(w), k_1, b1_, F_, m_1) + e
	 



#define noise (randn(n,) gives a array of normally-distributed random numbers of size n)
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
def complex_noise(n, noiselevel):
    global complexamplitudenoisefactor
    complexamplitudenoisefactor = 0.0005
    return noiselevel* complexamplitudenoisefactor * np.random.randn(n,)

## Calculate the amplitude and phase as spectra, possibly adding noise
def calculate_spectra(drive, vals_set, noiselevel, MONOMER, forceboth, 
                      use_complexnoise = True, usenoise = True):
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
        
    n = len(drive)
    
    if usenoise: # add a random vector of positive and negative numbers to the curve.

        if use_complexnoise: # apply noise in cartesian coordinates
            R1_real_amp = realamp1(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                   complex_noise(n,noiselevel), MONOMER, forceboth = forceboth)
            R1_im_amp   = imamp1  (drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                   complex_noise(n,noiselevel), MONOMER, forceboth = forceboth)
            R2_real_amp = realamp2(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                   complex_noise(n,noiselevel), forceboth = forceboth)
            R2_im_amp   = imamp2  (drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                   complex_noise(n,noiselevel), forceboth = forceboth)

            R1_amp   = amp(R1_real_amp, R1_im_amp)
            R2_amp   = amp(R2_real_amp, R2_im_amp)
            R1_phase = np.unwrap(np.angle(R1_real_amp + R1_im_amp*1j))
            R2_phase = np.unwrap(np.angle(R2_real_amp + R2_im_amp*1j))

        else: # apply noise in polar coordinates
            R1_amp   = curve1(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                              amp1_noise(n,noiselevel), MONOMER, forceboth = forceboth)
            R1_phase = theta1(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                              phase1_noise(n,noiselevel), MONOMER, forceboth = forceboth)
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

    else: ## This won't work later when I expand the drive list but I use it as a sanity check.
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
             drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set)
        
    return R1_amp, R1_phase, R2_amp, R2_phase, R1_real_amp, R1_im_amp, R2_real_amp, R2_im_amp, privilegedrsqrd
	
## Calculate the amplitude and phase as individual measurements, possibly adding noise
# Unfortunately doesn't yet work with multiple drive frequencies
def noisyR1ampphase(drive, vals_set, noiselevel, MONOMER, forceboth, use_complexnoise):
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set,MONOMER=MONOMER)
    n = listlength(drive)
    if use_complexnoise:
        ## calculate fresh the noise of amplitude 1. This is an independent noise calculation.
        R1_real_amp = realamp1(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                               noiselevel* complexamplitudenoisefactor * np.random.randn(n,), MONOMER, forceboth=forceboth)
        R1_im_amp   = imamp1  (drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                               noiselevel* complexamplitudenoisefactor * np.random.randn(n,), MONOMER, forceboth=forceboth)
        R1complexamp = R1_real_amp + 1j * R1_im_amp
        return amp(R1_real_amp,R1_im_amp), np.angle(R1_real_amp + R1_im_amp*1j), R1complexamp
    else:
        a = curve1(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                   noiselevel* amplitudenoisefactor1 * np.random.randn(n,), MONOMER, forceboth=forceboth)
        p = theta1(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                   noiselevel* phasenoisefactor1 * np.random.randn(n,), MONOMER, forceboth=forceboth)
        return a,p, complexamp(a,p)
        

def noisyR2ampphase(drive, vals_set, noiselevel, MONOMER, forceboth, use_complexnoise):
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
    n = listlength(drive)
    if use_complexnoise:
        ## calculate fresh the noise of amplitude. This is an independent noise calculation.
        R2_real_amp = realamp2(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                               noiselevel* complexamplitudenoisefactor * np.random.randn(n,), forceboth=forceboth)
        R2_im_amp   = imamp2  (drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                               noiselevel* complexamplitudenoisefactor * np.random.randn(n,), forceboth=forceboth)
        R2complexamp = R2_real_amp + 1j * R2_im_amp
        return amp(R2_real_amp,R2_im_amp), np.angle(R2_real_amp + R2_im_amp*1j), R2complexamp
    else:
        a = curve2(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, noiselevel* amplitudenoisefactor2 * np.random.randn(n,), forceboth=forceboth)
        p = theta2(drive, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, noiselevel* phasenoisefactor2 * np.random.randn(n,), forceboth=forceboth)
        return a,p, complexamp(a,p)