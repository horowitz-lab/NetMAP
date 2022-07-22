import numpy as np
from resonatorstats import rsqrd
from resonatorsimulator import *

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

""" Simulator privilege to determine SNR. 
    Only one (first) frequency will be used.
"""
def SNRknown(freq,vals_set, noiselevel, use_complexnoise, 
             MONOMER, detailed = False):
    A1,_,_= noisyR1ampphase(freq, vals_set=vals_set, noiselevel = 0, MONOMER=MONOMER) # privilege! no noise!
    A2,_,_= noisyR2ampphase(freq, vals_set=vals_set, noiselevel = 0, MONOMER=MONOMER)
    if use_complexnoise:
        STD1 = noiselevel* complexamplitudenoisefactor
        STD2 = STD1
    else:
        STD1 = noiselevel* amplitudenoisefactor1
        STD2 = noiselevel* amplitudenoisefactor2
        
    SNR_R1 = A1 / STD1
    SNR_R2 = A2 / STD2
    
    if detailed:
        # SNR, SNR, signal, noise, signal, noise
        return SNR_R1[0],SNR_R2[0], A1, STD1, A2, STD2
    else:
        return SNR_R1[0],SNR_R2[0]

def SNRs(freqs,vals_set, noiselevel, use_complexnoise, MONOMER,
         privilege=True,  detailed = False):
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
                    freq,vals_set, noiselevel, use_complexnoise,MONOMER, detailed = detailed)
            else:
                SNR_R1,SNR_R2, A1, STD1, A2, STD2 = SNRcalc(
                    freq,vals_set=vals_set, noiselevel = noiselevel, MONOMER=MONOMER, detailed = detailed)
            A1list.append(A1)
            STD1list.append(STD1)
            A2list.append(A2)
            STD2list.append(STD2)
        else:    
            if privilege:
                SNR_R1,SNR_R2 = SNRknown(freq,vals_set, noiselevel, use_complexnoise,MONOMER, detailed = detailed)
            else:
                SNR_R1,SNR_R2 = SNRcalc(freq,vals_set=vals_set, noiselevel = noiselevel, MONOMER=MONOMER, detailed = detailed)
        SNR_R1_list.append(SNR_R1) # list is in same order as frequencies
        SNR_R2_list.append(SNR_R2)
    
    if detailed:
        return max(SNR_R1_list),max(SNR_R2_list),min(SNR_R1_list),min(SNR_R2_list), \
            np.mean(SNR_R1_list),np.mean(SNR_R2_list), SNR_R1_list, SNR_R2_list, \
            np.mean(A1list), np.mean(STD1list), np.mean(A2list), np.mean(STD2list)
            
    else:
        return max(SNR_R1_list),max(SNR_R2_list),min(SNR_R1_list),min(SNR_R2_list), \
            np.mean(SNR_R1_list),np.mean(SNR_R2_list), SNR_R1_list, SNR_R2_list 

""" Experimentalist style to determine SNR """
def SNRcalc(freq, vals_set,noiselevel,MONOMER,forceboth,use_complexnoise,
            plot = False, ax = None,  detailed = False, ):
    n = 50 # number of randomized values to calculate
    amps1 = np.zeros(n)
    zs1 = np.zeros(n ,dtype=complex)
    amps2 = np.zeros(n)
    zs2 = np.zeros(n ,dtype=complex)
    for j in range(n):
        thisamp1, _, thisz1 = noisyR1ampphase(freq, vals_set,noiselevel,MONOMER,forceboth,use_complexnoise)
        amps1[j] = thisamp1
        zs1[j] = thisz1[0] # multiple simulated measurements of complex amplitude Z1 (of R1)
        thisamp2, _, thisz2 = noisyR2ampphase(freq, vals_set,noiselevel,MONOMER,forceboth,use_complexnoise)
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

#SNRcalc(drive[p[1]], plot = True)

def measurementdfcalc(drive, p, vals_set, noiselevel,
                      noisy_spectra = None, 
                      noiseless_spectra = None
                     ):
    
    if noisy_spectra is None:
        noisy_spectra = calculate_spectra(drive, vals_set, noiselevel=noiselevel, forceboth=forceboth)
    R1_amp, R1_phase, R2_amp, R2_phase, R1_real_amp, R1_im_amp, R2_real_amp, R2_im_amp,_ = noisy_spectra
    
    if noiseless_spectra is None:
        noiseless_spectra = calculate_spectra(drive, vals_set, noiselevel = 0, 
    	                                      MONOMER = MONOMER, forceboth = forceboth)
    R1_amp_noiseless, R1_phase_noiseless, R2_amp_noiseless, R2_phase_noiseless, \
        R1_real_amp_noiseless, R1_im_amp_noiseless, R2_real_amp_noiseless, R2_im_amp_noiseless, _ = noiseless_spectra
    
    table = []
    for i in range(len(p)):
        if False:
            print('p: ' + str(p))
            print('freq: ' + str(p[i]))
            print('Measured amplitude: ' + str(R1_amp[p[i]]))
            print('correct amplitude: ' + str(R1_amp_noiseless[p[i]]))
            print('Syserr: ', syserr(R1_amp[p[i]], R1_amp_noiseless[p[i]]), ' %')
        
        SNR_R1, SNR_R2 = SNRknown(drive[p[i]], vals_set=vals_set, noiselevel = noiselevel)
        table.append([drive[p[i]], R1_amp[p[i]], R1_phase[p[i]], R2_amp[p[i]], R2_phase[p[i]], 
                      complexamp(R1_amp[p[i]],R1_phase[p[i]] ),
                      complexamp(R2_amp[p[i]], R2_phase[p[i]]),
                      SNR_R1, SNR_R2,
                     syserr(R1_amp[p[i]], R1_amp_noiseless[p[i]]),
                     (R1_phase[p[i]] - R1_phase_noiseless[p[i]]),
                     syserr(R2_amp[p[i]],R2_amp_noiseless[p[i]]),
                     (R2_phase[p[i]]-R2_phase_noiseless[p[i]]),
                     ])

    df = pd.DataFrame(data = table, 
                      columns = ['drive', 'R1Amp', 'R1Phase', 'R2Amp', 'R2Phase',
                                 'R1AmpCom', 'R2AmpCom',
                                 'SNR_R1','SNR_R2', # less privileged
                                 'R1Amp_syserr%', 'R1Phase_diff', 'R2Amp_syserr%', 'R2Phase_diff']) # more privileged
    return df

def compile_rsqrd(R1_amp, R1_phase, R2_amp, R2_phase, R1_real_amp, R1_im_amp, R2_real_amp, R2_im_amp,
              drive, K1, K2, K12, B1, B2, FD, M1, M2, MONOMER, forceboth, label = '', 
                  oneminus = True, takelog = True): 
        
        theseresults = []
        theseresults_cols = []
        
        # Polar coordinates and cartesian coordinates
        rsqrdl = rsqrdlist(R1_amp, R1_phase, R2_amp, R2_phase, R1_real_amp, R1_im_amp, R2_real_amp, R2_im_amp,
              drive, K1, K2, K12, B1, B2, FD, M1, M2, MONOMER = MONOMER, forceboth = forceboth)
        expt_A1_rsqrd,expt_phase1_rsqrd,expt_A2_rsqrd,expt_phase2_rsqrd, expt_realZ1_rsqrd,expt_imZ1_rsqrd, expt_realZ2_rsqrd, expt_imZ2_rsqrd = rsqrdl
      
        # polar
        theseresults.append([expt_A1_rsqrd,  expt_phase1_rsqrd])
        theseresults_cols.append(['expt_A1_rsqrd', 'expt_phase1_rsqrd'])
        
        if MONOMER:
            avg_expt_polar_rsqrd = (expt_A1_rsqrd + expt_phase1_rsqrd) / 2
        else: 
            expt_ampavg_rsqrd = (expt_A1_rsqrd + expt_A2_rsqrd)/2
            theseresults.append([expt_A2_rsqrd,  expt_phase2_rsqrd, expt_ampavg_rsqrd])
            theseresults_cols.append(['expt_A2_rsqrd', 'expt_phase2_rsqrd', 'expt_ampavg_rsqrd'])
            avg_expt_polar_rsqrd = (expt_A1_rsqrd + expt_A2_rsqrd + expt_phase1_rsqrd + expt_phase2_rsqrd) / 4
            
        theseresults.append([avg_expt_polar_rsqrd])
        theseresults_cols.append(['avg_expt_polar_rsqrd']) 
        
        # cartesian
        theseresults.append([expt_realZ1_rsqrd,  expt_imZ1_rsqrd])
        theseresults_cols.append(['expt_realZ1_rsqrd', 'expt_imZ1_rsqrd'])
        
        if MONOMER:
            avg_expt_cartes_rsqrd = (expt_realZ1_rsqrd+expt_imZ1_rsqrd)/2
        else:
            theseresults.append([expt_realZ2_rsqrd,  expt_imZ2_rsqrd])
            theseresults_cols.append(['expt_realZ2_rsqrd', 'expt_imZ2_rsqrd'])
            avg_expt_cartes_rsqrd = (expt_realZ1_rsqrd + expt_imZ1_rsqrd + expt_realZ2_rsqrd + expt_imZ2_rsqrd)/4
        theseresults.append(avg_expt_cartes_rsqrd)
        theseresults_cols.append('avg_expt_cartes_rsqrd')
        
        theseresults = flatten(theseresults)
        theseresults_cols = flatten(theseresults_cols)
        
        if oneminus:
            theseresults = [1-rsqrd for rsqrd in theseresults]
            theseresults_cols = ['1-' + name for name in theseresults_cols]
            
        if takelog:
            theseresults = theseresults + [np.log10(element) for element in theseresults]
            theseresults_cols = theseresults_cols + ['log ' + name for name in theseresults_cols]
        
        if label != '':
            theseresults_cols = [name + '_' + label for name in theseresults_cols]
        
        return theseresults, theseresults_cols

