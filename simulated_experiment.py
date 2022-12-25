# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:08:21 2022

Simulated spectra + SVD recovery

@author: vhorowit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helperfunctions import \
    read_params, store_params, make_real_iff_real, flatten
from resonatorSVDanalysis import Zmat, \
    normalize_parameters_1d_by_force, normalize_parameters_assuming_3d, \
    normalize_parameters_to_m1_F_set_assuming_2d
from resonatorstats import syserr, combinedsyserr
from resonatorphysics import \
    approx_Q, approx_width, res_freq_weak_coupling, complexamp
from resonatorfrequencypicker import freqpoints
from resonatorsimulator import \
    calculate_spectra, SNRs, SNRknown, rsqrdlist, arclength_between_pair
from resonator_plotting import plot_SVD_results

global complexamplitudenoisefactor
complexamplitudenoisefactor = 0.0005

global use_complexnoise
use_complexnoise = True # this just works best. Don't use the other.

def describeresonator(vals_set, MONOMER, forceboth, noiselevel = None):
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
    
    if MONOMER:
        print('MONOMER')
    else:
        print('DIMER')
        if forceboth:
            print('Applying oscillating force to both masses.')
        else:
            print('Applying oscillating force to m1.')
    print('Approximate Q1: ' + "{:.2f}".format(approx_Q(k = k1_set, m = m1_set, b=b1_set)) + 
          ' width: ' + "{:.2f}".format(approx_width(k = k1_set, m = m1_set, b=b1_set)))
    if not MONOMER:
        print('Approximate Q2: ' + "{:.2f}".format(approx_Q(k = k2_set, m = m2_set, b=b2_set)) +
              ' width: ' + "{:.2f}".format(approx_width(k = k2_set, m = m2_set, b=b2_set)))
    print('Q ~ sqrt(m*k)/b')
    print('Set values:')
    if MONOMER:
        print('m: ' + str(m1_set) + ', b: ' + str(b1_set) + ', k: ' + str(k1_set) + ', F: ' + str(F_set))
        res1 = res_freq_weak_coupling(k1_set, m1_set, b1_set)
        print('res freq: ', res1)
    else:
        if forceboth:
            forcestr = ', F1=F2: '
        else:
            forcestr = ', F1: '

        print('m1: ' + str(m1_set) + ', b1: ' + str(b1_set) + ', k1: ' + str(k1_set) + forcestr + str(F_set))
        print('m2: ' + str(m2_set) + ', b2: ' + str(b2_set) + ', k2: ' + str(k2_set) + ', k12: ' + str(k12_set))
    if noiselevel is not None and use_complexnoise:
        print('noiselevel:', noiselevel)
        print('stdev sigma:', complexamplitudenoisefactor*noiselevel)


def measurementdfcalc(drive, p, 
                      R1_amp,R2_amp,R1_phase, R2_phase, 
                     R1_amp_noiseless,R2_amp_noiseless,
                      R1_phase_noiseless, R2_phase_noiseless,
                      vals_set, noiselevel, MONOMER, forceboth):
    table = []
    for i in range(len(p)):
        if False:
            print('p: ' + str(p))
            print('freq: ' + str(p[i]))
            print('Measured amplitude: ' + str(R1_amp[p[i]]))
            print('correct amplitude: ' + str(R1_amp_noiseless[p[i]]))
            print('Syserr: ', syserr(R1_amp[p[i]], R1_amp_noiseless[p[i]]), ' %')
        
        SNR_R1, SNR_R2 = SNRknown(drive[p[i]],vals_set=vals_set, noiselevel=noiselevel, MONOMER=MONOMER, forceboth=forceboth)
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
        expt_A1_rsqrd,expt_phase1_rsqrd,expt_A2_rsqrd,expt_phase2_rsqrd, \
            expt_realZ1_rsqrd,expt_imZ1_rsqrd, expt_realZ2_rsqrd, expt_imZ2_rsqrd = rsqrdl
      
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
            theseresults = theseresults + [np.log10(element) for element in theseresults] # may warn if rsqrd = 1.
            theseresults_cols = theseresults_cols + ['log ' + name for name in theseresults_cols]
        
        if label != '':
            theseresults_cols = [name + '_' + label for name in theseresults_cols]
        
        return theseresults, theseresults_cols

def assert_results_length(results, columns):
    try: 
        assert len(results) == len(columns)
    except:
        print("len(results)",  len(results))
        print("len(columns)", len(columns))
    try: 
        assert len(flatten(results)) == len(flatten(columns))
    except:
        print('Unequal!')
        print( "len(flatten(results))",  len(flatten(results)) )
        print( "len(flatten(columns))", len(flatten(columns)) )
        
       
# unscaled_vector = vh[-1] has elements: m1, b1, k1, f1
def describe_monomer_results(Zmatrix, smallest_s, unscaled_vector, M1, B1, K1, vals_set, absval = False ):
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, True)
    m_err = syserr(M1,m1_set, absval)
    b_err = syserr(B1,b1_set, absval)
    k_err = syserr(K1,k1_set, absval)
    sqrtkoverm_err = syserr(np.sqrt(K1/M1),np.sqrt(k1_set/m1_set), absval)
    
    print("The Z matrix is ", make_real_iff_real(Zmatrix), \
        ". Its smallest singular value, s_1=", smallest_s,  \
        ", corresponds to singular vector\n p\\vec\\hat=(m\\hat, b\\hat, k\\hat, F)=α(",  \
        unscaled_vector[0], " kg, ", #M
        unscaled_vector[1], "N/(m/s),", #B
        unscaled_vector[2], "N/m,", #K
        unscaled_vector[3], "N), where α=F_set/", unscaled_vector[3], "=", \
        F_set, "/" , unscaled_vector[3], "=", F_set/unscaled_vector[3], \
        "is a normalization constant obtained from our knowledge of the force amplitude F for a 1D-SVD analysis.", 
        "Dividing by α allows us to scale the singular vector to yield the modeled parameters vector.", 
        "Therefore, we obtain m\\hat= ", 
        M1, " kg, b\\hat=", 
        B1, " N/(m/s)  and k\\hat=", \
        K1, "N/m. The percent errors for each of these is", \
        m_err, "%,", \
        b_err, "%, and", \
        k_err, "%, respectively.", \
        "Each of these is within ", \
        max([abs(err) for err in [m_err, b_err, k_err]]), \
        "% of the correct values for m, b, and k.", \
        "We also see that the recovered value √(k ̂/m ̂ )=",
        np.sqrt(K1/M1), "rad/s is more accurate than the individually recovered values for mass and spring stiffness;",
        "this is generally true. ", 
        "The percent error for √(k ̂/m ̂ ) compared to √(k_set/m_set ) is",
        sqrtkoverm_err, "%. This high accuracy likely arises because we choose frequency ω_a at the peak amplitude."
        )


""" demo indicates that the data should be plotted without ticks"""
def simulated_experiment(measurementfreqs,  vals_set, noiselevel, MONOMER, forceboth,
                         drive=None,#np.linspace(minfreq,maxfreq,n), 
                         verbose = False, repeats=1,  labelcounts = False,
                         noiseless_spectra = None, noisy_spectra = None, freqnoise = False, overlay=False,
                         context = None, saving = False, demo = False,
                         resonatorsystem = None,  show_set = None,
                         figsizeoverride1 = None, figsizeoverride2 = None,):

    
    if verbose:
        print('Running simulated_experiment()', repeats, 'times.')
        describeresonator(vals_set, MONOMER, forceboth, noiselevel)
    
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
    
    if drive is None:
        drive = measurementfreqs # fastest way to do it, but R^2 isn't very accurate
    else:
        if noiseless_spectra is None and noisy_spectra is None: # a disadvantage of passing these: can't improve drive.
            drive = np.sort(np.unique(np.append(drive, measurementfreqs)))
        
    p = freqpoints(desiredfreqs = measurementfreqs, drive = drive)
    
    if noiseless_spectra is None: # calculate noiseless spectra
        noiseless_spectra = calculate_spectra(drive, vals_set, noiselevel = 0, MONOMER = MONOMER, forceboth = forceboth)
    R1_amp_noiseless, R1_phase_noiseless, R2_amp_noiseless, R2_phase_noiseless, \
        R1_real_amp_noiseless, R1_im_amp_noiseless, R2_real_amp_noiseless, R2_im_amp_noiseless, _ = noiseless_spectra

    """detailed SNRs returns:
    max(SNR_R1_list),max(SNR_R2_list),min(SNR_R1_list),min(SNR_R2_list), \
        np.mean(SNR_R1_list),np.mean(SNR_R2_list), SNR_R1_list, SNR_R2_list, \
        np.mean(A1list), np.mean(STD1list), np.mean(A2list), np.mean(STD2list)"""

    ## privileged SNR and amplitude
    maxSNR_R1,maxSNR_R2, minSNR_R1,minSNR_R2,meanSNR_R1,meanSNR_R2, SNR_R1_list, SNR_R2_list, \
        A1, STD1, A2, STD2 = SNRs( \
            drive[p],vals_set, noiselevel=noiselevel, MONOMER=MONOMER,forceboth=forceboth, 
            use_complexnoise=use_complexnoise, detailed = True,
            privilege = True)

    if len(p) <= 3:
        ZprivR1f1 = R1_real_amp_noiseless[p[0]] + R1_im_amp_noiseless[p[0]]*1j
        ZprivR1f2 = R1_real_amp_noiseless[p[1]] + R1_im_amp_noiseless[p[1]]*1j
        ## R1 arclength, assuming freq1 is at resonance1
        [arclength_R1, modifiedangle_R1, rad] = arclength_between_pair(A1, ZprivR1f1, ZprivR1f2)
    
    first = True
    results = []

    for i in range(repeats): # repeat the same measurement with different gaussian noise
        theseresults = []
        theseresults_cols = []
        
        theseresults.append(len(p))
        theseresults_cols.append([ 'num frequency points'])
        
        theseresults.append(vals_set) # Store vals_set # same for every row
        if MONOMER:
            theseresults_cols.append(['m1_set',  'b1_set',  'k1_set', 'F_set'])
        else:
            theseresults_cols.append(['m1_set', 'm2_set', 'b1_set', 'b2_set', 'k1_set', 'k2_set', 'k12_set', 'F_set'])
        theseresults.append([noiselevel, noiselevel * complexamplitudenoisefactor])
        theseresults_cols.append(['noiselevel', 'stdev'])
        
        """ if freqnoise:
            #measurementfreqs = [w+random for w in reslist] # to do: describe random
            drive = np.unique(np.sort(np.append(drive, measurementfreqs)))"""

        if noisy_spectra is None or i > 0 or freqnoise:
            # recalculate noisy spectra
            noisy_spectra = calculate_spectra(drive, vals_set=vals_set, 
                                              noiselevel=noiselevel,MONOMER=MONOMER, forceboth=forceboth)
        R1_amp, R1_phase, R2_amp, R2_phase, R1_real_amp, R1_im_amp, R2_real_amp, R2_im_amp,_ = noisy_spectra
        
        theseresults.append([A1])
        theseresults_cols.append(['A1']) # mean amplitude resonator 1
        if not MONOMER:
            theseresults.append([A2])
            theseresults_cols.append(['A2']) #  mean amplitude resonator 2.
        
        if len(p) < 40:
            for i in range(len(p)):
                theseresults.append([drive[p[i]], 
                                     R1_phase_noiseless[p[i]],
                                    R1_amp_noiseless[p[i]]])
                theseresults_cols.append(['Freq' + str(i+1), 
                                          'R1_phase_noiseless' + str(i+1),
                                          'R1_amp_noiseless' + str(i+1)])
                if not MONOMER:
                    theseresults.append([R2_phase_noiseless[p[i]], R2_amp_noiseless[p[i]]])
                    theseresults_cols.append(['R2_phase_noiseless' + str(i+1),'R2_amp_noiseless' + str(i+1)])                
                theseresults.append(SNR_R1_list[i])
                theseresults_cols.append('SNR_R1_f' + str(i+1))
                if not MONOMER:
                    theseresults.append(SNR_R2_list[i])
                    theseresults_cols.append('SNR_R2_f' + str(i+1))
                theseresults.append(R1_amp[p[i]])
                theseresults_cols.append('R1_amp_meas' + str(i+1))
                if not MONOMER:
                    theseresults.append(R2_amp[p[i]])
                    theseresults_cols.append('R2_amp_meas' + str(i+1))   
        assert_results_length(results=theseresults,columns = theseresults_cols)
        
        if len(p) == 2: # two frequency measurements
            theseresults.append(drive[p[1]] - drive[p[0]])
            theseresults_cols.append('Difference')
            theseresults.append(R1_phase_noiseless[p[1]] - R1_phase_noiseless[p[0]])
            theseresults_cols.append('R1_phase_diff')
            if not MONOMER:
                theseresults.append(R2_phase_noiseless[p[1]] - R2_phase_noiseless[p[0]])
                theseresults_cols.append('R2_phase_diff')
        
        # 'arclength_R1' is the arclength separation 
        # between the first two frequency points on the R1 complex spectrum plot
        if len(p) < 3:
            theseresults.append([arclength_R1,modifiedangle_R1])
            theseresults_cols.append(['arclength_R1', 'modifiedangle_R1'])
            
        theseresults.append(drive[p])
        theseresults_cols.append('frequencies')

        df = measurementdfcalc(drive, p, 
                     R1_amp=R1_amp,R2_amp=R2_amp,R1_phase=R1_phase, R2_phase=R2_phase, 
                     R1_amp_noiseless=R1_amp_noiseless,R2_amp_noiseless=R2_amp_noiseless,
                     R1_phase_noiseless=R1_phase_noiseless, R2_phase_noiseless=R2_phase_noiseless,
                     MONOMER=MONOMER, vals_set=vals_set, forceboth=forceboth,
                     noiselevel = noiselevel
                     )
        Zmatrix = Zmat(df, frequencycolumn = 'drive', 
                       complexamplitude1 = 'R1AmpCom', complexamplitude2 = 'R2AmpCom', 
                       MONOMER=MONOMER, forceboth=forceboth, dtype=complex)
        u, s, vh = np.linalg.svd(Zmatrix, full_matrices = True)
        vh = make_real_iff_real(vh)
        
        theseresults.append(approx_Q(m = m1_set, k = k1_set, b = b1_set))
        theseresults_cols.append('approxQ1')
        if not MONOMER:
            theseresults.append(approx_Q(m = m2_set, k = k2_set, b = b2_set))
            theseresults_cols.append('approxQ2')
        theseresults.append(df['R1Amp_syserr%'].mean())
        theseresults_cols.append('R1Ampsyserr%mean(priv)')
        theseresults.append(df.R1Phase_diff.mean())
        theseresults_cols.append('R1phasediffmean(priv)')
        if not MONOMER:
            theseresults.append(df['R2Amp_syserr%'].mean())
            theseresults_cols.append('R2Ampsyserr%mean(priv)')
            theseresults.append(df.R2Phase_diff.mean())
            theseresults_cols.append('R2phasediffmean(priv)')
        
        theseresults.append([s[-1], s[-2]])
        theseresults_cols.append(['smallest singular value', 'second smallest singular value'])
        
        assert_results_length(results=theseresults,columns = theseresults_cols)

        ## 1D NULLSPACE
        M1, M2, B1, B2, K1, K2, K12, FD = read_params(vh[-1], MONOMER) # the 7th singular value is the smallest one (closest to zero)

        # normalize parameters vector to the force, assuming 1D nullspace
        allparameters = normalize_parameters_1d_by_force([M1, M2, B1, B2, K1, K2, K12, FD], F_set)

        M1, M2, B1, B2, K1, K2, K12, FD = allparameters
        
        if MONOMER:
            theseresults.append([M1, B1,  K1, FD])
            theseresults_cols.append(['M1_1D', 'B1_1D','K1_1D', 'FD_1D'])            
        else:
            theseresults.append([M1, M2, B1, B2, K1, K2, K12, FD])
            theseresults_cols.append(['M1_1D', 'M2_1D', 'B1_1D', 'B2_1D', 'K1_1D', 'K2_1D', 'K12_1D', 'FD_1D'])
        if verbose and first:
            print("1D:")
            if MONOMER:
                describe_monomer_results(Zmatrix, s[-1], vh[-1], M1, B1, K1, vals_set)
            plot_SVD_results(drive,R1_amp,R1_phase,R2_amp,R2_phase, df,  K1, K2, K12, B1, B2, FD, M1, M2, vals_set, 
                             MONOMER=MONOMER, forceboth=forceboth, labelcounts = labelcounts, overlay = overlay,
                             context = context, saving = saving, labelname = '1D', demo=demo,
                             resonatorsystem = resonatorsystem, show_set = show_set,
                             figsizeoverride1 = figsizeoverride1, figsizeoverride2 = figsizeoverride2) 
            plt.show()
            

        el = store_params(M1, M2, B1, B2, K1, K2, K12, FD, MONOMER)
                            
        theseresults.append(any(x<0 for x in el))
        theseresults_cols.append('any neg 1D')

        ## experimentalist's r^2 value
        # I want to find out whether R^2 is a predictor of syserr (controlling for SNR?)
        # R^2 based on experimentally accessible information.
        rsqrdresults1D, rsqrdcolumns1D = compile_rsqrd(R1_amp, R1_phase, R2_amp, R2_phase, 
              R1_real_amp, R1_im_amp, R2_real_amp, R2_im_amp,
              drive, K1, K2, K12, B1, B2, FD, M1, M2, MONOMER = MONOMER, forceboth = forceboth, label="1D")
        
        # calculate how close the SVD-determined parameters are compared to the originally set parameters
        syserrs = [syserr(el[i], vals_set[i]) for i in range(len(el))]

        # Values to compare:
        # Set values: k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set
        # SVD-determined values: M1, M2, B1, B2, K1, K2, K12, FD
        K1syserr = syserr(K1,k1_set)
        B1syserr = syserr(B1,b1_set)
        FDsyserr = syserr(FD,F_set)
        M1syserr = syserr(M1,m1_set)
        if MONOMER:
            K2syserr = 0
            K12syserr = 0
            B2syserr = 0
            M2syserr = 0
        else:
            K2syserr = syserr(K2,k2_set)
            K12syserr = syserr(K12,k12_set)
            B2syserr = syserr(B2,b2_set)
            M2syserr = syserr(M2,m2_set)
        avgsyserr, rmssyserr, maxsyserr, Lavgsyserr = combinedsyserr(syserrs,1) # subtract 1 degrees of freedom for 1D nullspace

        if MONOMER:
            theseresults.append([K1syserr,  B1syserr,  FDsyserr, M1syserr])
            theseresults_cols.append(['K1syserr%_1D', 'B1syserr%_1D','FDsyserr%_1D', 'M1syserr%_1D'])
        else:
            theseresults.append([K1syserr,K2syserr, K12syserr, B1syserr, B2syserr, FDsyserr, M1syserr, M2syserr])
            theseresults_cols.append(['K1syserr%_1D','K2syserr%_1D', 'K12syserr%_1D', 
                                      'B1syserr%_1D', 'B2syserr%_1D', 'FDsyserr%_1D', 
                                      'M1syserr%_1D', 'M2syserr%_1D'])
        theseresults.append([avgsyserr, rmssyserr, maxsyserr, Lavgsyserr])
        theseresults_cols.append(['avgsyserr%_1D', 'rmssyserr%_1D', 'maxsyserr%_1D', 'Lavgsyserr%_1D'])
        theseresults.append([np.log10(avgsyserr), np.log10(rmssyserr), np.log10(maxsyserr), np.log10(Lavgsyserr)])
        theseresults_cols.append(['log avgsyserr%_1D', 'log rmssyserr%_1D', 'log maxsyserr%_1D', 'log Lavgsyserr%_1D'])

        ### Normalize parameters in 2D nullspace 
        """ # Problem: res1 formula only for weak coupling.
        [M1_2D, M2_2D, B1_2D, B2_2D, K1_2D, K2_2D, K12_2D, FD_2D] = \
            normalize_parameters_to_res1_and_F_2d(vh, vals_set = vals_set)
        coefa = np.nan
        coefb = np.nan"""
        #[M1_2D, M2_2D, B1_2D, B2_2D, K1_2D, K2_2D, K12_2D, FD_2D], coefa, coefb = \
        #    normalize_parameters_to_m1_set_k1_set_assuming_2d(vh)
        #[M1_2D, M2_2D, B1_2D, B2_2D, K1_2D, K2_2D, K12_2D, FD_2D], coefa, coefb = \
        #    normalize_parameters_to_m1_m2_assuming_2d(vh, verbose = False, m1_set = m1_set, m2_set = m2_set)
        el_2D, coefa, coefb = \
            normalize_parameters_to_m1_F_set_assuming_2d(vh, MONOMER,verbose = False, m1_set = m1_set, F_set = F_set)
        #normalizationpair = 'm1 and F'
        
        if MONOMER:
            theseresults.append(el_2D)
            theseresults_cols.append(['M1_2D','B1_2D', 'K1_2D', 'FD_2D'])
        else:
            theseresults.append(el_2D)
            theseresults_cols.append(['M1_2D', 'M2_2D', 'B1_2D', 'B2_2D', 'K1_2D', 'K2_2D', 'K12_2D', 'FD_2D'])
        M1_2D, M2_2D, B1_2D, B2_2D, K1_2D, K2_2D, K12_2D, FD_2D = read_params(el_2D, MONOMER=MONOMER)
            
        if verbose and first:
            print("2D:")
            plot_SVD_results(drive,R1_amp,R1_phase,R2_amp,R2_phase, df, 
                             K1_2D, K2_2D, K12_2D, B1_2D, B2_2D, FD_2D, M1_2D, M2_2D, vals_set,
                             MONOMER=MONOMER, forceboth=forceboth, labelcounts = labelcounts, overlay=overlay,
                             context = context,saving = saving, labelname = '2D', demo=demo,
                             resonatorsystem = resonatorsystem,  show_set = show_set,
                             figsizeoverride1 = figsizeoverride1, figsizeoverride2 = figsizeoverride2)

            plt.show()
                            
        theseresults.append(any(x<0 for x in el_2D))
        theseresults_cols.append('any neg 2D')
            
        # I want to find out whether R^2 is a predictor of syserr (controlling for SNR?)
        # R^2 based on experimentally accessible information.
        rsqrdresults2D, rsqrdcolumns2D = compile_rsqrd(R1_amp, R1_phase, R2_amp, R2_phase, R1_real_amp, R1_im_amp, R2_real_amp, R2_im_amp,
              drive, K1_2D, K2_2D, K12_2D, B1_2D, B2_2D, FD_2D, M1_2D, M2_2D, MONOMER = MONOMER, forceboth = forceboth, label="2D")
        
        syserrs_2D = [syserr(el_2D[i], vals_set[i]) for i in range(len(el_2D))]

        # Values to compare:
        # Set values: k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set
        # SVD-determined values: M1, M2, B1, B2, K1, K2, K12, FD

        K1syserr_2D = syserr(K1_2D,k1_set)
        B1syserr_2D = syserr(B1_2D,b1_set)
        FDsyserr_2D = syserr(FD_2D,F_set)
        M1syserr_2D = syserr(M1_2D,m1_set)

        if MONOMER:
            K2syserr_2D = 0
            K12syserr_2D = 0
            B2syserr_2D = 0
            M2syserr_2D = 0
        else:
            K2syserr_2D = syserr(K2_2D,k2_set)
            K12syserr_2D = syserr(K12_2D,k12_set)
            B2syserr_2D = syserr(B2_2D,b2_set)
            M2syserr_2D = syserr(M2_2D,m2_set)
        if MONOMER:
            theseresults.append([K1syserr_2D,B1syserr_2D,FDsyserr_2D,M1syserr_2D])
            theseresults_cols.append(['K1syserr%_2D','B1syserr%_2D','FDsyserr%_2D','M1syserr%_2D'])           
        else:
            theseresults.append([K1syserr_2D,B1syserr_2D,FDsyserr_2D,M1syserr_2D,
                                 K2syserr_2D,K12syserr_2D,B2syserr_2D,M2syserr_2D])
            theseresults_cols.append(['K1syserr%_2D','B1syserr%_2D','FDsyserr%_2D','M1syserr%_2D',
                                      'K2syserr%_2D','K12syserr%_2D','B2syserr%_2D','M2syserr%_2D'])

        avgsyserr_2D, rmssyserr_2D, maxsyserr_2D, Lavgsyserr_2D = combinedsyserr(syserrs_2D,2) # subtract 2 degrees of freedom for 2D nullspace
        theseresults.append([avgsyserr_2D, rmssyserr_2D, maxsyserr_2D, Lavgsyserr_2D])
        theseresults_cols.append(['avgsyserr%_2D', 'rmssyserr%_2D', 'maxsyserr%_2D',  'Lavgsyserr%_2D'])
        theseresults.append([np.log10(avgsyserr_2D), np.log10(rmssyserr_2D), np.log10(maxsyserr_2D), np.log10(Lavgsyserr_2D)])
        theseresults_cols.append(['log avgsyserr%_2D', 'log rmssyserr%_2D', 'log maxsyserr%_2D', 'log Lavgsyserr%_2D'])
        
        theseresults.append(avgsyserr-avgsyserr_2D)
        theseresults_cols.append('avgsyserr%_1D-avgsyserr%_2D')
        theseresults.append(np.log10(avgsyserr)-np.log10(avgsyserr_2D))
        theseresults_cols.append('log avgsyserr%_1D - log avgsyserr%_2D')
        
        ## 3D normalization.
        if MONOMER:
            el_3D, coefa, coefb, coefc = normalize_parameters_assuming_3d(vh,vals_set, MONOMER) 
        else:
            el_3D, coefa, coefb, coefc = normalize_parameters_assuming_3d(vh, vals_set, MONOMER=MONOMER)
        el_3D = [parameter.real for parameter in el_3D if parameter.imag == 0 ]
        
        if MONOMER:
            theseresults.append(el_3D)
            theseresults_cols.append(['M1_3D','B1_3D', 'K1_3D', 'FD_3D'])
        else:
            theseresults.append(el_3D)
            theseresults_cols.append(['M1_3D', 'M2_3D', 'B1_3D', 'B2_3D', 'K1_3D', 'K2_3D', 'K12_3D', 'FD_3D'])
        M1_3D, M2_3D, B1_3D, B2_3D, K1_3D, K2_3D, K12_3D, FD_3D = read_params(el_3D, MONOMER=MONOMER)
                            
        if verbose and first:
            print("3D:")
            plot_SVD_results(drive,R1_amp,R1_phase,R2_amp,R2_phase, df, 
                             K1_3D, K2_3D, K12_3D, B1_3D, B2_3D, FD_3D, M1_3D, M2_3D, vals_set,
                             MONOMER=MONOMER, forceboth=forceboth, labelcounts = labelcounts, overlay=overlay,
                             context = context,saving = saving, labelname = '3D', demo=demo,
                             resonatorsystem = resonatorsystem,  show_set = show_set,
                             figsizeoverride1 = figsizeoverride1, figsizeoverride2 = figsizeoverride2)

            plt.show()
            first = False
                            
        theseresults.append(any(x<0 for x in el_3D))
        theseresults_cols.append('any neg 3D')
            
        # I want to find out whether R^2 is a predictor of syserr (controlling for SNR?)
        # R^2 based on experimentally accessible information.
        rsqrdresults3D, rsqrdcolumns3D = compile_rsqrd(R1_amp, R1_phase, R2_amp, R2_phase, 
              R1_real_amp, R1_im_amp, R2_real_amp, R2_im_amp,
              drive, K1_3D, K2_3D, K12_3D, B1_3D, B2_3D, FD_3D, M1_3D, M2_3D, 
              MONOMER = MONOMER, forceboth = forceboth, label="3D")
                        
        syserrs_3D = [syserr(el_3D[i], vals_set[i]) for i in range(len(el_3D))]

        # Values to compare:
        # Set values: k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set
        # SVD-determined values: M1, M2, B1, B2, K1, K2, K12, FD

        K1syserr_3D = syserr(K1_3D,k1_set)
        B1syserr_3D = syserr(B1_3D,b1_set)
        FDsyserr_3D = syserr(FD_3D,F_set)
        M1syserr_3D = syserr(M1_3D,m1_set)

        if MONOMER:
            K2syserr_3D = 0
            K12syserr_3D = 0
            B2syserr_3D = 0
            M2syserr_3D = 0
        else:
            K2syserr_3D = syserr(K2_3D,k2_set)
            K12syserr_3D = syserr(K12_3D,k12_set)
            B2syserr_3D = syserr(B2_3D,b2_set)
            M2syserr_3D = syserr(M2_3D,m2_set)
        if MONOMER:
            theseresults.append([K1syserr_3D,B1syserr_3D,FDsyserr_3D,M1syserr_3D])
            theseresults_cols.append(['K1syserr%_3D','B1syserr%_3D','FDsyserr%_3D','M1syserr%_3D'])           
        else:
            theseresults.append([K1syserr_3D,B1syserr_3D,FDsyserr_3D,M1syserr_3D,
                                 K2syserr_3D,K12syserr_3D,B2syserr_3D,M2syserr_3D])
            theseresults_cols.append(['K1syserr%_3D','B1syserr%_3D','FDsyserr%_3D','M1syserr%_3D',
                                      'K2syserr%_3D','K12syserr%_3D','B2syserr%_3D','M2syserr%_3D'])

        avgsyserr_3D, rmssyserr_3D, maxsyserr_3D, Lavgsyserr_3D = \
            combinedsyserr(syserrs_3D,3) # subtract 3 degrees of freedom for 3D nullspace
        theseresults.append([avgsyserr_3D, rmssyserr_3D, maxsyserr_3D, Lavgsyserr_3D])
        theseresults_cols.append(['avgsyserr%_3D', 'rmssyserr%_3D', 'maxsyserr%_3D', 'Lavgsyserr%_3D'])
        theseresults.append([np.log10(avgsyserr_3D), np.log10(rmssyserr_3D), np.log10(maxsyserr_3D), np.log10(Lavgsyserr_3D)])
        theseresults_cols.append(['log avgsyserr%_3D', 'log rmssyserr%_3D', 'log maxsyserr%_3D', 'log Lavgsyserr%_3D'])
        
        theseresults.append(avgsyserr-avgsyserr_3D)
        theseresults_cols.append('avgsyserr%_1D-avgsyserr%_3D')
        theseresults.append(np.log10(avgsyserr)-np.log10(avgsyserr_3D))
        theseresults_cols.append('log avgsyserr%_1D - log avgsyserr%_3D')
                            
        theseresults.append(avgsyserr_2D-avgsyserr_3D)
        theseresults_cols.append('avgsyserr%_2D-avgsyserr%_3D')
        theseresults.append(np.log10(avgsyserr_2D)-np.log10(avgsyserr_3D))
        theseresults_cols.append('log avgsyserr%_2D - log avgsyserr%_3D')
        
        theseresults.append(len(drive))
        theseresults_cols.append('num_freq_used_for_rsqrd')
        theseresults.append(rsqrdresults1D)
        theseresults_cols.append(rsqrdcolumns1D)
        theseresults.append(rsqrdresults2D)
        theseresults_cols.append(rsqrdcolumns2D)
        theseresults.append(rsqrdresults3D)
        theseresults_cols.append(rsqrdcolumns3D)
    
        
        if MONOMER: # same for every row
            theseresults.append([maxSNR_R1, minSNR_R1,meanSNR_R1])
            theseresults_cols.append(['maxSNR_R1', 'minSNR_R1','meanSNR_R1'])
            theseresults.append(list(np.log10([maxSNR_R1, minSNR_R1,meanSNR_R1])))
            theseresults_cols.append(
                ['log ' + s for s in ['maxSNR_R1', 'minSNR_R1','meanSNR_R1']])
        else:
            theseresults.append([maxSNR_R1,maxSNR_R2, minSNR_R1,minSNR_R2, meanSNR_R1,meanSNR_R2])
            theseresults_cols.append(['maxSNR_R1','maxSNR_R2',  'minSNR_R1','minSNR_R2','meanSNR_R1','meanSNR_R2'])
            
            theseresults.append(list(np.log10([maxSNR_R1,maxSNR_R2, minSNR_R1,minSNR_R2, meanSNR_R1,meanSNR_R2])))
            theseresults_cols.append(
                ['log ' + s for s in ['maxSNR_R1','maxSNR_R2',  'minSNR_R1','minSNR_R2','meanSNR_R1','meanSNR_R2']])
            
        #theseresults.append([len(p), drive[p]]) # same for every row
        #theseresults_cols.append([ 'num frequency points','frequencies'])

        results.append(flatten(theseresults))
        assert_results_length(results=theseresults,columns = theseresults_cols)


    assert (len(flatten(theseresults)) == len(flatten(theseresults_cols)))
    resultsdf = pd.DataFrame(
            data=results, 
            columns = flatten(theseresults_cols))
    return resultsdf