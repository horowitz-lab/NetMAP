import numpy as np
from spectrum_SVD_plotter import *
from characterizeresonator import *
from resonatorfrequencypicker import *
from resonatorsimulator import *
from helperfunctions import *

def simulated_experiment(measurementfreqs, 
                         vals_set, noiselevel, MONOMER, forceboth, 
						 drive=np.linspace(minfreq,maxfreq,n), 
                         verbose = True, repeats=1, 
                         noiseless_spectra = None, noisy_spectra = None):
    
    if verbose:
        print('Running simulated_experiment()', repeats, 'times.')
    
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

    maxSNR_R1,maxSNR_R2, minSNR_R1,minSNR_R2,meanSNR_R1,meanSNR_R2, SNR_R1_list, SNR_R2_list = SNRs( \
            drive[p],vals_set, noiselevel, use_complexnoise)
    
    first = True
    results = []

    for i in range(repeats):
        theseresults = []
        theseresults_cols = []
        
        theseresults.append(len(p))
        theseresults_cols.append([ 'num frequency points'])
        
        theseresults.append(vals_set) # Store vals_set # same for every row
        if MONOMER:
            theseresults_cols.append(['m1_set',  'b1_set',  'k1_set', 'F_set'])
        else:
            theseresults_cols.append(['m1_set', 'm2_set', 'b1_set', 'b2_set', 'k1_set', 'k2_set', 'k12_set', 'F_set'])
        theseresults.append(noiselevel)
        theseresults_cols.append('noiselevel')
        
        if noisy_spectra is not None and i == 0:
            R1_amp, R1_phase, R2_amp, R2_phase, R1_real_amp, R1_im_amp, R2_real_amp, R2_im_amp,_ = noisy_spectra
        else:
            # recalculate noisy spectra
            R1_amp, R1_phase, R2_amp, R2_phase, R1_real_amp, R1_im_amp, R2_real_amp, R2_im_amp,_ = calculate_spectra(
                drive, vals_set, noiselevel=noiselevel, forceboth=forceboth)
        
        if len(p) < 40:
            for i in range(len(p)):
                theseresults.append([drive[p[i]], 
                                     R1_phase_noiseless[p[i]], R2_phase_noiseless[p[i]],
                                    R1_amp_noiseless[p[i]], R2_amp_noiseless[p[i]]])
                theseresults_cols.append(['Freq' + str(i+1), 
                                          'R1_phase_noiseless' + str(i+1),'R2_phase_noiseless' + str(i+1),
                                          'R1_amp_noiseless' + str(i+1),'R2_amp_noiseless' + str(i+1)])                   
                theseresults.append(SNR_R1_list[i])
                theseresults_cols.append('SNR_R1_f' + str(i+1))
                theseresults.append(SNR_R2_list[i])
                theseresults_cols.append('SNR_R2_f' + str(i+1))
                theseresults.append(R1_amp[p[i]])
                theseresults_cols.append('R1_amp_meas' + str(i+1))
                theseresults.append(R2_amp[p[i]])
                theseresults_cols.append('R2_amp_meas' + str(i+1))                
        
        if len(p) == 2:
            theseresults.append(drive[p[1]] - drive[p[0]])
            theseresults_cols.append('Difference')
            theseresults.append(R1_phase_noiseless[p[1]] - R1_phase_noiseless[p[0]])
            theseresults_cols.append('R1_phase_diff')
            theseresults.append(R2_phase_noiseless[p[1]] - R2_phase_noiseless[p[0]])
            theseresults_cols.append('R2_phase_diff')
            
        theseresults.append(drive[p])
        theseresults_cols.append('frequencies')

        df = measurementdfcalc(drive, p, 
                     R1_amp=R1_amp,R2_amp=R2_amp,R1_phase=R1_phase, R2_phase=R2_phase, 
                     R1_amp_noiseless=R1_amp_noiseless,R2_amp_noiseless=R2_amp_noiseless,
                     R1_phase_noiseless=R1_phase_noiseless, R2_phase_noiseless=R2_phase_noiseless
                     )
        Zmatrix = Zmat(df, frequencycolumn = 'drive', complexamplitude1 = 'R1AmpCom', complexamplitude2 = 'R2AmpCom', 
                       MONOMER=MONOMER, forceboth=forceboth)
        u, s, vh = np.linalg.svd(Zmatrix, full_matrices = True)
        vh = make_real_iff_real(vh)
            
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
        
        ## 1D NULLSPACE
        M1, M2, B1, B2, K1, K2, K12, FD = read_params(vh[-1], MONOMER) # the 7th singular value is the smallest one (closest to zero)

        # normalize parameters vector to the force, assuming 1D nullspace
        allparameters = normalize_parameters_1d_by_force([M1, M2, B1, B2, K1, K2, K12, FD], F_set)
        # recast as real, not complex # but real gets a warning
        allparameters = [thisparameter.real for thisparameter in allparameters if thisparameter.imag == 0 ]
        M1, M2, B1, B2, K1, K2, K12, FD = allparameters
        
        if MONOMER:
            theseresults.append([M1, B1,  K1, FD])
            theseresults_cols.append(['M1_1D', 'B1_1D','K1_1D', 'FD_1D'])            
        else:
            theseresults.append([M1, M2, B1, B2, K1, K2, K12, FD])
            theseresults_cols.append(['M1_1D', 'M2_1D', 'B1_1D', 'B2_1D', 'K1_1D', 'K2_1D', 'K12_1D', 'FD_1D'])
        if verbose and first:
            print("1D:")
            plot_SVD_results(drive,R1_amp,R1_phase,R2_amp,R2_phase, df,  K1, K2, K12, B1, B2, FD, M1, M2, vals_set, 
                             MONOMER=MONOMER, forceboth=forceboth) # %%%
            plt.show()

        el = store_params(M1, M2, B1, B2, K1, K2, K12, FD, MONOMER)

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
            theseresults_cols.append(['K1syserr%_1D','K2syserr%_1D', 'K12syserr%_1D', 'B1syserr%_1D', 'B2syserr%_1D', 'FDsyserr%_1D', 'M1syserr%_1D', 'M2syserr%_1D'])
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
            normalize_parameters_to_m1_F_set_assuming_2d(vh, verbose = False, m1_set = m1_set, F_set = F_set)
        normalizationpair = 'm1 and F'
        
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
                             MONOMER=MONOMER, forceboth=forceboth)
            plt.show()
            
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
            el_3D, coefa, coefb, coefc = normalize_parameters_assuming_3d(vh, vals_set, 0,1,2, MONOMER=MONOMER)
        else:
            el_3D, coefa, coefb, coefc = normalize_parameters_assuming_3d(vh, MONOMER=MONOMER)
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
                             MONOMER=MONOMER, forceboth=forceboth)
            plt.show()
            first = False
            
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

        avgsyserr_3D, rmssyserr_3D, maxsyserr_3D, Lavgsyserr_3D = combinedsyserr(syserrs_3D,3) # subtract 3 degrees of freedom for 3D nullspace
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

    assert (len(flatten(theseresults)) == len(flatten(theseresults_cols)))
    resultsdf = pd.DataFrame(
            data=results, 
            columns = flatten(theseresults_cols))
    return resultsdf