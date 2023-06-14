# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:07:55 2022

@author: vhorowit
"""

import numpy as np
import resonatorphysics
from resonatorphysics import res_freq_weak_coupling, calcnarrowerW
from helperfunctions import read_params
import matplotlib.pyplot as plt
from resonatorsimulator import \
    curve1, theta1, curve2, theta2#, realamp1, imamp1, realamp2, imamp2
from scipy.signal import find_peaks

# default settings
verbose = False
n=100
debug = False

""" Given a limited set of available frequencies called "drive", 
find those indices that most closely correspond to the desired frequencies. 
This will not throw an err if two are the same; that could be added by checking if nunique is a shorter length. """
def freqpoints(desiredfreqs, drive):
    p = [] # p stands for frequency points; these are the indicies of frequencies that we will be measuring.
    for f in desiredfreqs:
        absolute_val_array = np.abs(drive - f)
        f_index = absolute_val_array.argmin()
        p.append(f_index)
    return p


""" drive and phase are two lists of the same length 
    This will only return one frequency for each requested angle, even if there are additional solutions.
    It's helpful if drive is morefrequencies.
"""
def find_freq_from_angle(drive, phase, angleswanted = [-np.pi/4], returnindex = False, verbose = False):
    assert len(drive) == len(phase)
    
    #specialanglefreq = [drive[np.argmin(abs(phase%(2*np.pi)  - anglewanted%(2*np.pi)))] \
    #                        for anglewanted in angleswanted ]   
    
    threshold = np.pi/30 # small angle threshold
    specialanglefreq = [] # initialize list
    indexlist = []
    for anglewanted in angleswanted:
        index = np.argmin(abs(phase%(2*np.pi)  - anglewanted%(2*np.pi))) # find where phase is closest
               
        if index == 0 or index >= len(drive)-1: # edges of dataset require additional scrutiny
            ## check to see if it's actually close after all
            nearness = abs(phase[index]%(2*np.pi)-anglewanted%(2*np.pi))
            if nearness > threshold:
                continue # don't include this index
        specialanglefreq.append(drive[index])
        indexlist.append(index)
    
    if False:
        plt.figure()
        plt.plot(specialanglefreq,phase[indexlist]/np.pi)
        plt.xlabel('Freq')
        plt.ylabel('Angle (pi)')
        
    if returnindex:
        return specialanglefreq, indexlist
    else:
        return specialanglefreq  
    
""" n is the number of frequencies is the drive; we'll have more for more frequencies. 
 Can you improve this by calling create_drive_arrays afterward? """
def makemorefrequencies(vals_set, minfreq, maxfreq,MONOMER,forceboth,
                        res1 = None, res2 = None, 
                        includefreqs = None, n=n, staywithinlims = False):
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
    
    if res1 is None:
        res1 = res_freq_weak_coupling(k1_set, m1_set, b1_set)
    if not MONOMER and res2 is None:
        res2 = res_freq_weak_coupling(k2_set, m2_set, b2_set)
    
    morefrequencies = np.linspace(minfreq, maxfreq, num = n*60)
    if MONOMER:
        morefrequencies = np.append(morefrequencies, [res1])
    else:
        morefrequencies = np.append(morefrequencies, [res1,res2])
    
    if includefreqs is not None:
        morefrequencies = np.append(morefrequencies, np.array(includefreqs))
    
    try:
        W1 = resonatorphysics.approx_width(k = k1_set, m = m1_set, b=b1_set)
    except ZeroDivisionError:
        print('k1_set:', k1_set)
        print('m1_set:', m1_set)
        print('b1_set:', b1_set)
        W1 = (maxfreq - minfreq)/5
    morefrequencies = np.append(morefrequencies, np.linspace(res1-W1, res1+W1, num = 7*n))
    morefrequencies = np.append(morefrequencies, np.linspace(res1-2*W1, res1+2*W1, num = 10*n)) 
    if not MONOMER:
        W2 = resonatorphysics.approx_width(k = k2_set, m = m2_set, b=b2_set)
        morefrequencies = np.append(morefrequencies, np.linspace(res2-W2, res2+W2, num = 7*n))
        morefrequencies = np.append(morefrequencies, np.linspace(res2-2*W2, res2+2*W2, num = 10*n))
    morefrequencies = list(np.sort(np.unique(morefrequencies)))
    
    while morefrequencies[0] < 0:
        morefrequencies.pop(0)
        
    if staywithinlims:
        while morefrequencies[0] < minfreq:
            morefrequencies.pop(0)
        while morefrequencies[-1] > maxfreq:
            morefrequencies.pop(-1)
        
    return np.array(morefrequencies)
    

def create_drive_arrays(vals_set, MONOMER, forceboth, n=n, 
                        morefrequencies = None, 
                        minfreq = None, maxfreq = None, 
                        staywithinlims = False,
                        includefreqs = [],
                        callmakemore = False,
                        verbose = verbose):
    
    if verbose:
        print('Running create_drive_arrays()')
        
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
    
    if morefrequencies is None:
        if minfreq is None:
            minfreq = 0.1
        if maxfreq is None:
            maxfreq = 5
        morefrequencies=np.linspace(minfreq,maxfreq,50*n)
    if minfreq is None:
        minfreq = min(morefrequencies)
    if maxfreq is None:
        maxfreq = max(morefrequencies)
    
    if minfreq <= 0:
        minfreq = 1e-6
        
    if callmakemore:
        evenmore = makemorefrequencies(vals_set=vals_set, minfreq=minfreq, maxfreq=maxfreq,MONOMER=MONOMER,forceboth=forceboth,
                        res1 = None, res2 = None, 
                        includefreqs = None, n=n, staywithinlims = staywithinlims)
        morefrequencies = np.sort(np.unique(np.append(morefrequencies, evenmore)))
       
    Q1 = resonatorphysics.approx_Q(k1_set, m1_set, b1_set)

    # set the fraction of points that are spread evenly in frequency (versus evenly in phase)
    if MONOMER:
        if Q1 >= 30:
            fracevenfreq = .2
        elif Q1 >=10:
            fracevenfreq = .4   
        else:
            fracevenfreq = .5 # such a broad peak that we might as well spread evenly in frequency
    else:
        Q2 = resonatorphysics.approx_Q(k2_set, m2_set, b2_set)
        if Q1 >=30 and Q2 >= 30:
            fracevenfreq = .2
        else:
            fracevenfreq = .4
    if MONOMER:
        # choose length of anglelist
        m = n-3-int(fracevenfreq*n) # 3 are special angles; 20% are evenly spaced freqs
    else:
        m = int((n-3-(fracevenfreq*n))/2)
    
    morefrequencies = list(np.sort(morefrequencies))
    while morefrequencies[-1] > maxfreq:
        if False: # too verbose!
            print('Removing frequency', morefrequencies[-1])
        morefrequencies = morefrequencies[:-1]
    while morefrequencies[0]< minfreq:
        if False:
            print('Removing frequency', morefrequencies[0])
        morefrequencies = morefrequencies[1:]
    
    phaseR1 = theta1(morefrequencies, k1_set, k2_set, k12_set, 
                                        b1_set, b2_set, F_set, m1_set, m2_set, 
                                         0, MONOMER, forceboth=forceboth)
    
    anglelist = np.linspace(min(phaseR1), max(phaseR1), m) ## most of the points are evenly spaced in phase
    #anglelist = np.append(anglelist, -np.pi/3)
    #anglelist = np.append(anglelist, -2*np.pi/3)
    anglelist = np.append(anglelist, -np.pi/4) # special angle 1
    anglelist = np.append(anglelist, -3*np.pi/4) # special angle 2
    anglelist = np.unique(np.sort(np.append(anglelist, -np.pi/2))) # special angle 3
    
    freqlist = find_freq_from_angle(morefrequencies, 
                      phase = phaseR1,
                      angleswanted = anglelist, verbose = verbose)
    if False:
        print('anglelist/pi:', anglelist/np.pi, 'corresponds to frequency list:', freqlist, '. But still adding to chosendrive.')
    
    if not MONOMER:
        phaseR2 = theta2(morefrequencies, k1_set, k2_set, k12_set, 
                                             b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)
        
        del anglelist
        anglelist = np.linspace(min(phaseR2), max(phaseR2), m) 
        #anglelist = np.append(anglelist, -np.pi/3)
        #anglelist = np.append(anglelist, -2*np.pi/3)
        anglelist = np.append(anglelist, -np.pi/4)
        anglelist = np.append(anglelist, -3*np.pi/4)
        anglelist = np.unique(np.sort(np.append(anglelist, -np.pi/2)))
        
        freqlist2 = find_freq_from_angle(morefrequencies,
                                phase = phaseR2,
                                angleswanted = anglelist)
        if False:
            print('anglelist/pi: ', anglelist/np.pi)
            print('freqlist2: ', freqlist2)
        freqlist.extend(freqlist2)
        res2 = res_freq_weak_coupling(k2_set, m2_set, b2_set)
        freqlist.append(res2)
        morefrequencies = np.append(morefrequencies,res2)    
        
    freqlist.extend(includefreqs)
    morefrequencies = np.append(morefrequencies, includefreqs)
    res1 = res_freq_weak_coupling(k1_set, m1_set, b1_set)
    freqlist.append(res1)
    try:
        reslist = res_freq_numeric(vals_set=vals_set, MONOMER=MONOMER, mode = 'all', forceboth=forceboth,
                                   minfreq=minfreq, maxfreq=maxfreq, morefrequencies=morefrequencies,
                                   unique = True, veryunique = True, verboseplot = False, verbose=verbose, iterations = 3)
        freqlist.extend(reslist)
    except NameError:
        pass
    
    freqlist = list(np.sort(np.unique(freqlist)))
    
    while freqlist[0] < 0:
        freqlist.pop(0) # drop negative frequencies
        
    numwanted = n-len(freqlist) # how many more frequencies are wanted?
    evenlyspacedfreqlist = np.linspace(minfreq, maxfreq, 
                                       num = max(numwanted + 2,3)) #  I added 2 for the endpoints
    freqlist.extend(evenlyspacedfreqlist)
    #print(freqlist)
    chosendrive = list(np.sort(np.unique(np.array(freqlist))))
    
    if staywithinlims:
        while chosendrive[0] < minfreq or chosendrive[0] < 0:
            f = chosendrive.pop(0)
            if verbose:
                print('Warning: Unexpected frequency', f)
        while chosendrive[-1] > maxfreq:
            f = chosendrive.pop(-1)
            if verbose:
                print('Warning: Unexpected frequency', f)
    else:
        while chosendrive[0] < 0:
            f = chosendrive.pop(0)
            print('Warning: Unexpected negative frequency', f)
    chosendrive = np.array(chosendrive)
    
    #morefrequencies.extend(chosendrive)
    morefrequencies = np.concatenate((morefrequencies, chosendrive))
    morefrequencies = list(np.sort(np.unique(morefrequencies)))
    
    if staywithinlims:
        while morefrequencies[0] < minfreq:
            f = morefrequencies.pop(0)
            print('Warning: Unexpected frequency', f)
        while morefrequencies[-1] > maxfreq:
            f = morefrequencies.pop(-1)
            print('warning: Unexpected frequency', f)
    
    return chosendrive, np.array(morefrequencies)

def find_special_freq(drive, amp, phase, anglewanted = np.radians(225)):
    maxampfreq = drive[np.argmax(amp)]
    specialanglefreq = drive[np.argmin(abs(phase%(2*np.pi) - anglewanted%(2*np.pi)))]
    return maxampfreq, specialanglefreq


### res_freq_numeric()
## Uses privilege
## Not guaranteed to find all resonance peaks but should work ok for dimer
## Returns list of peak frequencies. 
## If numtoreturn is None, then any number of frequencies could be returned.
## You can also set numtoreturn to 1 or 2 to return that number of frequencies.
def res_freq_numeric(vals_set, MONOMER, forceboth,
                     mode = 'all',
                     minfreq=.1, maxfreq=5, morefrequencies=None, includefreqs = [],
                     unique = True, veryunique = True, numtoreturn = None, 
                     verboseplot = False, plottitle = None, verbose=verbose, iterations = 1,
                     use_R2_only = False,
                     returnoptions = False):
    
    if verbose:
        print('\nRunning res_freq_numeric() with mode ' + mode)
        if plottitle is not None:
            print(plottitle)
    m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set = read_params(vals_set, MONOMER)
    
    if MONOMER and numtoreturn != 2:   # 2 is a tricky case... just use the rest of the algorithm    
        if numtoreturn is not None and numtoreturn != 1:
            print('Cannot return ' + str(numtoreturn) + ' res freqs for Monomer.')
        if verbose:
            print('option 1')
            
        freqlist = [res_freq_weak_coupling(k1_set, m1_set, b1_set)] # just compute it directly for Monomer
        if returnoptions:
            return freqlist, 1
        return freqlist
    
    approx_res_freqs = [res_freq_weak_coupling(k1_set, m1_set, b1_set)]
    if not MONOMER:
        approx_res_freqs.append(res_freq_weak_coupling(k2_set, m2_set, b2_set))
        
    for f in approx_res_freqs:
        if f > maxfreq or f < minfreq:
            print('Warning! Check minfreq and maxfreq')
            print('minfreq', minfreq)
            print('maxfreq', maxfreq)
            print('Approx resonant freq', f)
        
    if morefrequencies is None:
        morefrequencies = makemorefrequencies(vals_set=vals_set, minfreq=minfreq, maxfreq=maxfreq,
                                                  forceboth=forceboth, includefreqs = approx_res_freqs,
                                                  MONOMER=MONOMER, n=n)
    else:
        morefrequencies = np.append(morefrequencies, approx_res_freqs)
    morefrequencies = np.sort(np.unique(morefrequencies))
    
    # init
    indexlist = []
    
    if MONOMER:
        freqlist = [res_freq_weak_coupling(k1_set, m1_set, b1_set)]
        resfreqs_from_amp = freqlist
    else:
        first = True
        for i in range(iterations):
            if not first: # not first. This is a repeated iteration. indexlist has been defined.
                if verbose:
                    print('indexlist:', indexlist)
                    if max(indexlist) > len(morefrequencies):
                        print('len(morefrequencies):', len(morefrequencies))
                    print('morefrequencies:', morefrequencies)
                    print('indexlist:', indexlist)
                    print('Repeating with finer frequency mesh around frequencies:', morefrequencies[np.sort(indexlist)])

                assert min(morefrequencies) >= minfreq
                assert max(morefrequencies) <= maxfreq
                if debug:
                    print('minfreq', minfreq)
                    print('Actual min freq', min(morefrequencies))
                    print('maxfreq', maxfreq)
                    print('Actual max freq', max(morefrequencies))
                morefrequenciesprev = morefrequencies.copy()
                for index in indexlist:
                    try:
                        spacing = abs(morefrequenciesprev[index] - morefrequenciesprev[index-1])
                    except:
                        if verbose:
                            print('morefrequenciesprev:',morefrequenciesprev)
                            print('index:', index)
                        spacing = abs(morefrequenciesprev[index+1] - morefrequenciesprev[index])
                    finerlist = np.linspace(max(minfreq,morefrequenciesprev[index]-spacing), 
                                            min(maxfreq,morefrequenciesprev[index] + spacing), 
                                            num = n)
                    assert min(finerlist) >= minfreq
                    assert max(finerlist) <= maxfreq
                    morefrequencies = np.append(morefrequencies,finerlist)
                morefrequencies = np.sort(np.unique(morefrequencies))


            while morefrequencies[-1] > maxfreq:
                if False: # too verbose!
                    print('Removing frequency', morefrequencies[-1])
                morefrequencies = morefrequencies[:-1]
            while morefrequencies[0]< minfreq:
                if False:
                    print('Removing frequency', morefrequencies[0])
                morefrequencies = morefrequencies[1:]
            R1_amp_noiseless = curve1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                      0, MONOMER, forceboth=forceboth)
            R1_phase_noiseless = theta1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                        0, MONOMER, forceboth=forceboth)
            R1_phase_noiseless = np.unwrap(R1_phase_noiseless)
            if debug:
                plt.figure()
                plt.plot(morefrequencies, R1_amp_noiseless, label = 'R1_amp')
                plt.plot(morefrequencies, R1_phase_noiseless, label = 'R1_phase')
            if not MONOMER:
                R2_amp_noiseless = curve2(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                          0, forceboth=forceboth)
                R2_phase_noiseless = theta2(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                            0, forceboth=forceboth)
                R2_phase_noiseless = np.unwrap(R2_phase_noiseless)
                if debug:
                    plt.plot(morefrequencies, R2_amp_noiseless, label = 'R2_amp')
                    plt.plot(morefrequencies, R2_phase_noiseless, label = 'R2_phase')

            ## find maxima
            index1 = np.argmax(R1_amp_noiseless)
            if not MONOMER and not use_R2_only:
                indexlist1, heights = find_peaks(R1_amp_noiseless, height=.015, distance = 5)
                if debug:
                    print('index1:', index1)
                    print('indexlist1:',indexlist1)
                    print('heights', heights)
                    plt.axvline(morefrequencies[index1])
                    for i in indexlist1:
                        plt.axvline(morefrequencies[i])
                assert index1 <= len(morefrequencies)
                if len(indexlist1)>0:
                    assert max(indexlist1) <= len(morefrequencies)
                else:
                    print('Warning: find_peaks on R1_amp returned indexlist:', indexlist1)
                    plt.figure()
                    plt.plot(R1_amp_noiseless)
                    plt.xlabel(R1_amp_noiseless)
                    plt.figure()
            else:
                indexlist1 = []
            if MONOMER:
                indexlist2 = []
            else:
                index2 = np.argmax(R2_amp_noiseless)
                indexlist2, heights2 = find_peaks(R2_amp_noiseless, height=.015, distance = 5)
                assert index2 <= len(morefrequencies)
                if len(indexlist2) >0:
                    assert max(indexlist2) <= len(morefrequencies)

            if verbose:
                print('Maximum amplitude for R1 is ',  R1_amp_noiseless[index1], 'at', morefrequencies[index1])
                if not MONOMER:
                    print('Maximum amplitude for R2 is ',  R2_amp_noiseless[index2], 'at', morefrequencies[index2])

            indexlistampR1 = np.append(indexlist1,index1)
            assert max(indexlistampR1) <= len(morefrequencies)
            if False: # too verbose!
                print('indexlistampR1:', indexlistampR1)
            if MONOMER:
                indexlist = indexlistampR1
                assert max(indexlist) <= len(morefrequencies)
                indexlistampR2 = []
            else:
                indexlistampR2 = np.append(indexlist2, index2)
                if False:
                    print('indexlistampR2:',indexlistampR2)
                assert max(indexlistampR2) <= len(morefrequencies)
                indexlist = np.append(indexlistampR1, indexlistampR2)
                if False:
                    print('indexlist:', indexlist)

            assert max(indexlist) <= len(morefrequencies)
            indexlist = list(np.unique(indexlist))
            indexlist = [int(index) for index in indexlist]
            first = False
    
        ## Check to see if findpeaks just worked
        if (numtoreturn == 2) and (mode != 'phase'):
            thresh = .006
            if len(indexlist2) == 2:
                if verbose:
                    print("Used findpeaks on R2 amplitude (option 2)")
                opt2freqlist = list(np.sort(morefrequencies[indexlist2]))
                if abs(opt2freqlist[1]-opt2freqlist[0]) > thresh:
                    if returnoptions:
                        return opt2freqlist, 2
                    return opt2freqlist
            if len(indexlist1) == 2 and not use_R2_only:
                opt3freqlist = list(np.sort(morefrequencies[indexlist1]))
                if abs(opt3freqlist[1]-opt3freqlist[0]) > thresh:
                    if verbose:
                        print("Used findpeaks on R1 amplitude (option 3)")
                    if returnoptions:
                        return opt3freqlist, 3
                    return opt3freqlist
            if verbose:
                print('indexlist1 from R1 amp find_peaks is', indexlist1)
                print('indexlist2 from R2 amp find_peaks is', indexlist2)

        if verbose:
            print('indexlist:',indexlist)
        resfreqs_from_amp = morefrequencies[indexlist]
    
    if not MONOMER or mode == 'phase':
        ## find where angles are resonant angles
        angleswanted = [np.pi/2, -np.pi/2] # the function will wrap angles so don't worry about mod 2 pi.
        R1_flist,indexlistphaseR1 = find_freq_from_angle(morefrequencies, R1_phase_noiseless, angleswanted=angleswanted, returnindex=True)
        if MONOMER: 
            assert mode == 'phase'
            resfreqs_from_phase = R1_flist
        else:
            R2_flist,indexlistphaseR2 = find_freq_from_angle(morefrequencies, R2_phase_noiseless, angleswanted=angleswanted, 
                                                             returnindex=True)
            resfreqs_from_phase = np.append(R1_flist, R2_flist)
    else:
        assert MONOMER
        resfreqs_from_phase = [] # don't bother with this for the MONOMER
        indexlistphaseR1 = []
        indexlistphaseR2 = []
        R1_flist = []
    
    if verboseplot:
        if MONOMER: # still need to calculate the curves
            R1_amp_noiseless = curve1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                      0, MONOMER, forceboth=forceboth)
            R1_phase_noiseless = theta1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                        0, MONOMER, forceboth=forceboth)
            R1_phase_noiseless = np.unwrap(R1_phase_noiseless)
            indexlistampR1 = [np.argmin(abs(w  - morefrequencies )) for w in resfreqs_from_amp]
        print('Plotting!')
        fig, (ampax, phaseax) = plt.subplots(2,1,gridspec_kw={'hspace': 0}, sharex = 'all') 
        plt.sca(ampax)
        plt.title(plottitle)
        plt.plot(morefrequencies, R1_amp_noiseless, color='gray')
        if not MONOMER:
            plt.plot(morefrequencies, R2_amp_noiseless, color='lightblue')

        plt.plot(morefrequencies[indexlistampR1],R1_amp_noiseless[indexlistampR1], '.')
        if not MONOMER:
            plt.plot(morefrequencies[indexlistampR2],R2_amp_noiseless[indexlistampR2], '.')
        
        plt.sca(phaseax)
        plt.plot(morefrequencies,R1_phase_noiseless, color='gray' )
        if not MONOMER:
            plt.plot(morefrequencies,R2_phase_noiseless, color = 'lightblue')
        plt.plot(R1_flist, theta1(np.array(R1_flist), k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                  0, MONOMER, forceboth=forceboth), '.')
        if not MONOMER:
            plt.plot(R2_flist, theta2(np.array(R2_flist), k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                      0, forceboth=forceboth), '.')

    if mode == 'maxamp' or mode == 'amp' or mode == 'amplitude':
        freqlist = resfreqs_from_amp
    elif mode == 'phase':
        freqlist = resfreqs_from_phase
    else:
        if mode != 'all':
            print("Set mode to any of 'all', 'maxamp', or 'phase'. Recovering to 'all'.")
        # mode is 'all'
        freqlist = np.sort(np.append(resfreqs_from_amp, resfreqs_from_phase))
    

    if veryunique: # Don't return both close frequencies; just pick the higher amplitude frequency of the two.
        ## I obtained indexlists four ways: indexlistampR1, indexlistampR2, indexlistphaseR1, indexlistphaseR2
        indexlist = indexlist + indexlistphaseR1
        if not MONOMER:
            indexlist = indexlist + indexlistphaseR2
        indexlist = list(np.sort(np.unique(indexlist)))
        if verbose:
            print('indexlist:', indexlist)
        
        narrowerW = calcnarrowerW(vals_set, MONOMER)
        
        """ a and b are indices of morefrequencies """
        def veryclose(a,b):
            ## option 1: veryclose if indices are within 2.
            #return abs(b-a) <= 2 
            
            ## option 2: very close if frequencies are closer than .01 rad/s
            #veryclose = abs(morefrequencies[a]-morefrequencies[b]) <= .1  
            
            ## option 3: very close if freqeuencies are closer than W/20
            veryclose = abs(morefrequencies[a]-morefrequencies[b]) <= narrowerW/20
            
            return veryclose
        
        if len(freqlist) > 1:
            ## if two elements of indexlist are veryclose to each other, want to remove the smaller amplitude.
            removeindex = [] # create a list of indices to remove
            try:
                tempfreqlist = morefrequencies[indexlist] # indexlist is indicies of morefrequencies.
                    # if the 10th element of indexlist is indexlist[10]=200, then tempfreqlist[10] = morefrequencies[200]
            except:
                print('indexlist:', indexlist)
            A2 = curve2(tempfreqlist, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)
                # and then A2[10] is the amplitude of R2 at the frequency morefrequencies[200]
                # and then the number 10 is the sort of number we will add to a removeindex list
            for i in range(len(indexlist)-1):
                if veryclose(indexlist[i], indexlist[i-1]):
                    if A2[i] < A2[i-1]: # remove the smaller amplitude
                        removeindex.append(i)
                    else:
                        removeindex.append(i-1)
            numtoremove = len(removeindex)
            if verbose and numtoremove > 0:
                print('Removing', numtoremove, 'frequencies')
                
            removeindex = list(np.unique(removeindex))
            indexlist = list(indexlist)
            ## Need to work on removal from the end of the list 
            ## in order to avoid changing index numbers while working with the list
            while removeindex != []:
                i = removeindex.pop(-1) # work backwards through indexes to remove
                el = indexlist.pop(i) # remove it from indexlist
                if numtoremove < 5 and verbose:
                    print('Removed frequency', morefrequencies[el])

            freqlist = morefrequencies[indexlist]
    
    freqlist = np.sort(freqlist)
    
    if unique or veryunique or (numtoreturn is not None): ## Don't return multiple copies of the same number.
        freqlist =  np.unique(np.array(freqlist))
        
    if verbose:
        print('Possible frequencies are:', freqlist)
    
    if numtoreturn is not None:
        if len(freqlist) == numtoreturn:
            if verbose:
                print ('option 4')
            if returnoptions:
                return list(freqlist), 4
            return list(freqlist)
        if len(freqlist) < numtoreturn:
            if verbose:
                print('Warning: I do not have as many resonant frequencies as was requested.')
            freqlist = list(freqlist)
            # instead I should add another frequency corresponding to some desireable phase.
            if verbose:
                print('Returning instead a freq2 at phase -3pi/4.')
            goodphase = -3*np.pi/4
            for i in range(iterations):
                f2, ind2 = find_freq_from_angle(drive = morefrequencies, 
                                         phase = theta1(morefrequencies, 
                                                        k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                                        0, MONOMER, forceboth=forceboth),
                                         angleswanted = [goodphase], returnindex = True)
                ind2 = ind2[0]
                try:
                    spacing = abs(morefrequencies[ind2] - morefrequencies[ind2-1])
                except IndexError:
                    spacing = abs(morefrequencies[ind2+1] - morefrequencies[ind2])
                finermesh = np.linspace(morefrequencies[ind2] - spacing,morefrequencies[ind2] + spacing, num=n)
                morefrequencies = np.append(morefrequencies, finermesh)
            f2 = f2[0]
            freqlist.append(f2)
            if verboseplot:
                plt.sca(phaseax)
                plt.plot(f2, theta1(f2, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                  0, MONOMER, forceboth=forceboth), '.')
                print('Appending: ', f2)
            for i in range(numtoreturn - len(freqlist)):  
                # This is currently unlikely to be true, but I'm future-proofing 
                # for a future when I want to set the number to an integer greater than 2.
                freqlist.append(np.nan)  # increase list to requested length with nan
            if verboseplot:
                plt.sca(ampax)
                plt.plot(freqlist, curve1(freqlist, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                  0, MONOMER, forceboth=forceboth), 'x')
            if verbose:
                print ('option 5')
            if returnoptions:
                return freqlist, 5
            return freqlist
        
        R1_amp_noiseless = curve1(freqlist, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                  0, MONOMER, forceboth=forceboth)
        R2_amp_noiseless = curve2(freqlist, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                  0, forceboth=forceboth)
        
        topR1index = np.argmax(R1_amp_noiseless)
        
        if numtoreturn == 1:
            # just return the one max amp frequency.
            if verbose:
                print('option 6')
            if returnoptions:
                return [freqlist[topR1index]],6
            return [freqlist[topR1index]]
        
        if numtoreturn != 2:
            print('Warning: returning ' + str(numtoreturn) + ' frequencies is not implemented. Returning 2 frequencies.')
        
        # Choose a second frequency to return.
        topR2index = np.argmax(R2_amp_noiseless)
        threshold = .2 # rad/s
        if abs(freqlist[topR1index] - freqlist[topR2index]) > threshold:
            freqlist = list([freqlist[topR1index], freqlist[topR2index]])
            if verboseplot:
                plt.sca(ampax)
                plt.plot(freqlist, curve1(freqlist, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                  0, MONOMER, forceboth=forceboth), 'x')
            if verbose:
                print('option 7')
            if returnoptions:
                return freqlist, 7
            return freqlist
        else:
            R1_amp_noiseless = list(R1_amp_noiseless)
            freqlist = list(freqlist)
            f1 = freqlist.pop(topR1index)
            R1_amp_noiseless.pop(topR1index)
            secondR1index = np.argmax(R1_amp_noiseless)
            f2 = freqlist.pop(secondR1index)
            if abs(f2-f1) > threshold:
                freqlist = list([f1, f2]) # overwrite freqlist
                if verboseplot:
                    plt.sca(ampax)
                    plt.plot(freqlist, curve1(freqlist, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                                      0, MONOMER, forceboth=forceboth), 'x')
                if verbose:
                    print('option 8')
                if returnoptions:
                    return freqlist, 8
                return freqlist
            else: # return whatever element of the freqlist is furthest
                freqlist.append(f2)                
                # is f1 closer to top or bottom of freqlist?
                if abs(f1 - min(freqlist)) > abs(f1 - max(freqlist)):
                    if verbose:
                        print('option 9')
                    if returnoptions:
                        return [f1, min(freqlist)], 9
                    return [f1, min(freqlist)]
                else:
                    if verbose:
                        print('option 10')
                    if returnoptions:
                        return [f1, max(freqlist)], 10
                    return [f1, max(freqlist)]
                
            
    else:
        if verbose:
            print('option 11')
        if returnoptions:
            return list(freqlist),11
        return list(freqlist)
    
    
# create list of all measured frequencies, centered around res (the resonance frequency), and spaced out by freqdiff
def allmeasfreq_one_res(res, max_num_p, freqdiff):
    newfreqplus = res
    newfreqminus = res
    freqlist = [res]
    while len(freqlist) < max_num_p:
        newfreqplus = newfreqplus + freqdiff
        newfreqminus = newfreqminus - freqdiff
        freqlist.append(newfreqplus)
        freqlist.append(newfreqminus)
    if min(freqlist) < 0:
        print('Value less than zero!')
        print('min(freqlist):', min(freqlist))
    return freqlist

# create list of all measured frequencies, centered around res1 and res2, respectively, and spaced out by freqdiff
def allmeasfreq_two_res(res1, res2, max_num_p, freqdiff):
    newfreq1plus = res1
    newfreq1minus = res1
    newfreq2plus = res2
    newfreq2minus = res2
    freqlist = [res1, res2]
    while len(freqlist) < max_num_p:
        newfreq1plus = newfreq1plus + freqdiff
        newfreq1minus = newfreq1minus - freqdiff
        newfreq2plus = newfreq2plus + freqdiff
        newfreq2minus = newfreq2minus - freqdiff
        freqlist.append(newfreq1plus)  ## this order might matter
        freqlist.append(newfreq2plus)
        freqlist.append(newfreq1minus)
        freqlist.append(newfreq2minus)
    if min(freqlist) < 0:
        print('Value less than zero!')
        print('min(freqlist):', min(freqlist))
    return freqlist
    


def best_choice_freq_set(vals_set, MONOMER, forceboth, reslist, num_p = 10):
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
    narrowerW = calcnarrowerW(vals_set, MONOMER)
    freqdiff = round(narrowerW/6,4)
    if MONOMER:
        measurementfreqs = allmeasfreq_one_res(reslist[0], num_p, freqdiff)
    else:
        measurementfreqs = allmeasfreq_two_res(reslist[0], reslist[1], num_p, freqdiff)
        
    return measurementfreqs[:num_p]
