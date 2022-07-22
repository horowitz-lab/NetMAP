import numpy as np
import resonatorphysics
from helperfunctions import read_params
from resonatorsimulator import *

n = 100
minfreq = .1
maxfreq = 5

""" drive and phase are two lists of the same length 
    This will only return one frequency for each requested angle, even if there are additional solutions.
"""
def find_freq_from_angle(drive, phase, angleswanted = [-np.pi/4], returnindex = False):
    
    #specialanglefreq = [drive[np.argmin(abs(phase%(2*np.pi)  - anglewanted%(2*np.pi)))] \
    #                        for anglewanted in angleswanted ]   
    
    threshold = np.pi/30 # small angle threshold
    specialanglefreq = [] # initialize list
    indexlist = []
    for anglewanted in angleswanted:
        index = np.argmin(abs(phase%(2*np.pi)  - anglewanted%(2*np.pi)))
        if index == 0 or index >= len(drive)-1: # edges of dataset require additional scrutiny
            ## check to see if it's actually close after all
            nearness = abs(phase[index]%(2*np.pi)-anglewanted%(2*np.pi))
            if nearness > threshold:
                continue # don't include this index
        specialanglefreq.append(drive[index])
        indexlist.append(index)
        
    if returnindex:
        return specialanglefreq, indexlist
    else:
        return specialanglefreq  
		

def freqs_chosen_by_phase(vals_set, MONOMER, forceboth, res1 = None,res2 = None, n=n, morefrequencies = np.linspace(minfreq,maxfreq,30*n),
                         verbose = False):
    
    if verbose:
        print('Running freqs_chosen_by_phase()')
    
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
    
    if res1 is None:
        res1 = resonatorphysics.res_freq_weak_coupling(k1_set, m1_set, b1_set)
    if res2 is None and not MONOMER:
        res2 = resonatorphysics.res_freq_weak_coupling(k2_set, m2_set, b2_set)
    
    phaseR1 = theta1(morefrequencies, k1_set, k2_set, k12_set, 
                                     b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER, forceboth=forceboth)
    
    if MONOMER:
        # choose length of anglelist
        m = n-3-int(.2*n) # 3 are special angles; 20% are evenly spaced freqs
    else:
        m = int((n-3-int(.2*n))/2)
    
    anglelist = np.linspace(min(phaseR1), max(phaseR1), m) 
    #anglelist = np.append(anglelist, -np.pi/3)
    #anglelist = np.append(anglelist, -2*np.pi/3)
    anglelist = np.append(anglelist, -np.pi/4)
    anglelist = np.append(anglelist, -3*np.pi/4)
    anglelist = np.unique(np.sort(np.append(anglelist, -np.pi/2)))
    
    freqlist = find_freq_from_angle(morefrequencies, 
                      phase = phaseR1,
                      angleswanted = anglelist)
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
        if verbose:
            print('anglelist/pi: ', anglelist/np.pi)
            print('freqlist2: ', freqlist2)
        freqlist.extend(freqlist2)
        freqlist.append(res2)
        morefrequencies = np.append(morefrequencies,res2)
        
    freqlist.append(res1)
    try:
        reslist = res_freq_numeric(vals_set=vals_set, MONOMER=MONOMER, mode = 'all', 
                                   minfreq=minfreq, maxfreq=maxfreq, morefrequencies=morefrequencies,
                                   unique = True, veryunique = True, verboseplot = False, verbose=verbose, iterations = 3)
        freqlist.extend(reslist)
    except NameError:
        pass
    
    freqlist = list(np.sort(np.unique(freqlist)))
    
    while freqlist[0] < 0:
        freqlist.pop(0) # drop negative frequencies
        
    numwanted = n-len(freqlist) # how many more frequencies are wanted?
    evenlyspacedfreqlist = np.linspace(min(morefrequencies), max(morefrequencies), 
                                       num = numwanted + 2) #  I added 2 for the endpoints
    freqlist.extend(evenlyspacedfreqlist)
    #print(freqlist)
    chosendrive = list(np.sort(np.unique(np.array(freqlist))))
    
    while chosendrive[0] < 0:
        f = chosendrive.pop(0)
        print('Warning: Unexpected negative frequency', f)
    chosendrive = np.array(chosendrive)
    
    #morefrequencies.extend(chosendrive)
    morefrequencies = np.concatenate((morefrequencies, chosendrive))
    morefrequencies = list(np.sort(np.unique(morefrequencies)))
    
    while morefrequencies[0] < 0:
        f = morefrequencies.pop(0)
        print('Warning: Unexpected negative frequency', f)
    
    return chosendrive, np.array(morefrequencies)
	
    
""" n is the number of frequencies is the drive; we'll have more for more frequencies. 
 Can you improve this by calling freqs_chosen_by_phase afterward? """
def makemorefrequencies( minfreq, maxfreq,vals_set, MONOMER,res1 = None, res2 = None, includefreqs = None,  n=n):
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
    
    if res1 is None:
        res1 = resonatorphysics.res_freq_weak_coupling(k1_set, m1_set, b1_set)
    if res2 is None:
        res2 = resonatorphysics.res_freq_weak_coupling(k2_set, m2_set, b2_set)
	
    morefrequencies = np.linspace(minfreq, maxfreq, num = n*40)
    morefrequencies = np.append(morefrequencies, [res1, res2])
    
    if includefreqs is not None:
        morefrequencies = np.append(morefrequencies, np.array(includefreqs))
    
    W1 = resonatorphysics.approx_width(k = k1_set, m = m1_set, b=b1_set)
    morefrequencies = np.append(morefrequencies, np.linspace(res1-W1, res1+W1, num = 7*n))
    morefrequencies = np.append(morefrequencies, np.linspace(res1-2*W1, res1+2*W1, num = 10*n)) 
    if not MONOMER:
        W2 = resonatorphysics.approx_width(k = k2_set, m = m2_set, b=b2_set)
        morefrequencies = np.append(morefrequencies, np.linspace(res2-W2, res2+W2, num = 7*n))
        morefrequencies = np.append(morefrequencies, np.linspace(res2-2*W2, res2+2*W2, num = 10*n))
    morefrequencies = list(np.sort(np.unique(morefrequencies)))
    
    while morefrequencies[0] < 0:
        morefrequencies.pop(0)
        
    return np.array(morefrequencies)
	
def find_special_freq(drive, amp, phase, anglewanted = np.radians(225)):
    maxampfreq = drive[np.argmax(amp)]
    specialanglefreq = drive[np.argmin(abs(phase%(2*np.pi) - anglewanted%(2*np.pi)))]
    return maxampfreq, specialanglefreq
	
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
	
	
### res_freq_numeric()
## Uses privilege
## Not guaranteed to find all resonance peaks but should work ok for dimer
## Returns list of peak frequencies. 
## If numtoreturn is None, then any number of frequencies could be returned.
## You can also set numtoreturn to 1 or 2 to return that number of frequencies.
def res_freq_numeric(vals_set, MONOMER, mode = 'all', minfreq=minfreq, maxfreq=maxfreq, morefrequencies=None,
                     unique = True, veryunique = True, numtoreturn = None, verboseplot = False, verbose=False, iterations = 1, n=n):
    m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set = read_params(vals_set, MONOMER)
    
    if MONOMER: # just compute it directly for Monomer
        if numtoreturn is not None and numtoreturn != 1:
            print('Cannot return ' + str(numtoreturn) + ' res freqs for Monomer.')
        return [res_freq_weak_coupling(k1_set, m1_set, b1_set)]

    first = True
    for i in range(iterations):
        if morefrequencies is None:
            morefrequencies = makemorefrequencies(minfreq, maxfreq,includefreqs = None, vals_set = vals_set, MONOMER=MONOMER, n=n)
        elif first is False:
            indexlist1 = np.append(indexlist1,index1)
            if not MONOMER:
                indexlist2 = np.append(indexlist2,index2)
                indexlist = np.append(indexlist1, indexlist2)
            indexlist = np.unique(indexlist)
            if verbose:
                print('Repeating with finer frequency mesh around frequencies:', morefrequencies[np.sort(indexlist)])

            morefrequenciesprev = morefrequencies.copy()
            for index in indexlist:
                finerlist = np.linspace(morefrequenciesprev[index-1], morefrequenciesprev[index+1], num = n)
                morefrequencies = np.append(morefrequencies,finerlist)
            morefrequencies = np.sort(np.unique(morefrequencies))
            
        R1_amp_noiseless = curve1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER, forceboth=forceboth)
        R1_phase_noiseless = theta1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER, forceboth=forceboth)
        R2_amp_noiseless = curve2(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)
        R2_phase_noiseless = theta2(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)
        R1_phase_noiseless = np.unwrap(R1_phase_noiseless)
        R2_phase_noiseless = np.unwrap(R2_phase_noiseless)

        ## find maxima
        index1 = np.argmax(R1_amp_noiseless)
        indexlist1, heights = find_peaks(R1_amp_noiseless, height=.015)
        if MONOMER:
            index2 = np.nan
            indexlist2 = []
        else:
            index2 = np.argmax(R2_amp_noiseless)
            indexlist2, heights2 = find_peaks(R2_amp_noiseless, height=.015)
        
        
        if verbose:
            print('Maximum amplitude for R1 is ',  R1_amp_noiseless[index1], 'at', morefrequencies[index1])
            print('Maximum amplitude for R2 is ',  R2_amp_noiseless[index2], 'at', morefrequencies[index2])
            
        first = False
    

    
    ## Check to see if findpeaks just worked
    if (numtoreturn == 2) and (mode != 'phase'):
        if len(indexlist2) == 2:
            if verbose:
                print("Used findpeaks on R2 amplitude")
            return list(np.sort(morefrequencies[indexlist2]))
        if len(indexlist1) == 2:
            if verbose:
                print("Used findpeaks on R1 amplitude")
            return list(np.sort(morefrequencies[indexlist1]))
    
    indexlistampR1 = np.append(index1, indexlist1)
    if not MONOMER:
        indexlistampR2 = np.append(index2, indexlist2)
    indexlist = np.append(indexlistampR1, indexlistampR2)
    resfreqs_from_amp = morefrequencies[indexlist]
    
    ## find where angles are resonant angles
    angleswanted = [np.pi/2, -np.pi/2] # the function will wrap angles so don't worry about mod 2 pi.
    R1_flist,indexlistphaseR1 = find_freq_from_angle(morefrequencies, R1_phase_noiseless, angleswanted=angleswanted, returnindex=True)
    R2_flist,indexlistphaseR2 = find_freq_from_angle(morefrequencies, R2_phase_noiseless, angleswanted=angleswanted, returnindex=True)
    
    resfreqs_from_phase = np.append(R1_flist, R2_flist)
    
    if verboseplot:
        plt.figure()
        plt.plot(morefrequencies, R1_amp_noiseless, color='gray')
        plt.plot(morefrequencies, R2_amp_noiseless, color='lightblue')
        plt.plot(morefrequencies[indexlistampR1],R1_amp_noiseless[indexlistampR1], '.')
        plt.plot(morefrequencies[indexlistampR2],R2_amp_noiseless[indexlistampR2], '.')
        
        plt.figure()
        plt.plot(morefrequencies,R1_phase_noiseless, color='gray' )
        plt.plot(morefrequencies,R2_phase_noiseless, color = 'lightblue')
        plt.plot(R1_flist, theta1(np.array(R1_flist), k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER), '.')
        plt.plot(R2_flist, theta2(np.array(R2_flist), k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0), '.')

    if mode == 'maxamp' or mode == 'amp' or mode == 'amplitude':
        freqlist = resfreqs_from_amp
    elif mode == 'phase':
        freqlist = resfreqs_from_phase
    else:
        if mode != 'all':
            print("Set mode to any of 'all', 'maxamp', or 'phase'. Recovering to 'all'.")
        # mode is 'all'
        freqlist = np.sort(np.append(resfreqs_from_amp, resfreqs_from_phase))
        
    if veryunique: # Don't return both close frequencies; just pick the higher frequency of the two.
        ## I obtained indexlists four ways: indexlistampR1, indexlistampR2, indexlistphaseR1, indexlistphaseR2
        indexlist = np.append(indexlist,indexlistphaseR1)
        indexlist = np.append(indexlist, indexlistphaseR2)
        indexlist = np.sort(np.unique(indexlist))
        
        W1=resonatorphysics.approx_width(k1_set, m1_set, b1_set)
        W2=resonatorphysics.approx_width(k2_set, m2_set, b2_set)
        narrowerW = min(W1,W2)
        
        """ a and b are indices of morefrequencies """
        def veryclose(a,b):
            if verbose:
                print('checking if veryclose', morefrequencies[a], morefrequencies[b])
            ## option 1: veryclose if indices are within 2.
            #return abs(b-a) <= 2 
            
            ## option 2: very close if frequencies are closer than .01 rad/s
            #veryclose = abs(morefrequencies[a]-morefrequencies[b]) <= .1  
            
            ## option 3: very close if freqeuencies are closer than W/20
            veryclose = abs(morefrequencies[a]-morefrequencies[b]) <= narrowerW/20
            
            if verbose:
                print('Very close: ', veryclose)
            return veryclose
        
        ## if two elements of indexlist are veryclose to each other, want to remove the smaller amplitude.
        removeindex = [] # create a list of indices to remove
        tempfreqlist = morefrequencies[indexlist] # indexlist is indicies of morefrequencies.
            # if the 10th element of indexlist is indexlist[10]=200, then tempfreqlist[10] = morefrequencies[200]
        A2 = curve2(tempfreqlist, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0)
            # and then A2[10] is the amplitude of R2 at the frequency morefrequencies[200]
            # and then the number 10 is the sort of number we will add to a removeindex list
        for i in range(len(indexlist)-1):
            if veryclose(indexlist[i], indexlist[i-1]):
                if A2[i] < A2[i-1]: # remove the smaller amplitude
                    if verbose:
                        print('Removing frequency ', morefrequencies[indexlist[i]], 
                              'and leaving frequency', morefrequencies[indexlist[i-1]])
                    removeindex.append(i)
                else:
                    if verbose:
                        print('Removing frequency ', morefrequencies[indexlist[i-1]], 
                              'and leaving frequency', morefrequencies[indexlist[i]])
                    removeindex.append(i-1)
        removeindex = list(np.unique(removeindex))
        indexlist = list(indexlist)
        ## Need to work on removal from the end of the list 
        ## in order to avoid changing index numbers while working with the list
        while removeindex != []:
            i = removeindex.pop(-1) # work backwards through indexes to remove
            el = indexlist.pop(i) # remove it from indexlist
            if verbose:
                print('Removed frequency ', morefrequencies[el])
        
        freqlist = morefrequencies[indexlist]
    
    freqlist = np.sort(freqlist)
    
    if unique or veryunique or (numtoreturn is not None): ## Don't return multiple copies of the same number.
        freqlist =  np.unique(np.array(freqlist))
    
    if numtoreturn is not None:
        if len(freqlist) == numtoreturn:
            return list(freqlist)
        if len(freqlist) < numtoreturn:
            if verbose:
                print('Warning: I do not have as many resonant frequencies as was requested.')
            freqlist = list(freqlist)
            #### instead I should add another frequency corresponding to some desireable phase.
            if verbose:
                print('Returning instead a freq2 at phase -pi/4.')
            goodphase = -np.pi/4
            f2 = find_freq_from_angle(drive = morefrequencies, 
                                     phase = R1_phase_noiseless,
                                     angleswanted = [goodphase], returnindex = False)
            f2 = f2[0]
            freqlist.append(f2)
            if verbose:
                print('Appending: ', f2)
            for i in range(numtoreturn - len(freqlist)):  
                # This is currently unlikely to be true, but I'm future-proofing 
                # for a future when I want to set the number to an integer greater than 2.
                freqlist.append(np.nan)  # increase list to requested length with nan
            return freqlist
        
        R1_amp_noiseless = curve1(freqlist, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER, forceboth=forceboth)
        R2_amp_noiseless = curve2(freqlist, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)
        
        topR1index = np.argmax(R1_amp_noiseless)
        
        if numtoreturn == 1:
            # just return the one max amp frequency.
            return [freqlist[topR1index]]
        
        if numtoreturn != 2:
            print('Warning: returning ' + str(numtoreturn) + ' frequencies is not implemented. Returning 2 frequencies.')
        
        # Choose a second frequency to return.
        topR2index = np.argmax(R2_amp_noiseless)
        if topR1index != topR2index:
            return list([freqlist[topR1index], freqlist[topR2index]])
        else:
            R1_amp_noiseless = list(R1_amp_noiseless)
            freqlist = list(freqlist)
            f1 = freqlist.pop(topR1index)
            R1_amp_noiseless.pop(topR1index)
            secondR1index = np.argmax(R1_amp_noiseless)
            f2 = freqlist.pop(secondR1index)
            return list([f1, f2])
            
    else:
        return list(freqlist)