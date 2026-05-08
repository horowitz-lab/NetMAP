# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:08:31 2022

@author: vhorowit
"""

import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
try:
    import winsound
except:
    pass

def datestring():
    return datetime.datetime.today().strftime('%Y-%m-%d %H;%M;%S')

## source: https://stackabuse.com/python-how-to-flatten-list-of-lists/
def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def listlength(list1):
    try:
        length = len(list1)
    except TypeError:
        length = 1
    return length

def printtime(repeats, before, after, dobeep = True):
    print('Ran ' + str(repeats) + ' times in ' + str(round(after-before,3)) + ' sec')
    if dobeep:
        beep()
        
""" vh is often complex but its imaginary part is actually zero, so let's store it as a real list of vectors instead """
def make_real_iff_real(vh):
    vhr = [] # real list of vectors
    for vect in vh:
        vhr.append([v.real for v in vect if v.imag == 0]) # make real if and only if real
    return (np.array(vhr))

""" Store parameters extracted from SVD """
def store_params(M1, M2, M3, B1, B2, B3, K1, K2, K3, K4, FD):
    params = [M1, M2, M3, B1, B2, B3, K1, K2, K3, K4, FD]
    return params

def read_params(vect):
    [M1, M2, M3, B1, B2, B3, K1, K2, K3, K4, FD] = vect
    return [M1, M2, M3, B1, B2, B3, K1, K2, K3, K4, FD]

def savefigure(savename):
    try:
        plt.savefig(savename + '.svg', dpi = 600, bbox_inches='tight', transparent=True)
    except:
        print('Could not save svg')
    try:
        plt.savefig(savename + '.pdf', dpi = 600, bbox_inches='tight', transparent=True)
           # transparent true source: https://jonathansoma.com/lede/data-studio/matplotlib/exporting-from-matplotlib-to-open-in-adobe-illustrator/
    except:
        print('Could not save pdf')
    plt.savefig(savename + '.png', dpi = 600, bbox_inches='tight', transparent=True)
    print("Saved:\n", savename + '.png')


def calc_error_interval(resultsdf, resultsdfmean, groupby, fractionofdata = .95):
    for column in ['E_lower_1D', 'E_upper_1D','E_lower_2D', 'E_upper_2D','E_lower_3D', 'E_upper_3D']:
        resultsdfmean[column] = np.nan
    dimensions =  ['1D', '2D', '3D']
    items = resultsdfmean[groupby].unique()
    
    for item in items:
        for D in dimensions:
            avgerr = resultsdf[resultsdf[groupby]== item]['avgsyserr%_' + D]
            avgerr = np.sort(avgerr)
            halfalpha = (1 - fractionofdata)/2
            ## literally select the 95% fraction by tossing out the top 2.5% and the bottom 2.5% 
            ## For 95%, It's ideal if I do 40*N measurements for some integer N.
            lowerbound = np.mean([avgerr[int(np.floor(halfalpha*len(avgerr)))], avgerr[int(np.ceil(halfalpha*len(avgerr)))]])
            upperbound = np.mean([avgerr[-int(np.floor(halfalpha*len(avgerr))+1)],avgerr[-int(np.ceil(halfalpha*len(avgerr))+1)]])
            resultsdfmean.loc[resultsdfmean[groupby]== item,'E_lower_'+ D] = lowerbound
            resultsdfmean.loc[resultsdfmean[groupby]== item,'E_upper_' + D] = upperbound
    return resultsdf, resultsdfmean

def beep():
    try:
        winsound.PlaySound(r'C:\Windows\Media\Speech Disambiguation.wav', flags = winsound.SND_ASYNC)
        return
    except:
        pass
    try:
        winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
        return
    except:
        pass
    try:
        winsound.Beep(450,150)
        return
    except:
        pass