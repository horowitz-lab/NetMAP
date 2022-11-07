# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:08:31 2022

@author: vhorowit
"""

import matplotlib.pyplot as plt
import numpy as np
try:
    import winsound
except:
    pass

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
def store_params(M1, M2, B1, B2, K1, K2, K12, FD, MONOMER):
    if MONOMER:
        params = [M1,  B1,  K1,  FD]
    else:
        params = [M1, M2, B1, B2, K1, K2, K12, FD]
    return params

def read_params(vect, MONOMER):
    if MONOMER:
        [M1, B1, K1, FD] = vect
        K12 = 0
        M2 = 0
        B2 = 0
        K2= 0
    else:
        [M1, M2, B1, B2, K1, K2, K12, FD] = vect
    return [M1, M2, B1, B2, K1, K2, K12, FD]

def savefigure(savename):
    plt.savefig(savename + '.svg', dpi = 600, bbox_inches='tight')
    plt.savefig(savename + '.pdf', dpi = 600, bbox_inches='tight')
    plt.savefig(savename + '.png', dpi = 600, bbox_inches='tight',)
    print("Saved:\n", os.path.join(savefolder,savename) + '.png')


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