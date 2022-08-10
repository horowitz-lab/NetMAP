# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 22:44:41 2022

@author: vhorowit
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from resonatorsimulator import \
    curve1, theta1, curve2, theta2, realamp1, imamp1, realamp2, imamp2
from resonatorphysics import complexamp, res_freq_weak_coupling
from helperfunctions import read_params
from resonatorfrequencypicker import makemorefrequencies

global co1
global co2
global co3
global datacolor

co1 = 'C0'
co2 = 'C1'
co3 = 'C2'
datacolor = 'C4'

#Plots of singular value decomposition

alpha_circles = .8 ## set transparency for the black circles around the measurement values
alpha_model = .8   ## set transparency for the dashed black line
alpha_data = .8

""" Plot amplitude or phase versus frequency with set values, simulated data, and SVD results """
def spectrum_plot(drive, noisydata,morefrequencies, noiseless, curvefunction,
                  K1, K2, K12, B1, B2, FD, M1, M2,
                  MONOMER, forceboth,
                  dfcolumn,
                  ylabel,
                  title, labelfreqs, 
                  measurementdf, ax, unitsofpi = False, labelcounts = False,):
    if unitsofpi:
        divisor = np.pi
    else:
        divisor = 1
    ax.plot(morefrequencies, noiseless/divisor, 
            '-', color = 'gray', label='set values') # intended curves
    ax.plot(drive, noisydata/divisor, 
            '.', color = datacolor, alpha = alpha_data, label='simulated data') # simulated data
    ax.plot(morefrequencies, 
            curvefunction(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 
                          0, MONOMER=MONOMER, forceboth=forceboth)/divisor, 
            '--', color='black', alpha = alpha_model, label='SVD results') # predicted spectrum from SVD)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    #For loop to plot R1 amplitude values from table
    for i in range(measurementdf.shape[0]):
        ax.plot(measurementdf.drive[i], (measurementdf[dfcolumn])[i]/divisor, 
                'ko', fillstyle='none', markeredgewidth = 3, alpha = alpha_circles)
        if labelcounts:   # number the measurements in the order they were added (just for vary_num_p)        
            plt.annotate(text=str(i+1), 
                         xy=(measurementdf.drive[i],(measurementdf[dfcolumn])[i]/divisor) )
    ax.set_xlabel('Freq (rad/s)')
    
    plt.legend()
    
    if MONOMER or labelfreqs is None or labelfreqs == []:
        return;
    labelfreqs = list(labelfreqs)
    
    tooclose = 1
    try:
        if abs(labelfreqs[1] - labelfreqs[0]) < tooclose:
            return; # Don't put two labels that are too close
    except:
        pass
    
    if abs(min(labelfreqs) - min(drive)) < tooclose:
        labelfreqs.append(min(drive))
    if abs(max(labelfreqs) - max(drive)) < tooclose:
        labelfreqs.append(max(drive))

    plt.sca(ax)
    plt.xticks(labelfreqs)
    
    
""" label_markers are frequencies to label 
    parameter = drive is normal """
def plotcomplex(complexZ, parameter, title = 'Complex Amplitude', cbar_label='Frequency (rad/s)', 
                label_markers=[],  ax=plt.gca(), s=50, cmap = 'rainbow'):
    assert len(complexZ) == len(parameter)
    plt.sca(ax)
    sc = ax.scatter(np.real(complexZ), np.imag(complexZ), s=s, c = parameter, cmap = cmap, label = 'simulated data' ) # s is marker size
    cbar = plt.colorbar(sc)
    cbar.outline.set_visible(False)
    cbar.set_label(cbar_label)
    ax.set_xlabel('$\mathrm{Re}(Z)$ (m)')
    ax.set_ylabel('$\mathrm{Im}(Z)$ (m)')
    ax.axis('equal');
    plt.title(title)
    plt.gcf().canvas.draw() # draw so I can get xlim and ylim.
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    plt.vlines(0, ymin=ymin, ymax = ymax, colors = 'k', linestyle='solid', alpha = .5)
    plt.hlines(0, xmin=xmin, xmax = xmax, colors = 'k', linestyle='solid', alpha = .5)
    #ax.plot([0,1],[0,0], lw=10,transform=ax.xaxis.get_transform() )#,transform=ax.xaxis.get_transform() ) #transform=ax.transAxes
    
    # label markers that are closest to the desired frequencies
    for label in label_markers:
        if label is None:
            continue
        absolute_val_array = np.abs(parameter - label)
        label_index = absolute_val_array.argmin()
        closest_parameter = parameter[label_index]
        plt.annotate(text=str(round(closest_parameter,2)), 
                     xy=(np.real(complexZ[label_index]), np.imag(complexZ[label_index])) )
        
        ## def plotcomplex draws an empty graph because I call canvas.draw()

"""
measurementdf has a row for each measured frequency
columns: drive, R1Amp, R1Phase, R2Amp, R2Phase, R1AmpCom, R2AmpCom
"""
def plot_SVD_results(drive,R1_amp,R1_phase,R2_amp,R2_phase, measurementdf,  K1, K2, K12, B1, B2, FD, M1, M2, 
                     vals_set,  MONOMER, forceboth,labelfreqs = None,labelcounts = False, datacolor=datacolor):
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
        
    Z1 = complexamp(R1_amp, R1_phase)
    Z2 = complexamp(R2_amp, R2_phase)
    
    morefrequencies = makemorefrequencies(minfreq=min(drive), maxfreq = max(drive), forceboth=forceboth,
                                          includefreqs = drive, 
                                          vals_set = vals_set, MONOMER=MONOMER, n=10*len(drive))
    
    R1_amp_noiseless = curve1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER, forceboth=forceboth)
    R1_phase_noiseless = theta1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER, forceboth=forceboth)
    if not MONOMER:
        R2_amp_noiseless = curve2(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)
        R2_phase_noiseless = theta2(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)

    """
    R1_real_amp_noiseless = realamp1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER, forceboth=forceboth)
    R1_im_amp_noiseless = imamp1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER, forceboth=forceboth)
    if not MONOMER:
        R2_real_amp_noiseless = realamp2(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)
        R2_im_amp_noiseless = imamp2(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)
    """
    
    if labelfreqs is None:
        if len(measurementdf) <= 4:
            labelfreqs = list(measurementdf.drive)
        elif not MONOMER and k12_set < 10:
            # This is an approximation for weak coupling, 
            # not so good for strong coupling, but still a nice pair of frequencies to identify.
            res1 = res_freq_weak_coupling(k1_set, m1_set, b1_set)
            res2 = res_freq_weak_coupling(k2_set, m2_set, b2_set)
            labelfreqs = [res1, res2]
        else:
            labelfreqs = list(np.sort(measurementdf.drive))[::2]
    try:
        labelfreqs = [f.real for f in labelfreqs if f.imag == 0] # force labelfreqs to be real
    except:
        pass
    
    #fig, ((ax1, ax3),(ax2,ax4),(ax5, ax6)) = plt.subplots(3,2, figsize = (10,10))
    if MONOMER:
        fig, ((ax1),(ax2)) = plt.subplots(2,1, figsize = (9.5/2,5), gridspec_kw={'hspace': 0}, sharex = 'all' )
    else:
        fig, ((ax1, ax3),(ax2,ax4)) = plt.subplots(2,2, figsize = (9.5,5), gridspec_kw={'hspace': 0}, sharex = 'all' )

    spectrum_plot(drive=drive, noisydata=R1_amp,
                morefrequencies=morefrequencies, noiseless=R1_amp_noiseless, 
                curvefunction = curve1,
                K1=K1, K2=K2, K12=K12, B1=B1, B2=B2, FD=FD, M1=M1, M2=M2,
                MONOMER=MONOMER, forceboth=forceboth,
                dfcolumn = 'R1Amp',
                ylabel = 'Amplitude $A$ (m)\n',
                title = 'Simulated R1 Spectrum', labelfreqs=labelfreqs, labelcounts = labelcounts,
                measurementdf = measurementdf,
                ax = ax1) 
        
    spectrum_plot(drive=drive, noisydata=R1_phase,
                morefrequencies=morefrequencies, noiseless=R1_phase_noiseless, 
                curvefunction = theta1,
                K1=K1, K2=K2, K12=K12, B1=B1, B2=B2, FD=FD, M1=M1, M2=M2,
                  MONOMER=MONOMER, forceboth=forceboth,
                dfcolumn = 'R1Phase',
                ylabel = 'Phase $\delta$ ($\pi$)', unitsofpi = True,
                title = None, labelfreqs=labelfreqs,labelcounts = labelcounts,
                  measurementdf = measurementdf,
                ax = ax2) 

    if not MONOMER:
        spectrum_plot(drive=drive, noisydata=R2_amp,
                morefrequencies=morefrequencies, noiseless=R2_amp_noiseless, 
                curvefunction = curve2,
                K1=K1, K2=K2, K12=K12, B1=B1, B2=B2, FD=FD, M1=M1, M2=M2,
                      MONOMER=MONOMER, forceboth=forceboth,
                dfcolumn = 'R2Amp',
                ylabel = 'Amplitude $A$ (m)\n', unitsofpi = False,
                title = 'Simulated R2 Spectrum', labelfreqs=labelfreqs,
                      measurementdf = measurementdf,labelcounts = labelcounts,
                ax = ax3) 
        
        spectrum_plot(drive=drive, noisydata=R2_phase,
                morefrequencies=morefrequencies, noiseless=R2_phase_noiseless, 
                curvefunction = theta2,
                K1=K1, K2=K2, K12=K12, B1=B1, B2=B2, FD=FD, M1=M1, M2=M2,
                      MONOMER=MONOMER, forceboth=forceboth,
                dfcolumn = 'R2Phase',
                ylabel = 'Phase $\delta$ ($\pi$)', unitsofpi = True,
                title = None, labelfreqs=labelfreqs,labelcounts = labelcounts,
                      measurementdf = measurementdf,
                ax = ax4) 

    plt.tight_layout()    

    if MONOMER:
        fig2, ax5 = plt.subplots(1,1, figsize = (10/2,4))
    else:
        fig2, ((ax5, ax6)) = plt.subplots(1,2, figsize = (10,4))
    
    # svd curves
    #morefrequencies = np.linspace(minfreq, maxfreq, num = n*10)
    ax5.plot(realamp1(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 0, MONOMER, forceboth=forceboth), 
             imamp1(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 0, MONOMER, forceboth=forceboth), 
             '--', color='black', alpha = alpha_model)
    if not MONOMER:
        ax6.plot(realamp2(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 0,forceboth=forceboth,), 
                 imamp2(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 0,forceboth=forceboth,), 
                 '--', color='black', alpha = alpha_model)

    plotcomplex(Z1, drive, 'Complex Amplitude $Z_1$', ax=ax5, label_markers=labelfreqs)
    ax5.scatter(np.real(measurementdf.R1AmpCom), np.imag(measurementdf.R1AmpCom), 
                s=150, facecolors='none', edgecolors='k', label="points for analysis")
    if labelcounts:
        for i in range(len(measurementdf)):
            plt.annotate(text=str(i+1), 
                         xy=(np.real(measurementdf.R1AmpCom), 
                             np.imag(measurementdf.R1AmpCom)) )
    if not MONOMER:
        plotcomplex(Z2, drive, 'Complex Amplitude $Z_2$', ax=ax6, label_markers=labelfreqs)
        ax6.scatter(np.real(measurementdf.R2AmpCom), np.imag(measurementdf.R2AmpCom), 
                    s=150, facecolors='none', edgecolors='k', label="points for analysis") 
        if labelcounts:
            for i in range(len(measurementdf)):
                plt.annotate(text=str(i+1), 
                             xy=(np.real(measurementdf.R2AmpCom), 
                                 np.imag(measurementdf.R2AmpCom)) )
        
    # true curves
    #morefrequencies = np.linspace(minfreq, maxfreq, num = n*10)
    ax5.plot(realamp1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                      0, MONOMER=MONOMER, forceboth=forceboth,), 
             imamp1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                    0, MONOMER=MONOMER, forceboth=forceboth,), 
             color='gray', alpha = .5)
    if not MONOMER:
        ax6.plot(realamp2(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                          0,MONOMER=MONOMER, forceboth=forceboth,), 
                 imamp2(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 
                        0,MONOMER=MONOMER, forceboth=forceboth,), 
                 color='gray', alpha = .5)

    plt.tight_layout()
    
    if MONOMER:
        axs = [ax1, ax2, ax5]
    else:
        axs = [ax1, ax2, ax3, ax4, ax5, ax6]
        
    for ax in axs:
        plt.sca(ax)
        plt.legend()
    return axs

  
""" convert from a format used for resultsdf (in the frequency sweep below) 
    to the format used in the above function plot_SVD_results() """
def convert_to_measurementdf(resultsdfitem):
    # columns of measurementdf: drive, R1Amp, R1Phase, R2Amp, R2Phase, R1AmpCom, R2AmpCom
    try:
        mdf = pd.DataFrame([[resultsdfitem.Freq1, resultsdfitem.R1Amp_a, resultsdfitem.R1Phase_a, resultsdfitem.R2Amp_a, 
             resultsdfitem.R2Phase_a, resultsdfitem.R1AmpCom_a, resultsdfitem.R2AmpCom_a],
             [resultsdfitem.Freq2, resultsdfitem.R1Amp_b, resultsdfitem.R1Phase_b, resultsdfitem.R2Amp_b, 
             resultsdfitem.R2Phase_b, resultsdfitem.R1AmpCom_b, resultsdfitem.R2AmpCom_b]],
                columns = ['drive', 'R1Amp', 'R1Phase', 'R2Amp', 'R2Phase', 'R1AmpCom', 'R2AmpCom'])
    except AttributeError:
        mdf = pd.DataFrame([[resultsdfitem.Freq1, resultsdfitem.R1Amp_meas1, resultsdfitem.R1Phase_meas1, resultsdfitem.R2Amp_meas1, 
             resultsdfitem.R2Phase_meas1, resultsdfitem.R1AmpCom_meas1, resultsdfitem.R2AmpCom_meas1],
             [resultsdfitem.Freq2, resultsdfitem.R1Amp_meas2, resultsdfitem.R1Phase_meas2, resultsdfitem.R2Amp_meas2, 
             resultsdfitem.R2Phase_meas2, resultsdfitem.R1AmpCom_meas2, resultsdfitem.R2AmpCom_meas2]],
                columns = ['drive', 'R1Amp', 'R1Phase', 'R2Amp', 'R2Phase', 'R1AmpCom', 'R2AmpCom'])
    return mdf

#*** still figuring out these names
def convert_to_measurementdf2(resultsdfitem):
    # columns of measurementdf: drive, R1Amp, R1Phase, R2Amp, R2Phase, R1AmpCom, R2AmpCom

    mdf = pd.DataFrame([[resultsdfitem.Freq1, resultsdfitem.R1_amp_meas1, resultsdfitem.R1Phase_meas1, resultsdfitem.R2_amp_meas1, 
             resultsdfitem.R2Phase_meas1, resultsdfitem.R1AmpCom_meas1, resultsdfitem.R2AmpCom_meas1],
             [resultsdfitem.Freq2, resultsdfitem.R1Amp_meas2, resultsdfitem.R1Phase_meas2, resultsdfitem.R2Amp_meas2, 
             resultsdfitem.R2Phase_meas2, resultsdfitem.R1AmpCom_meas2, resultsdfitem.R2AmpCom_meas2]],
                columns = ['drive', 'R1Amp', 'R1Phase', 'R2Amp', 'R2Phase', 'R1AmpCom', 'R2AmpCom'])
    return mdf