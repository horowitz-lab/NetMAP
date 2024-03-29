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
import helperfunctions
from helperfunctions import read_params
from resonatorfrequencypicker import makemorefrequencies
import seaborn as sns
from datetime import datetime
import matplotlib as mpl

global co1
global co2
global co3
global figwidth 

co1 = 'C0'
co2 = 'C1'
co3 = 'C2'
purplecolor = 'C4' 
maxfigwidth = 7.086 # 180 mm
figwidth = maxfigwidth/2



#Plots of singular value decomposition

alpha_circles = .8 ## set transparency for the black circles around the measurement values
alpha_model = .8   ## set transparency for the dashed black line
alpha_data = .8


# Nature says: (https://www.nature.com/npp/authors-and-referees/artwork-figures-tables)
# Wait but this is for neuropsychopharmacology!
#Figure width - single image	86 mm (3.38 in) (should be able to fit into a single column of the printed journal)
#Line width	Between 0.5 and 1 point
# https://www.nature.com/nphys/submission-guidelines/aip-and-formatting>  font size 5 to 7 pt
# Figure panels should be prepared at a minimum resolution of 300 dpi and saved at a maximum width of 180 mm.  

def set_format():
    sns.set_context('paper')
    # default plotting parameters
    
    params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
    plt.rcParams.update(params)
    plt.rcParams['axes.labelsize'] = 7
    plt.rcParams['axes.titlesize'] = 7
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    #plt.rcParams['font.sans-serif'] = "Comic Sans MS"
    
    # Key mathtext.fontset:
    # supported values are ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']
    plt.rcParams['mathtext.fontset'] = 'dejavusans' # I can also copy-paste Greek in Illustrator to get an italic arial font.

    font = {'family' : 'sans-serif',
            'size'   : 7}
    mpl.rc('font', **font)
    plt.rcParams.update({'font.size': 7}) ## Nature Physics wants font size 5 to 7.
    #plt.rcParams.update({
    #    "pdf.use14corefonts": True # source: https://github.com/matplotlib/matplotlib/issues/21893
    #}) # findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    
    #plt.rcParams["length"] = 3
    plt.rcParams['axes.linewidth'] = 0.7
    plt.rcParams['xtick.major.width'] = 0.7
    plt.rcParams['ytick.major.width'] = 0.7
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['ytick.minor.width'] = 0.5
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['ytick.major.size'] = 2
    plt.rcParams['xtick.minor.size'] = 0.5 # VERY short.
    plt.rcParams['ytick.minor.size'] = 0.5
    plt.rcParams['figure.dpi']= 150
    #plt.rcParams['figure.figsize'] = (3.38/2,3.38/2)
     
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.minor.visible'] = True
    plt.minorticks_on()
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.right'] = True
    # source: https://physicalmodelingwithpython.blogspot.com/2015/06/making-plots-for-publication.html
    plt.rcParams['pdf.fonttype'] = 42 # Don't outline text for NPhys
    plt.rcParams['svg.fonttype'] = 'none'
    
    plt.rcParams['axes.titlepad'] = -5 
    
    plt.rcParams['pdf.fonttype']=42 
        # source: Nature https://drive.google.com/drive/folders/15m_c_ZfP2X4C9G7bOtQBdSlcLmJkUA7D
    plt.rcParams['ps.fonttype'] = 42
        # source: https://jonathansoma.com/lede/data-studio/matplotlib/exporting-from-matplotlib-to-open-in-adobe-illustrator/
    
def text_color_legend(**kwargs):
    l = plt.legend(**kwargs)
    # set text color in legend
    for text in l.get_texts():
        if '1D' in str(text):
            text.set_color(co1)
        elif '2D' in str(text):
            text.set_color(co2)
        elif '3D' in str(text):
            text.set_color(co3)
    return l

""" Plot amplitude or phase versus frequency with set values, simulated data, and SVD results.
    Demo: if true, plot without tick marks """
def spectrum_plot(drive, noisydata,morefrequencies, noiseless, curvefunction,
                  K1, K2, K12, B1, B2, FD, M1, M2,
                  MONOMER, forceboth,
                  dfcolumn,
                  ylabel,
                  title, labelfreqs, 
                  measurementdf, ax, unitsofpi = False, labelcounts = False,
                  legend = False, demo = False,
                  show_points = True, # show the spectra data
                  show_output = True, # show the SVD output plot
                  show_set = True, # show the set values
                  show_selected_points = True,
                  include_zero = True,
                  verbose = False,
                  cmap = 'rainbow',s=50, bigcircle = 150, 
                  rainbow_colors = True, datacolor = purplecolor):
    
    set_format()
    plt.sca(ax)
    
    if verbose:
        print('Running spectrum_plot(), show_points is', 
            show_points, ', show_output is', show_output,
            ', show_set is', show_set, ', and show_selected points is',
            show_selected_points)
    
    if unitsofpi:
        divisor = np.pi
    else:
        divisor = 1
        
    if include_zero:
        # I want to include zero amplitude / phase in my plot.
        # Plotting an invisible point there will work.
        plt.plot(measurementdf.drive[0],0, alpha = 0) 
        
    if show_selected_points:
        for i in range(measurementdf.shape[0]):
            plt.axvline(measurementdf.drive[i], color = 'gray', lw = 0.5)
        
    if show_set:
        ax.plot(morefrequencies, noiseless/divisor, 
                '-', color = 'gray', alpha = 0.8, label='set values') # intended curves
    
    if show_points:
        if rainbow_colors:
            ax.scatter(drive, noisydata/divisor, 
                            s=s,c = drive, cmap = cmap, label = 'simulated data' ) # s is marker size
        else:
            ax.plot(drive, noisydata/divisor, 
                    '.', color = datacolor, alpha = alpha_data, label='simulated data') # simulated data
    if show_output:               
        ax.plot(morefrequencies, 
                curvefunction(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 
                              0, MONOMER=MONOMER, forceboth=forceboth)/divisor, 
                '--', color='black', alpha = alpha_model, label='SVD results') # predicted spectrum from SVD)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if show_selected_points:
        #For loop to plot R1 amplitude values from table
        for i in range(measurementdf.shape[0]):
            ax.scatter(measurementdf.drive[i], (measurementdf[dfcolumn])[i]/divisor, 
                    facecolors='none', edgecolors='k', label="points for analysis",
                    s=bigcircle,alpha = alpha_circles)
            if labelcounts:   # number the measurements in the order they were added (just for vary_num_p)        
                plt.annotate(text=str(i+1), 
                             xy=(measurementdf.drive[i],(measurementdf[dfcolumn])[i]/divisor) )
    ax.set_xlabel('$\omega$ (rad/s)')
    
    if legend:
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05), ncol=1,)
    
    if demo:
        plt.sca(ax)
        plt.xticks([])
        plt.yticks([])
        return;
    
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
    set_format()
    assert len(complexZ) == len(parameter)
    plt.sca(ax)
    plt.axvline(0,  color = 'k', linestyle='solid',  linewidth = .5)
    plt.axhline(0, color = 'k', linestyle='solid',  linewidth = .5)
    # colorful circles
    sc = ax.scatter(np.real(complexZ), np.imag(complexZ), s=s, c = parameter,
                    cmap = cmap, label = 'simulated data' ) # s is marker size
    cbar = plt.colorbar(sc)
    cbar.outline.set_visible(False)
    cbar.set_label(cbar_label)
    ax.set_xlabel('$\mathrm{Re}(Z)$ (m)')
    ax.set_ylabel('$\mathrm{Im}(Z)$ (m)')
    ax.axis('equal');
    plt.title(title)
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
                     vals_set,  MONOMER, forceboth,labelfreqs = None,labelcounts = False, datacolor=purplecolor,
                     overlay = False, legend = False, context = None, saving= False, labelname = '', demo=False,
                     resonatorsystem = None, show_set=None,
                     figsizeoverride1 = None, figsizeoverride2 = None):
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
    
    show_set_override = show_set
    set_format()
    if saving:
        datestr = datetime.today().strftime('%Y-%m-%d %H;%M;%S')
        
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
    
    if context:
        sns.set_context(context)
        
    figratio = 5/(9.5)
    if context == 'paper':
        set_format()
        if (MONOMER and not overlay):
            figsize = (figwidth/2, figratio * figwidth )
        elif (MONOMER and overlay):
            figsize = (figwidth*.6, figratio * figwidth*.8 )
        else:
            figsize = (figwidth, figratio * figwidth )
        s = 25 # increased from 3, 2022-12-29
        bigcircle = 30
        amplabel = '$A\;$(m)'
        phaselabel = '$\phi\;(\pi)$'
        titleR1 = ''
        titleR2 = ''
    else:
        if (MONOMER and not overlay):
            figsize = (9.5/2,5)
        else:
            figsize = (9.5,5)
        s=50
        bigcircle = 150
        amplabel = 'Amplitude $A$ (m)\n'
        phaselabel = 'Phase $\phi$ ($\pi$)'
        titleR1= 'Simulated R1 Spectrum'
        titleR2 = 'Simulated R2 Spectrum'
    if demo: # overwrite all these
        figsize = (1,1)
        s = 3
        bigcircle = 30
        amplabel = ''
        phaselabel = ''
        titleR1 = ''
        titleR2 = ''
        plotcount = 2 # plot two steps
    else:
        plotcount = 1 # normally plot everything together
        
    if figsizeoverride1 is not None:
        figsize = figsizeoverride1
    print('figsize1:', figsize)
    
    for i in range(plotcount):
        if plotcount == 1:
            show_points = True # show the spectra data
            show_output = True # show the SVD output plot
            show_set = True # show the set values
            show_selected_points = True
        elif i <=0 and plotcount == 2:
            # first demo plot:
            show_points = True # show the spectra data
            show_output = False # show the SVD output plot
            show_set = False # show the set values
            show_selected_points = True
            print('Not showing SVD output, unless buggy')
        elif i > 0 and plotcount == 2:
            # previous demo plot
            plt.tight_layout() 
            if saving: # save the first round of plots
                filename = datestr + 'demo1spectrum' + labelname
                helperfunctions.savefigure(filename)
            plt.show()
            # next demo plot
            show_points = True # show the spectra data
            show_output = True # show the SVD output plot
            show_set = False # show the set values
            show_selected_points = False

            
        if overlay:
            if MONOMER:
                fig, ax1 = plt.subplots(1,1,figsize = figsize)
            else:
                fig, (ax1, ax3) = plt.subplots(1,2, figsize = figsize)
                ax4 = ax3.twinx()
            ax2 = ax1.twinx()
        
        
        else:
            #fig, ((ax1, ax3),(ax2,ax4),(ax5, ax6)) = plt.subplots(3,2, figsize = (10,10))
            if MONOMER: # in future might want to include the circular plot in this
                fig, ((ax1),(ax2)) = plt.subplots(2,1, 
                    figsize = figsize, gridspec_kw={'hspace': 0}, sharex = 'all' )
            else:
                fig, ((ax1, ax3),(ax2,ax4)) = plt.subplots(2,2,
                    figsize = figsize, gridspec_kw={'hspace': 0}, sharex = 'all' )

        if demo:
            rainbow_colors = False
        else:
            rainbow_colors = True
        datacolor1 = 'C3'
        datacolor2 = 'C4'
        if show_set_override is not None:
            show_set = show_set_override

        spectrum_plot(drive=drive, noisydata=R1_amp,
                    morefrequencies=morefrequencies, noiseless=R1_amp_noiseless, 
                    curvefunction = curve1,
                    K1=K1, K2=K2, K12=K12, B1=B1, B2=B2, FD=FD, M1=M1, M2=M2,
                    MONOMER=MONOMER, forceboth=forceboth,
                    dfcolumn = 'R1Amp',
                    ylabel = amplabel,
                    show_points = show_points, 
                    show_output = show_output, # show the SVD output plot
                    show_set = show_set, # show the set values
                    show_selected_points = show_selected_points,
                    title = titleR1, labelfreqs=labelfreqs, labelcounts = labelcounts,
                    measurementdf = measurementdf,
                    legend = legend, s=s, bigcircle = bigcircle, demo=demo,
                    rainbow_colors = rainbow_colors, datacolor = datacolor1, # only used if demo
                    ax = ax1) 
            
        spectrum_plot(drive=drive, noisydata=R1_phase,
                    morefrequencies=morefrequencies, noiseless=R1_phase_noiseless, 
                    curvefunction = theta1,
                    K1=K1, K2=K2, K12=K12, B1=B1, B2=B2, FD=FD, M1=M1, M2=M2,
                    MONOMER=MONOMER, forceboth=forceboth,
                    dfcolumn = 'R1Phase',
                    ylabel = phaselabel, unitsofpi = True,
                    title = None, labelfreqs=labelfreqs,labelcounts = labelcounts,
                    measurementdf = measurementdf,
                    show_points = show_points, 
                    show_output = show_output, # show the SVD output plot
                    show_set = show_set, # show the set values
                    show_selected_points = show_selected_points,
                    legend = legend,s=s,bigcircle = bigcircle, demo=demo,
                    rainbow_colors = rainbow_colors, datacolor = datacolor2, # only used if demo
                    ax = ax2)
        if demo:
            ax1.set_xlabel("")
            ax2.set_xlabel("")
            ax1.set_ylabel("")
            ax2.set_ylabel("")
            
    if resonatorsystem == 2: # specific instructions for nanuscript figure
        ax1.set_yticks([0,50])
        ax2.set_yticks([0,-1])

    if not MONOMER:
        spectrum_plot(drive=drive, noisydata=R2_amp,
                morefrequencies=morefrequencies, noiseless=R2_amp_noiseless, 
                curvefunction = curve2,
                K1=K1, K2=K2, K12=K12, B1=B1, B2=B2, FD=FD, M1=M1, M2=M2,
                MONOMER=MONOMER, forceboth=forceboth,
                dfcolumn = 'R2Amp',
                ylabel = amplabel, unitsofpi = False,
                title = titleR2, labelfreqs=labelfreqs,
                measurementdf = measurementdf,labelcounts = labelcounts,
                legend = legend,s=s,bigcircle = bigcircle, demo=demo,
                ax = ax3) 
        
        spectrum_plot(drive=drive, noisydata=R2_phase,
                morefrequencies=morefrequencies, noiseless=R2_phase_noiseless, 
                curvefunction = theta2,
                K1=K1, K2=K2, K12=K12, B1=B1, B2=B2, FD=FD, M1=M1, M2=M2,
                      MONOMER=MONOMER, forceboth=forceboth, 
                dfcolumn = 'R2Phase',
                ylabel = phaselabel, unitsofpi = True,
                title = None, labelfreqs=labelfreqs,labelcounts = labelcounts,
                measurementdf = measurementdf,
                legend = legend,s=s,bigcircle = bigcircle,demo=demo,
                ax = ax4) 
    set_format()
    plt.tight_layout()    
    if saving:
        filename = datestr + 'spectrum' + labelname
        helperfunctions.savefigure(filename)
    plt.show()        

    if context == 'paper':
        if MONOMER:
            figsize2 = (figwidth/2, figwidth/2.2) # what would best fit a circle?
        else:
            figsize2 = (figwidth, (1/2)*figwidth)
    else:
        if MONOMER:
            figsize2 = (5,4)
        else:
            figsize2 = (10, 4)
    
    if figsizeoverride2 is not None:
        figsize2 = figsizeoverride2
    print('figsize2:', figsize2)
    if MONOMER:
        fig2, ax5 = plt.subplots(1,1, figsize = figsize2)
    else:
        fig2, ((ax5, ax6)) = plt.subplots(1,2, figsize = figsize2)
    
    # svd curves
    #morefrequencies = np.linspace(minfreq, maxfreq, num = n*10)
    ax5.plot(realamp1(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 0, MONOMER, forceboth=forceboth), 
             imamp1(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 0, MONOMER, forceboth=forceboth), 
             '--', color='black', alpha = alpha_model)
    if not MONOMER:
        ax6.plot(realamp2(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 0,forceboth=forceboth,), 
                 imamp2(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 0,forceboth=forceboth,), 
                 '--', color='black', alpha = alpha_model)

    if context == 'paper':
        title1 = ''
        title2 = ''
        #cbar_label = '$\omega$ (rad/s)'
        cbar_label = ''
    else:
        title1 = 'Complex Amplitude $Z_1$'
        title2 = 'Complex Amplitude $Z_2$'
        cbar_label = 'Frequency (rad/s)'
    plotcomplex(Z1, drive, title1, ax=ax5, cbar_label=cbar_label,s=s,
                label_markers=labelfreqs )
    ax5.scatter(np.real(measurementdf.R1AmpCom), np.imag(measurementdf.R1AmpCom), 
                s=bigcircle, facecolors='none', edgecolors='k', label="points for analysis")
    if labelcounts:
        for i in range(len(measurementdf)):
            plt.annotate(text=str(i+1), 
                         xy=(np.real(measurementdf.R1AmpCom), 
                             np.imag(measurementdf.R1AmpCom)) )
    if MONOMER:
        plt.xlabel('Re($Z$) (m)')
        plt.ylabel('Im($Z$) (m)')
    else:
        plt.xlabel('Re($Z_1$) (m)')
        plt.ylabel('Im($Z_1$) (m)')        
        plotcomplex(Z2, drive,title2, ax=ax6, cbar_label=cbar_label,s=s,
                    label_markers=labelfreqs)
        ax6.scatter(np.real(measurementdf.R2AmpCom), np.imag(measurementdf.R2AmpCom), 
                    s=bigcircle, facecolors='none', edgecolors='k', label="points for analysis") 
        if labelcounts:
            for i in range(len(measurementdf)):
                plt.annotate(text=str(i+1), 
                             xy=(np.real(measurementdf.R2AmpCom), 
                                 np.imag(measurementdf.R2AmpCom)) )
        plt.xlabel('Re($Z_2$) (m)')
        plt.ylabel('Im($Z_2$) (m)')
       
    if show_set:
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
    
    if MONOMER:
        axs = [ax1, ax2, ax5]
    else:
        axs = [ax1, ax2, ax3, ax4, ax5, ax6]
       
    if legend:
        for ax in axs:
            plt.sca(ax)
            plt.legend()
            
    plt.tight_layout()
    if saving:
            filename = datestr + 'spectrumZ' 
            helperfunctions.savefigure(filename)
    plt.show()
            
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