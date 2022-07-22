"""
#File created 2022-07-22 by Viva Horowitz
#Plots of singular value decomposition
"""

import matplotlib.pyplot as plt
import numpy as np



alpha_circles = .8 ## set transparency for the black circles around the measurement values
alpha_model = .8   ## set transparency for the dashed black line
alpha_data = .8

def spectrum_plot(drive, noisydata,morefrequencies, noiseless, curvefunction,
                  K1, K2, K12, B1, B2, FD, M1, M2,
                  dfcolumn,
                  ylabel,
                  title, labelfreqs,
                  measurementdf, ax, unitsofpi = False,tooclose = .1):
    if unitsofpi:
        divisor = np.pi
    else:
        divisor = 1
    ax.plot(morefrequencies, noiseless/divisor, 
            '-', color = 'gray', label='set values') # intended curves
    ax.plot(drive, noisydata/divisor, 
            '.', color = 'C0', alpha = alpha_data, label='simulated data') # simulated data
    ax.plot(morefrequencies, curvefunction(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 0)/divisor, 
            '--', color='black', alpha = alpha_model, label='SVD results') # predicted spectrum from SVD)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    #For loop to plot R1 amplitude values from table
    for i in range(measurementdf.shape[0]):
        ax.plot(measurementdf.drive[i], (measurementdf[dfcolumn])[i]/divisor, 
                'ko', fillstyle='none', markeredgewidth = 3, alpha = alpha_circles)
    ax.set_xlabel('Freq (rad/s)')
    
    labelfreqs = list(labelfreqs)
    if MONOMER or labelfreqs is None or labelfreqs == []:
        return;
    
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

"""
measurementdf has a row for each measured frequency
columns: drive, R1Amp, R1Phase, R2Amp, R2Phase, R1AmpCom, R2AmpCom
"""
def plot_SVD_results(drive,R1_amp,R1_phase,R2_amp,R2_phase, measurementdf,  K1, K2, K12, B1, B2, FD, M1, M2, 
                     vals_set, MONOMER, forceboth, labelfreqs = None):
    [m1_set, m2_set, b1_set, b2_set, k1_set, k2_set, k12_set, F_set] = read_params(vals_set, MONOMER)
        
    Z1 = complexamp(R1_amp, R1_phase)
    Z2 = complexamp(R2_amp, R2_phase)
    
    morefrequencies = makemorefrequencies(min(drive), max(drive), includefreqs = drive, 
                                          vals_set = vals_set, MONOMER=MONOMER, n=10*len(drive))
    
    R1_amp_noiseless = curve1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER, forceboth=forceboth)
    R1_phase_noiseless = theta1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER, forceboth=forceboth)
    if not MONOMER:
        R2_amp_noiseless = curve2(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)
        R2_phase_noiseless = theta2(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)

    R1_real_amp_noiseless = realamp1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER, forceboth=forceboth)
    R1_im_amp_noiseless = imamp1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER, forceboth=forceboth)
    if not MONOMER:
        R2_real_amp_noiseless = realamp2(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)
        R2_im_amp_noiseless = imamp2(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, forceboth=forceboth)
    
    if labelfreqs is None:
        if len(measurementdf) <= 3:
            labelfreqs = list(measurementdf.drive)
        elif not MONOMER and k12_set < 10:
            # This is an approximation for weak coupling, 
            # not so good for strong coupling, but still a nice pair of frequencies to identify.
            res1 = res_freq_weak_coupling(k1_set, m1_set, b1_set)
            res2 = res_freq_weak_coupling(k2_set, m2_set, b2_set)
            labelfreqs = [res1, res2]
        # otherwise labelfreqs is still None
    labelfreqs = [f.real for f in labelfreqs if f.imag == 0] # force labelfreqs to be real
    
    #fig, ((ax1, ax3),(ax2,ax4),(ax5, ax6)) = plt.subplots(3,2, figsize = (10,10))
    if MONOMER:
        fig, ((ax1),(ax2)) = plt.subplots(2,1, figsize = (9.5/2,5), gridspec_kw={'hspace': 0}, sharex = 'all' )
    else:
        fig, ((ax1, ax3),(ax2,ax4)) = plt.subplots(2,2, figsize = (9.5,5), gridspec_kw={'hspace': 0}, sharex = 'all' )

    spectrum_plot(drive=drive, noisydata=R1_amp,
                morefrequencies=morefrequencies, noiseless=R1_amp_noiseless, 
                curvefunction = curve1,
                K1=K1, K2=K2, K12=K12, B1=B1, B2=B2, FD=FD, M1=M1, M2=M2,
                dfcolumn = 'R1Amp',
                ylabel = 'Amplitude $A$ (m)\n',
                title = 'Simulated R1 Spectrum', labelfreqs=labelfreqs,
                measurementdf = measurementdf,
                ax = ax1) 
        
    spectrum_plot(drive=drive, noisydata=R1_phase,
                morefrequencies=morefrequencies, noiseless=R1_phase_noiseless, 
                curvefunction = theta1,
                K1=K1, K2=K2, K12=K12, B1=B1, B2=B2, FD=FD, M1=M1, M2=M2,
                dfcolumn = 'R1Phase',
                ylabel = 'Phase $\delta$ ($\pi$)', unitsofpi = True,
                title = None, labelfreqs=labelfreqs,
                  measurementdf = measurementdf,
                ax = ax2) 
        
    """    ax2.plot(morefrequencies, R1_phase_noiseless/np.pi, '-', color = 'gray', label='set values') # intended curves
    ax2.plot(drive, R1_phase/np.pi, '.', color = 'C0', alpha = alpha_data)
    ax2.plot(morefrequencies, theta1(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 0)/np.pi, 
             '--', color='black', alpha = alpha_model)
    ax2.set_ylabel('Phase $\delta$ ($\pi$)')
    #ax2.set_title('Simulated R1 Phase')

    #For loop to plot R1 amplitude values from table
    for i in range(measurementdf.shape[0]):
        ax2.plot(measurementdf.drive[i], measurementdf.R1Phase[i]/np.pi, 'ko', 
                 fillstyle='none', markeredgewidth = 3, alpha = alpha_circles)"""

    if not MONOMER:
        spectrum_plot(drive=drive, noisydata=R2_amp,
                morefrequencies=morefrequencies, noiseless=R2_amp_noiseless, 
                curvefunction = curve2,
                K1=K1, K2=K2, K12=K12, B1=B1, B2=B2, FD=FD, M1=M1, M2=M2,
                dfcolumn = 'R2Amp',
                ylabel = 'Amplitude $A$ (m)\n', unitsofpi = False,
                title = 'Simulated R2 Spectrum', labelfreqs=labelfreqs,
                      measurementdf = measurementdf,
                ax = ax3) 
            
        """        ax3.plot(morefrequencies, R2_amp_noiseless, '-', color = 'gray', label='set values') # intended curves
        ax3.plot(morefrequencies, curve2(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 0), 
                 '--', color='black', alpha = alpha_model)
        ax3.plot(drive, R2_amp, '.', color = 'C0', alpha = alpha_data)
        ax3.set_ylabel('Amplitude $A$ (m)\n')
        ax3.set_title('Simulated R2 Spectrum')

        #For loop to plot R1 amplitude values from table
        for i in range(measurementdf.shape[0]):
            ax3.plot(measurementdf.drive[i], measurementdf.R2Amp[i], 
                     'ko', fillstyle='none', markeredgewidth = 3, alpha = alpha_circles)"""
        
        spectrum_plot(drive=drive, noisydata=R2_phase,
                morefrequencies=morefrequencies, noiseless=R2_phase_noiseless, 
                curvefunction = theta2,
                K1=K1, K2=K2, K12=K12, B1=B1, B2=B2, FD=FD, M1=M1, M2=M2,
                dfcolumn = 'R2Phase',
                ylabel = 'Phase $\delta$ ($\pi$)', unitsofpi = True,
                title = None, labelfreqs=labelfreqs,
                      measurementdf = measurementdf,
                ax = ax4) 
        
        """        ax4.plot(morefrequencies, theta2(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 0)/np.pi, 
                 '--', color='black', alpha = alpha_model)
        ax4.plot(morefrequencies, R2_phase_noiseless/np.pi, '-', color = 'gray', label='set values') # intended curves
        ax4.plot(drive, R2_phase/np.pi, '.', color = 'C0', alpha = alpha_data)

        ax4.set_ylabel('Phase $\delta$ ($\pi$)')
        #ax4.set_title('Simulated R2 Phase')

        #For loop to plot R1 amplitude values from table
        for i in range(measurementdf.shape[0]):
            ax4.plot(measurementdf.drive[i], measurementdf.R2Phase[i]/np.pi, 
                     'ko', fillstyle='none', markeredgewidth = 3, alpha = alpha_circles)"""

    plt.tight_layout()    

    if MONOMER:
        fig2, ax5 = plt.subplots(1,1, figsize = (10/2,4))
    else:
        fig2, ((ax5, ax6)) = plt.subplots(1,2, figsize = (10,4))
    
    # svd curves
    #morefrequencies = np.linspace(minfreq, maxfreq, num = n*10)
    ax5.plot(realamp1(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 0, MONOMER), 
             imamp1(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 0, MONOMER), 
             '--', color='black', alpha = alpha_model)
    if not MONOMER:
        ax6.plot(realamp2(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 0), 
                 imamp2(morefrequencies, K1, K2, K12, B1, B2, FD, M1, M2, 0), 
                 '--', color='black', alpha = alpha_model)

    plotcomplex(Z1, drive, 'Complex Amplitude $Z_1$', ax=ax5, label_markers=labelfreqs)
    ax5.scatter(np.real(measurementdf.R1AmpCom), np.imag(measurementdf.R1AmpCom), s=150, facecolors='none', edgecolors='k')
    if not MONOMER:
        plotcomplex(Z2, drive, 'Complex Amplitude $Z_2$', ax=ax6, label_markers=labelfreqs)
        ax6.scatter(np.real(measurementdf.R2AmpCom), np.imag(measurementdf.R2AmpCom), s=150, facecolors='none', edgecolors='k') 

    # true curves
    #morefrequencies = np.linspace(minfreq, maxfreq, num = n*10)
    ax5.plot(realamp1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER), 
             imamp1(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0, MONOMER), 
             color='gray', alpha = .5)
    if not MONOMER:
        ax6.plot(realamp2(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0), 
                 imamp2(morefrequencies, k1_set, k2_set, k12_set, b1_set, b2_set, F_set, m1_set, m2_set, 0), 
                 color='gray', alpha = .5)

    plt.tight_layout()
	
""" convert from a format used for resultsdf (in the frequency sweep below) 
    to the format used in the above function plot_SVD_results() """
def convert_to_measurementdf(resultsdfitem):
    # columns of measurementdf: drive, R1Amp, R1Phase, R2Amp, R2Phase, R1AmpCom, R2AmpCom
    mdf = pd.DataFrame([[resultsdfitem.Freq1, resultsdfitem.R1Amp_a, resultsdfitem.R1Phase_a, resultsdfitem.R2Amp_a, 
         resultsdfitem.R2Phase_a, resultsdfitem.R1AmpCom_a, resultsdfitem.R2AmpCom_a],
         [resultsdfitem.Freq2, resultsdfitem.R1Amp_b, resultsdfitem.R1Phase_b, resultsdfitem.R2Amp_b, 
         resultsdfitem.R2Phase_b, resultsdfitem.R1AmpCom_b, resultsdfitem.R2AmpCom_b]],
            columns = ['drive', 'R1Amp', 'R1Phase', 'R2Amp', 'R2Phase', 'R1AmpCom', 'R2AmpCom'])
    return mdf # measurement dataframe
	
""" label_markers are frequencies to label 
    parameter = drive """
def plotcomplex(complexZ, parameter, title = 'Complex Amplitude', cbar_label='Frequency (rad/s)', 
                label_markers=[], ax=plt.gca(), s=50, cmap = 'rainbow'):
    plt.sca(ax)
    sc = ax.scatter(np.real(complexZ), np.imag(complexZ), s=s, c = parameter, cmap = cmap ) # s is marker size
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
        
        ## I'm not sure why def plotcomplex draws an empty graph below.