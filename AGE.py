############ IMPORTS #################
import os
import json
import gc
import time
import sys
import shutil
import matplotlib

import numpy as np
import radio_beam as rb
import matplotlib.gridspec as gridspec

from pyBBarolo import GalMod as GM
from pyBBarolo import FitMod3D as Fit3D
from spectral_cube import SpectralCube as SC
from astropy import units as u
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy import wcs

# Our own library
from Functions import quickimage, getSNR, getσ, getμ, pickregion

class AGE(object):
    '''Artificial Galaxy Editor '''
    def __init__(self,skeleton_file,name_of_galaxy, noise_amplifier=9):
        '''Initialize skeleton for fits file and name of galaxy'''
         # class variables
        self.skeleton_file = skeleton_file
        self.name = name_of_galaxy
        self.noise_amplifier = noise_amplifier

        if 'F' in name_of_galaxy: self.ch = 50
        else: self.ch = 50

        # Report
        print('-'*20,'\nLoaded ' + self.skeleton_file + ' as skeleton file')
        print('Galaxy name: ',self.name)

        print('\n------------  WARNING -----------')
        print('DIRECTORY WILL BE MADE AT LOCATION OF PYTHON FILE\n\n')
        time.sleep(1.5)
    def clean_mem(self):
        '''Check python reference list'''
        print('Cleaning up memory....')
        gc.collect()
    def DirCheck(self):
        '''Create directory tree'''
        default = os.path.dirname(os.path.abspath(__file__))
        print('Checking directories...')
        time.sleep(1)
        if (os.path.isdir(default + '/AGs') == False):
            print('Directory not detected...creating...')
            os.mkdir(default + '/AGs')

        if (os.path.isdir(default + '/AGs/' + self.name) == False):
            print('Directory not detected...creating...')
            os.mkdir(default + '/AGs/' + self.name)
        self.output = default + '/AGs/' + self.name
    def Create_AG(self,params):
        '''Creates an artificial galaxy from input dictionary containing parameters'''
        # Do a check
        assert isinstance(params,dict), "Submitted parameters must be inside a dictionary"

        # Store parameter variable
        # Create directory tree
        self.params = params
        self.DirCheck()

        # Start model
        model = GM(self.skeleton_file)

        # Get position angle beam
        self.bpa = fits.open(self.skeleton_file)[0].header['bpa']
  
        # Initialize model
        model.init(radii=params['radii'],xpos=params['xpos'],ypos=params['ypos'],vsys=params['vsys'],vrot=params['vrot'],vrad=params['vrad'],
                        vdisp=params['vdisp'],z0=params['z0'],inc=params['inc'],phi=params['phi'],dens=params['dens'])

        model.set_options(ltype=params['ltype'])
        
        # Compute
        mymodel = model.compute()
        
        # Give galaxy a name
        mymodel.header['OBJECT'] = self.name

        # Write model result
        print('Raw model written to: ',self.output + '/' + 'raw_' + self.name + '.fits')
        mymodel.writeto(self.output + '/' + 'raw_' + self.name + '.fits',overwrite=True)
    def add_noise(self,signal_to_noise):
        print('In order to add noise please give a region of the signal in the format: (row1,row2,column1,column2)')
        beam = rb.Beam(major=self.params['radii'][-1]*self.params['beamfactor'][0]*u.arcsec, minor=self.params['radii'][-1]*self.params['beamfactor'][1]*u.arcsec, pa=self.params['phi']*u.deg)
        pickregion(self.output + '/' + 'raw_' + self.name + '.fits', beam)
        region = np.array(input().split(',')).astype(int)
        self.region = region

        print('Adding gaussian noise to galaxy.....')
        self.signal_to_noise = signal_to_noise
        rawfile = self.output + '/' + 'raw_' + self.name + '.fits'
        noisefile = self.output + '/' + 'noise_' + self.name + '.fits'

        SNR = getSNR(region,self.noise_amplifier,self.signal_to_noise,rawfile,beam,noisefile)

        # If the noise amplifier wasn't enough start from scratch
        if abs(round(SNR[0],1) - self.signal_to_noise)>0.1:
            N = 0
            # Find the ideal noise amplifier to realise the SNR of user input
            low = 1;high=18;
            while abs(round(SNR[0],1) - self.signal_to_noise)>.1 and N < 100:
                self.noise_amplifier = np.random.uniform(low=low,high=high)
                SNR = getSNR(region,self.noise_amplifier,self.signal_to_noise,rawfile,beam,noisefile)
                if (SNR[0] - self.signal_to_noise) > 3: low += 0.2
                if (SNR[0] - self.signal_to_noise) < 3: high -= 0.2
                print(SNR[0],SNR[1] ,self.noise_amplifier)
                N += 1
            if N == 100: sys.exit('No convergence for noise')
            SN = SNR[0]
            
        self.params['std'] = SNR[1]

        self.clean_mem()
        print('Memory cleaned')
        
        c_gal = SC.read(self.output + '/' + 'noise_' + self.name + '.fits')
        # Results
        print('Smoothened galaxy with beamsize ' + '(Major Axis, Minor Axis) = '+ f"{self.params['beamfactor']}" + f"x {self.params['radii'][-1]}" + 'arcseconds')
        print('FINAL GALAXY written to ',self.output + '/' + 'final_' + self.name + '.fits')
        c_gal.write(self.output + '/' + 'final_' + self.name + '.fits',overwrite=True)

        # Store uncertainty
        #self.params['std'] = getσ(c_gal.hdu.data)

        # Final result!
        self.c_gal = c_gal
    def Report(self):
        '''Display and process all results of the created datacube'''

        # moment maps
        moment_0 = self.c_gal.moment()
        print('2D GALAXY VIEW written to ',self.output + '/' + 'final_' + self.name + '_moment' +'.fits')
        moment_0.write(self.output + '/' + 'final_' + self.name + '_moment' +'.fits',overwrite=True)

        # Surface density fix
        self.params['dens'] = self.params['dens'].tolist()

        # Print some results, can later include some plots
        print('+'*20)
        print('DATACUBE STATISTICS:')
        print('\nσ: ',self.params['std'])
        print('Mean flux of signal: ',getμ(self.c_gal, self.region, self.ch))
        print('+'*20)

        
        # Storage of parameters
        print('Parameter settings written to ',self.output + '/' + 'params_' + self.name + '.json')
        with open(self.output + '/' + 'params_' + self.name + '.json','w') as ff:
            ff.write(json.dumps(self.params))

    def apply_bbarolo(self,pguess,discover=True,incfact=[1,1.1,1],PAfact=[1,1,1.25]):
        assert isinstance(pguess,dict), "Submitted parameters must be inside a dictionary"
        matplotlib.rcParams.update({'font.size': 13})
        colors = ["black","green","blue"]

        # Wait 5s, reading output/terminal etc
        print('\nStandby...5s')
        time.sleep(5)

        if discover == True:
            options= []
            print('\nDiscover mode is on, this will take a while... (est. 25s)\n')
            time.sleep(2)
            for i in range(3):
                f3d = Fit3D(self.output + '/' + 'final_' + self.name + '.fits')
                f3d.init(radii=pguess['radii'], xpos=pguess['xpos'], ypos=pguess['ypos'], vsys=pguess['vsys'], vrot=pguess['vrot'], vdisp=pguess['vdisp'], vrad=pguess['vrad'],
                    z0=pguess['z0'], inc=pguess['inc']*incfact[i], phi=pguess['phi']*PAfact[i])
        
                f3d.set_options(free='VROT VDISP', wfunc=pguess['wfunc'], ftype=pguess['ftype'], distance=3.2, outfolder=self.output + '/output' + str(i))
                f3d.set_beam(bmaj=pguess['radii'][-1]*pguess['beamfactor'][0], bmin=pguess['radii'][-1]*pguess['beamfactor'][1], bpa=pguess['phi']*u.deg)
                f3d.compute(threads=4)
                f3d.plot_model()
            
                options.append(f"incl = {round(pguess['inc']*incfact[i],1)},PA = {round(pguess['phi']*PAfact[i],1)}")

            try:
                BB_data1 = np.genfromtxt(self.output + '/output1' + '/ringlog1.txt')
            except:
                BB_data1 = np.genfromtxt(self.output + '/output1' + '/ringlog2.txt')
            BB_dprof1 = np.genfromtxt(self.output + '/output1' + '/densprof.txt')

            try:
                BB_data2 = np.genfromtxt(self.output + '/output2' + '/ringlog1.txt')
            except:
                BB_data2 = np.genfromtxt(self.output + '/output2' + '/ringlog2.txt')
            BB_dprof2 = np.genfromtxt(self.output + '/output2' + '/densprof.txt')

            BB_dens1 = BB_dprof1[:,10];BB_disp1 = BB_data1[:,3];BB_Rcurve1 = BB_data1[:,2];BB_radius1 = BB_data1[:,1]
            BB_dens2 = BB_dprof2[:,10];BB_disp2 = BB_data2[:,3];BB_Rcurve2 = BB_data2[:,2];BB_radius2 = BB_data2[:,1]

        else:
            options=[]

            f3d = Fit3D(self.output + '/' + 'final_' + self.name + '.fits')
            f3d.init(radii=pguess['radii'], xpos=pguess['xpos'], ypos=pguess['ypos'], vsys=pguess['vsys'], vrot=pguess['vrot'], vdisp=pguess['vdisp'], vrad=pguess['vrad'],
                z0=pguess['z0'], inc=pguess['inc']*incfact[i], phi=pguess['phi']*PAfact[i])
        
            f3d.set_options(mask='SEARCH', free='VROT VDISP', wfunc=pguess['wfunc'], ftype=pguess['ftype'], distance=3.2, outfolder=self.output + '/output0')
            f3d.set_beam(bmaj=pguess['radii'][-1]*pguess['beamfactor'][0], bmin=pguess['radii'][-1]*pguess['beamfactor'][1], bpa=pguess['phi']*u.deg)
            f3d.compute(threads=4)
            f3d.plot_model()

            options.append(f"incl = {round(pguess['inc'],1)},PA = {round(pguess['phi'],1)}")

        print('\n\nSummarizing results....')
        TData = json.load(open(self.output + '/' + 'params_' + self.name + '.json'))
        BB_data0 = np.genfromtxt(self.output + '/output0' + '/ringlog1.txt')
        BB_dprof0 = np.genfromtxt(self.output + '/output0' + '/densprof.txt')
        
        BB_dens0 = BB_dprof0[:,10];BB_disp0 = BB_data0[:,3];BB_Rcurve0 = BB_data0[:,2];BB_radius0 = BB_data0[:,1]
        Tdisp = TData['vdisp'];TRcurve = np.array(TData['vrot']);Tradius = np.array(TData['radii']);Tdens = TData['dens']

        if discover == True:
            Tvrot = pguess['rotfunc'](np.array(BB_radius0))
            #res_elements = np.array([indx for indx in range(len(Tradius)) for ii in range(len(BB_Rcurve0)) if abs(Tradius[indx] - BB_radius0[ii])<=35]) # radii are the same for BB , 16 for 20 radii, 10 for 40 radii
            #res_elements = res_elements.astype(int)
            #residual = [np.sum(abs(TRcurve[res_elements]-BB_Rcurve0)),np.sum(abs(TRcurve[res_elements]-BB_Rcurve1)),np.sum(abs(TRcurve[res_elements]-BB_Rcurve2))]
            residual = [np.sum(abs(Tvrot - BB_Rcurve0)), np.sum(abs(Tvrot - BB_Rcurve1)), np.sum(abs(Tvrot - BB_Rcurve2))]
        else: 
            Tvrot = pguess['rot'](np.array(BB_radius0))
            #res_elements = np.array([indx for indx in range(len(Tradius)) for ii in range(len(BB_Rcurve0)) if abs(Tradius[indx] - BB_radius0[ii])<=35]) # radii are the same for BB , 16 for 20 radii, 10 for 40 radii
            #res_elements = res_elements.astype(int)
            #residual = [np.sum(abs(TRcurve[res_elements]-BB_Rcurve0)),np.sum(abs(TRcurve[res_elements]-BB_Rcurve1)),np.sum(abs(TRcurve[res_elements]-BB_Rcurve2))]
            residual = [np.sum(abs(Tvrot - BB_Rcurve0))]

        # Plot body:
        matplotlib.rcParams.update({'font.size': 14})
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(nrows=2, ncols=3,height_ratios=[1, 0.8])
        
        # Vrot
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(Tradius,TRcurve,'k--',color='red')
        ax0.scatter(BB_radius0, BB_Rcurve0,color='black',marker=(5,1),label=r'$^{3D}Barolo$#0')
        if discover==True:
            ax0.scatter(BB_radius1, BB_Rcurve1,color='green',marker=(5,0),label=r'$^{3D}Barolo$#1')
            ax0.scatter(BB_radius2, BB_Rcurve2,color='blue',marker=(5,2),label=r'$^{3D}Barolo$#2')
        ax0.set_title("Rotation curve",fontsize=14)
        ax0.set_ylabel(r"$V(R)$ (km/s)")
        ax0.set_xlabel("Radius [arcseconds]")

        #Vdisp
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.plot(Tradius,Tdisp, 'k--',color='red')
        ax1.scatter(BB_radius0, BB_disp0,color='black',marker=(5,1),label=r'$^{3D}Barolo$#0')
        if discover==True:
            ax1.scatter(BB_radius1, BB_disp1,color='green',marker=(5,0),label=r'$^{3D}Barolo$#1')
            ax1.scatter(BB_radius2, BB_disp2,color='blue',marker=(5,2),label=r'$^{3D}Barolo$#2')
                
        ax1.set_title("Velocity dispersion",fontsize=14)
        ax1.set_ylabel(r"$\sigma_{gas}$ (km/s)")
        ax1.set_xlabel("Radius [arcseconds]")

        # Surface density Msun pc^2
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.plot(Tradius,Tdens, 'k--',color='red')            
        ax2.plot(BB_radius0,BB_dens0/np.max(BB_dens0),'k--',color='black',label=r'$^{3D}Barolo$#0')
        ax2.plot(BB_radius1,BB_dens1/np.max(BB_dens1),'k--',color='green',label=r'$^{3D}Barolo$#1')
        ax2.plot(BB_radius2,BB_dens2/np.max(BB_dens2),'k--',color='blue',label=r'$^{3D}Barolo$#2')
        ax2.set_title("Surface density",fontsize=14)
        ax2.set_ylabel(r"$\Sigma_{HI}$ Normalized")
        ax2.set_xlabel("Radius [arcseconds]")

        # Residual plot of Vrot
        ax3 = fig.add_subplot(gs[:,2])
        ax3.spines["top"].set_visible(False)    
        ax3.spines["bottom"].set_visible(False)    
        ax3.spines["right"].set_visible(False)    
        ax3.spines["left"].set_visible(False)
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().tick_left()   
        ax3.set_title('Sum of residuals',fontsize=14)

        for rank, column in enumerate(options):    
            lgnd = [r'$^{3D}Barolo$#0',r'$^{3D}Barolo$#1',r'$^{3D}Barolo$#2']
            ax3.hlines(y=residual[rank],xmin=Tradius[0],xmax=Tradius[-1],lw=2.5, color=colors[rank],label=lgnd[rank]) 
            plt.text(Tradius[-1], residual[rank], column, fontsize=12, color=colors[rank])

        plt.text(0, 0,f"Artificial Galaxy analysis: (Galaxy={self.name})(S/N={self.signal_to_noise} )",
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax3.transAxes)

        ax0.legend();ax1.legend();ax2.legend();ax3.legend(loc='lower center',bbox_to_anchor=(0.5,1.05),
                ncol=3, fancybox=True, shadow=True)
        fig.tight_layout()
        plt.savefig(self.output + '/AGs-BBaroloComparison.png')
        plt.show()



#######################################################################################################

rmax = 900# arcsecs
h = 70
radii=np.linspace(0,rmax,30) # becomes converted to a list later on no worries.
xpos=76;ypos=76 # pixels
vsys= 120
vrot=(2/np.pi * 120 * np.arctan(radii/70)).tolist() # 80-140 is the maximum range
vrad=0
vdisp= (10*np.exp(-radii/(300)) + 8.5).tolist()  #  30-5
rotfunc = lambda r: 2/np.pi * 120 * np.arctan(r/h)
RHI =  990 #(10**((np.log10(3e9) - 6.92)/1.72))/2
print((10**((np.log10(3e9) - 6.92)/1.72))/2)
dens= (10*np.exp(-(radii - 0.39*RHI)**2 /(2*(0.35*RHI)**2)))/np.max(10*np.exp(-(radii - 0.39*RHI)**2 /(2*(0.35*RHI)**2)))
z0=1
inc=50;phi=90 # degrees
beamfactor = (0.25,0.2) # Major axis > Minor axis
ltype = 3 # Exponential = 3
ftype = 2 # Chi-sq = 1, |mod-obs| = 2
wfunc = 0 # Uniform = 0, cos theta^2 = 2

# flat: 'vrot':[77.48,104.377,110.542,113.227], 'vdisp':[13.224,10.731,9.554,8.998]
# rising: 'vrot':[86.066,100.849,111.547,121.198], 'vdisp':[13.224,10.731,9.554,8.998]
''' 
Modify pdict to produce an artificial galaxy, and modify pguess to have BBarolo retrieve the parameters stated in pdict.
'''
pdict = {'radii':radii.tolist(), 'xpos':xpos, 'ypos':ypos, 'vsys':vsys, 'vrad':vrad, 'vrot':vrot, 'vdisp':vdisp, 'z0':z0, 'inc':inc, 'phi':phi, 'beamfactor':beamfactor, 'ltype':ltype,'dens':dens , 'std':None}
pguess = {'radii':[225/2,225/2 + 225,225/2 + 2*225,225/2 + 3*225], 'xpos':xpos, 'ypos':ypos, 'vsys':vsys, 'vrad':vrad, 'vrot':[96.958,108.211,112.106,114.070], 'vdisp':[13.224,10.731,9.554,8.998], 'z0':z0, 'inc':inc, 'phi':phi, 'beamfactor':beamfactor, 'ltype':ltype, 'ftype':ftype, 'wfunc':wfunc, 'dens':[0.9,0.2], 'rotfunc':rotfunc}
#######################################################################################################
'''
Change the skeleton pathfile if it is not located at location_of_this_pythonfile/examples/U3_25160_Ha.fits.
Change the name of the galaxy by modifying the 3rd parameter of the initialization of the object. This python script will only compare models if generic name 'Galaxy1' is used. (Just to be safe) 
'''

AG = AGE(os.path.dirname(os.path.abspath(__file__))+ '/examples/ngc2403.fits', 'G10FC')
AG.Create_AG(pdict)
AG.add_noise(signal_to_noise=10)
AG.Report()

'''
Turn the statement below off, if you don't want BBarolo to retrieve the parameters via pyBBarolo
'''
AG.apply_bbarolo(pguess)
#######################################################################################################
