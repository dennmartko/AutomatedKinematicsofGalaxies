import os
import json
import gc
import time
import shutil
import scipy.stats
import sys
import aplpy
import matplotlib

import numpy as np
import radio_beam as rb
import matplotlib.gridspec as gridspec
import seaborn as sns

from pyBBarolo import GalMod as GM
from pyBBarolo import FitMod3D as Fit3D
from spectral_cube import SpectralCube as SC
from astropy import units as u
from astropy.io import fits
from astropy.coordinates import Galactic
from astropy import wcs
from pvextractor import PathFromCenter,Path
from pvextractor import extract_pv_slice
from matplotlib import pyplot as plt
from astropy import wcs
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
from matplotlib import ticker, cm
import matplotlib as mpl 
from matplotlib.patches import Circle
from numpy import ma
from kapteyn import maputils
from astropy.visualization import LinearStretch, PowerStretch 
from astropy.visualization.mpl_normalize import ImageNormalize 
from astropy.visualization import PercentileInterval 
from Functions import createmodelgal, NumpyEncoder,getσ
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_global(file, save):
    with fits.open(file) as ff:
        cube = SC.read(ff)
        vel = cube.spectral_axis * u.s / u.km
        print(vel)

        slice_unmasked = cube.unmasked_data[:,:,:]/cube.unit

        vel_emission = [np.sum(slice_unmasked[i]) for i in range(len(cube))]
        vel_emission = vel_emission/(np.max(vel_emission))# Normalized

        plt.plot(vel,vel_emission, color='grey')
        plt.xlabel(r'$V_{LOS} [km/s]$')
        plt.ylabel('Normalized Intensity')
        plt.title('Observed HI line emission')
        plt.savefig(save + '/globalprofile.png')
        plt.show()

def plot_moments(file, save):
    # FIX UNITS
    fits.setval(file, 'CUNIT3', value='km/s')

    with fits.open(file) as ff:
        cube = SC.read(ff)
        vel = cube.spectral_axis * u.s / u.km

        slice_unmasked = cube.unmasked_data[:,:,:]/cube.unit

        m0 = cube.moment(order=0)
        m1 = cube.moment(order=1)
        m2 = cube.moment(order=2)

        l = len(m1.value[0]); b = len(m1.value[:,0])
        LL, BB = np.meshgrid(np.arange(0,l),np.arange(0,b))

        z = m1.hdu.data
        #z = ma.masked_where(m1.hdu.data < vel[-1], m1.hdu.data)
        #z = ma.masked_where(z < 0, z)
        
        ticks = np.logspace(1,2.5,30)
        plt.imshow(z, cmap='plasma')
        cs = plt.contourf(LL,BB,z ,levels=ticks[::1],locator=ticker.LogLocator() ,cmap='plasma', alpha=0.9)
        cbar = plt.colorbar(cs)
        cbar.set_ticklabels(np.round(ticks[::2]).astype(int))
        cbar.set_label(r'$V_{LOS}$ [km/s]')
        plt.axis('off')
        plt.savefig(save + '_velocityfield.png')
        plt.show()

        z = m0.hdu.data/np.max(m0.hdu.data)

        ticks = np.arange(0,1,0.001)
        plt.imshow(z,origin='lower', cmap='plasma')
        cs = plt.contourf(LL,BB,z,origin='lower',levels=ticks[::1], cmap='plasma', alpha=0.9)
        cbar = plt.colorbar(cs)
        cbar.set_ticklabels(np.round(ticks[::100],2))
        plt.axis('off')
        plt.title('Normalized HI Flux')
        plt.savefig(save + '_Galaxyflux.png')
        plt.show()

        z = np.abs(m2.hdu.data) #/np.max(m2.hdu.data)

        ticks = np.logspace(0,4,50)
        plt.imshow(z, cmap='plasma')
        cs = plt.contourf(LL,BB,z,levels=ticks[::1], cmap='plasma', alpha=0.9)
        cbar = plt.colorbar(cs)
        cbar.set_ticklabels(np.round(ticks[::1],2))
        plt.axis('off')
        #plt.title('Normalized HI Flux')
        plt.savefig(save + '_dispersion.png')
        plt.show()

def plot_comparison(galname):
    '''Only works for 90 degrees PA // any n * 90 degrees where n != 1, is also psosible but needs manually adjustment'''
    matplotlib.rcParams.update({'font.size': 15})
    if os.path.isdir(os.path.dirname(os.path.abspath(__file__)) + "/Comparison") == False:
        os.mkdir(os.path.dirname(os.path.abspath(__file__)) + "/Comparison")
        sys.exit("Put all relevant files in the directory: {}".format(os.path.dirname(os.path.abspath(__file__))+ "/Comparison"))

    path_to_dir = os.path.dirname(os.path.abspath(__file__)) + "/Comparison"
    # Load params
    TData = json.load(open(path_to_dir + '/params_' + galname + '.json'))
    MData = json.load(open(path_to_dir + '/params_model.json'))

    params = {'radii':MData['radii'], 'xpos':TData['xpos'], 'ypos':TData['ypos'], 'vsys':TData['vsys'], 'vrad':TData['vrad'], 'vrot':MData['vrot'], 'vdisp':MData['vdisp'], 'z0':TData['z0'], 'inc':MData['inc'], 'phi':MData['phi'],
             'ltype':TData['ltype'], 'std':TData['std'], 'beamfactor':TData['beamfactor'], 'bpa':TData['phi']}
    createmodelgal(path_to_dir + '/model.fits',path_to_dir + '/examples/ngc2403.fits',params)

    galaxy1 = path_to_dir + '/final_' + galname + '.fits'
    galaxy2 = path_to_dir + '/model.fits'
    mask = path_to_dir + '/output0/mask.fits'

    # FIX UNITS
    fits.setval(galaxy1, 'CUNIT3', value='km/s')
    fits.setval(galaxy2, 'CUNIT3', value='km/s')
    with fits.open(galaxy1) as ff:
        with fits.open(galaxy2) as fff:
            with fits.open(mask) as mk:
                # Refine the data
                realgal = SC.read(ff)
                modelgal = SC.read(fff)
                
                maskgal = mk[0].data
                img_mas = maskgal
                maskgal = np.sum(maskgal, axis=0)

                realgal_unmasked = realgal.unmasked_data[:,:,:]/realgal.unit
                modelgal_unmasked = modelgal.unmasked_data[:,:,:]/modelgal.unit

                realgal_vel = realgal.spectral_axis * u.s / u.km
                modelgal_vel = modelgal.spectral_axis * u.s / u.km

                realgal_m1 = realgal.moment(order=1)
                realgal_m1.write('/home/dennmarko/temp_realgalm1.fits',overwrite=True)
                realgal_m1 = fits.open('/home/dennmarko/temp_realgalm1.fits')[0].data
                modelgal_m1 = modelgal.moment(order=1)
                modelgal_m1.write('/home/dennmarko/temp_modelgalm1.fits',overwrite=True)
                modelgal_m1 = fits.open('/home/dennmarko/temp_modelgalm1.fits')[0].data

                realgal_pv = np.array([[col-TData['vsys'] if maskgal[indx1,indx2] != 0 else 0 for indx2,col in enumerate(row)] for indx1,row in enumerate(realgal_m1)])
                modelgal_pv = np.array([[col-TData['vsys'] if maskgal[indx1,indx2] != 0 else 0 for indx2,col in enumerate(row)] for indx1,row in enumerate(modelgal_m1)])
                    
                realgal_PV = [realgal_pv[75][i] for i in range(len(realgal_pv)) if realgal_pv[75][i] != 0]
                modelgal_PV = modelgal_pv[75]
                #modelgal_PV = [modelgal_pv[75][i] for i in range(len(modelgal_pv)) if modelgal_pv[75][i] != 0] 
                pixel = [i for i in range(len(realgal_pv)) if realgal_pv[75][i] != 0]

                # Convert pixels to arcseconds
                header = fits.getheader(os.path.dirname(os.path.abspath(__file__)) + '/Comparison/final_' + galname + '.fits')
                pixel2arcsec = lambda pixels,xpos: (pixels-xpos)*(header['cdelt']*3600)
                arcsecperpix = header['cdelt1']*3600

                parampixelindx_receding = np.rint(np.array(MData['radii'])/arcsecperpix).astype(int) + TData['xpos']
                parampixelindx_approaching = TData['xpos'] - np.rint(np.array(MData['radii'])/arcsecperpix).astype(int)
                parampixelindx = np.sort(np.concatenate((parampixelindx_approaching,parampixelindx_receding)))
                realgalpixelindx_receding = (np.array(TData['radii'])/arcsecperpix).astype(int) - TData['xpos']
                realgalpixelindx_approaching = TData['xpos'] + (np.array(TData['radii'])/arcsecperpix).astype(int)
                realgalpixelindx = np.sort(np.concatenate((realgalpixelindx_approaching,realgalpixelindx_receding)))

                def cross_hair(x, y, ax=None, **kwargs):
                    if ax is None:
                        ax = plt.gca()
                    horiz = ax.axhline(y, **kwargs)
                    vert = ax.axvline(x, **kwargs)
                    return horiz, vert

                ##################
                ### PV-diagram ###
                ##################
                header = fits.getheader(galaxy1)
                mywcs = wcs.WCS(header)

                pix_coords = [(0,76,0), (150,76,0), (76,76,0)]
                world_coords = mywcs.wcs_pix2world(pix_coords, 0)
                ra,dec = world_coords[0][:2]
                datacube = SC.read(galaxy1)
                modelcube = SC.read(galaxy2)
                vel = (datacube.spectral_axis * u.s / u.km)
                path = Path([(0,76),(150,76)], width=1)
                dataslice = extract_pv_slice(datacube, path)
                modelslice = extract_pv_slice(modelcube, path)

                # extract axis information
                mpl.rc('xtick',direction='in') 
                mpl.rc('ytick',direction='in') 
                mpl.rcParams['contour.negative_linestyle'] = 'solid' 
                plt.rc('font',family='sans-serif',serif='Helvetica',size=10)  
                params = {'text.usetex': False, 'mathtext.fontset': 'cm', 'mathtext.default': 'regular', 'errorbar.capsize': 0} 
                plt.rcParams.update(params) 
                crpixpv = header['crpix1']
                cdeltpv = header['cdelt1']
                crvalpv = header['crval1']
                xminpv,xmaxpv = np.floor(crpixpv-1-83), np.ceil(crpixpv-1+83)
                if xminpv == 0: xminpv = 0
                if xmaxpv >= header['naxis1']: xmaxpv=header['naxis1']-1
                zmin = 0
                zmax = len(vel) - 1
                data_maj = dataslice.data[zmin:zmax+1,int(xminpv):int(xmaxpv)+1] 
                model_maj = dataslice.data[zmin:zmax+1,int(xminpv):int(xmaxpv)+1] 
                xmin_wcs = ((xminpv+1-crpixpv)*cdeltpv+crvalpv)
                xmax_wcs = ((xmaxpv+1-crpixpv)*cdeltpv+crvalpv)

                zmin_wcs, zmax_wcs = 289.611, -24.7135

                datarad = np.array(TData['radii']) - 112.5
                dataradius = np.concatenate((-datarad,datarad))
                modelrad = np.array(MData['radii']) - 112.5
                modelradius = np.concatenate((-modelrad,modelrad))

                datavrot, dataincl, datavsystem, dataphi = np.array(TData['vrot']),TData['inc'],TData['vsys'],TData['phi']
                modelvrot, modelincl, modelvsystem, modelphi = np.array(MData['vrot']),MData['inc'],MData['vsys'],MData['phi']

                datavlos1 = datavrot*np.sin(np.deg2rad(dataincl))+datavsystem 
                datavlos2 = datavsystem-datavrot*np.sin(np.deg2rad(dataincl)) # reverse
                datavlos = np.concatenate((datavlos1,datavlos2)) 
                modelvlos1 = modelvrot*np.sin(np.deg2rad(modelincl))+modelvsystem 
                modelvlos2 = modelvsystem-modelvrot*np.sin(np.deg2rad(modelincl)) # reverse
                modelvlos = np.concatenate((modelvlos1,modelvlos2)) 

                xcen, ycen= [np.nanmean(MData['xpos'])-0,np.nanmean(MData['ypos'])-0] 

                datavsysmean, datapamean = np.nanmean(datavsystem), np.nanmean(dataphi) 
                modelvsysmean, modelpamean = np.nanmean(modelvsystem), np.nanmean(modelphi)
                dataext = [(114.248-xmin_wcs)*3600,(114.248-xmax_wcs)*3600,zmin_wcs-datavsysmean,zmax_wcs-datavsysmean]
                modelext = [(114.248-xmin_wcs)*3600,(114.248-xmax_wcs)*3600,zmin_wcs-modelvsysmean,zmax_wcs-modelvsysmean]

                cont = TData['std']
                v = np.array([-2,2,4,8,16,32])*cont
                labsize=15
                
                # Plot body of real galaxy vs model
                fig = plt.figure(figsize=(15, 15))
                gs1 = gridspec.GridSpec(nrows=1, ncols=1)
                ##############
                ### DATAPV ###
                ##############
                axis0 = fig.add_subplot(gs1[0,0])
                axis0.tick_params(which='major',length=8, labelsize=labsize)
                axis0.set_xlabel('Offset (arcsec)',fontsize=labsize+2)
                axis0.set_ylabel('$\mathrm{\Delta V_{LOS}}$ (km/s)',fontsize=labsize+2) 
                axis0.text(1, 1.02,'$\phi = $90$^\circ$',ha='right',transform=axis0.transAxes,fontsize=labsize+4) 
                axis00 = axis0.twinx() 
                axis00.set_xlim([dataext[0],dataext[1]]) 
                axis00.set_ylim([dataext[2]+datavsysmean,dataext[3]+datavsysmean]) 
                axis00.tick_params(which='major',length=8, labelsize=labsize) 
                axis00.set_ylabel('$\mathrm{V_{LOS}}$ (km/s)',fontsize=labsize+5) 
                axis0.imshow(dataslice.data,origin='lower',cmap=mpl.cm.Greys,extent=dataext,aspect='auto') 
                axis0.contour(dataslice.data,v,origin='lower',linewidths=0.7,colors='#00008B',extent=dataext) 
                #axis.contour(toplot[0],-cont,origin='lower',linewidths=0.1,colors='gray',extent=ext) 
                axis0.axhline(y=0,color='black') 
                axis0.axvline(x=0,color='black') 
                #axis00.plot(dataradius,datavlos,'yo')

                ##############
                ## MODELPV ###
                ##############

                axis0.contour(modelslice.data,v,origin='lower',linewidths=0.7,colors='red',extent=dataext) 
                axis00.plot(modelradius,modelvlos,'yo')
                #axis.contour(toplot[0],-cont,origin='lower',linewidths=0.1,colors='gray',extent=ext) 
                '''
                axis1.axhline(y=0,color='black') 
                axis1.axvline(x=0,color='black') 
                '''
                plt.savefig(path_to_dir + "/" + galname + "-PV-Diagram.pdf")
                plt.show()
                plt.close()

                ###################################
                ## VELOCITY FIELD + CHANNEL MAPS ##
                ###################################
                widths = [1, 1,1,1,1,1,1,1,1,1]
                heights = [0.7, 1]
                matplotlib.rcParams.update({'font.size': 16})
                fig = plt.figure(figsize=(20, 20))
                gs2 = gridspec.GridSpec(nrows=2, ncols=10,width_ratios=widths,
                          height_ratios=heights)
                xmin, xmax = 14, 138
                ymin, ymax = 14, 138
                mom1 = realgal_m1
                maskmap = np.nansum(img_mas,axis=0).astype(np.float)
                maskmap[maskmap>0]  = 1
                maskmap[maskmap==0] = np.nan
                ext = [0,xmax-xmin,0, ymax-ymin] 
                rad_pix = datarad/16
                x_pix = rad_pix*np.cos(np.radians(dataphi-90)) 
                y_pix = rad_pix*np.sin(np.radians(dataphi-90))
                x = np.arange(0,xmax-xmin,0.1) 
                y = np.tan(np.radians(dataphi-90))*(x-xcen)+ycen

                ###########################
                ### Velocity field plot ###
                ###########################
                rbeam = TData['beamfactor'][0]/2 *TData['radii'][-1]/(abs(cdeltpv)*3600)
                beamarr = np.array([76 - 112.5/(abs(cdeltpv)*3600),76 -(112.5+225)/(abs(cdeltpv)*3600),76 -(112.5+225*2)/(abs(cdeltpv)*3600),76 -(112.5+225*3)/(abs(cdeltpv)*3600),
                            76 + 112.5/(abs(cdeltpv)*3600),76 + (112.5+225)/(abs(cdeltpv)*3600),76 + (112.5+225*2)/(abs(cdeltpv)*3600),76 + (112.5+225*3)/(abs(cdeltpv)*3600)])
                print(rbeam)


                cmap = 'RdBu_r' #plt.get_cmap('RdBu_r',25)
                l = len(realgal_pv[0]); b = len(realgal_pv[:,0])
                LL, BB = np.meshgrid(np.arange(0,l),np.arange(0,b))

                z1 = mom1*maskmap - datavsysmean 
                z2 = modelgal_m1*maskmap- datavsysmean

                L,B = np.meshgrid(np.arange(0,len(mom1[0])),np.arange(0,len(mom1[:,0])))

                ax1 = fig.add_subplot(gs2[0,0:5])
                vmax = (realgal_vel-datavsysmean)
                ticks = np.linspace(-1*MData['vrot'][-1]-10,MData['vrot'][-1]+10,20)
                print(ticks)
                img1 = ax1.imshow(z1, cmap=cmap,interpolation='nearest')
                # add beam
                ax1.plot(TData['xpos'],TData['ypos'],'b+',color='black',markersize=10,mew=5,alpha=.5)
                for xx,yy in zip(beamarr,np.repeat(76,len(parampixelindx))):
                    ax1.add_patch(Circle((xx,yy),radius=rbeam/3,color='black',alpha=0.5))

                cs = ax1.contourf(LL,BB,z1,levels=ticks[::1],cmap=cmap, alpha=0.9)
                ax1.contour(cs, colors='k')
                cbar = fig.colorbar(cs,ax=ax1, ticks=ticks)
                cbar.ax.tick_params(labelsize=20)
                cbar.set_ticklabels(np.round(ticks).astype(int))
                cbar.set_label(r'$V(R)$ [km/s]')
                ax1.axis('off')
                ax1.set_title("DATA [LEFT]")
                ax1.add_patch(Circle((20,130),radius=rbeam,color='black',alpha=0.5))
                ax1.text(20-rbeam,130-rbeam-1,"Beam")

                ax2 = fig.add_subplot(gs2[0,5:])
                ax2.imshow(z2, cmap=cmap,interpolation='nearest')
                # add beam
                ax2.plot(TData['xpos'],TData['ypos'],'b+',color='black',markersize=10,mew=5,alpha=.5)
                for xx,yy in zip(beamarr,np.repeat(76,len(parampixelindx))):
                    ax2.add_patch(Circle((xx,yy),radius=rbeam/3,color='black',alpha=0.5))
                cs = ax2.contourf(LL,BB,z2 ,levels=ticks,cmap=cmap, alpha=0.9)
                ax2.contour(cs, colors='k')
                cbar = fig.colorbar(cs,ax=ax2, ticks=ticks)
                cbar.ax.tick_params(labelsize=20)
                cbar.set_ticklabels(np.round(ticks).astype(int))
                cbar.set_label(r'$V(R)$ [km/s]')
                ax2.axis('off')   
                ax2.set_title("MODEL [RIGHT]")  
                ax2.add_patch(Circle((20,130),radius=rbeam,color='black',alpha=0.5))
                ax2.text(20-rbeam-1,130-rbeam-1,"Beam")

                # Plot channelmap
                matplotlib.rcParams.update({'font.size': 13})
                # Data
                def plot_chan(axch,ch,data):
                    cm = 'coolwarm'
                    sigma = TData['std']
                    axch.set_title('Channelmap',fontsize=10)
                    if galname == 'G10FD' or galname == 'G10FC' or galname == 'G10RC' or galname == 'G10RD' or galname == 'G10':
                        ticks = [r'$-2σ_{noise}$',r'$2σ_{noise}$',r'$4σ_{noise}$',r'$8σ_{noise}$',r'$16σ_{noise}$']
                        cs = axch.contourf(LL,BB,data,levels=[-2*sigma,2*sigma,4*sigma,8*sigma,16*sigma], cmap=cm, alpha=0.9)
                    elif galname[-1:] == '6':
                        ticks = [r'$-2σ_{noise}$',r'$2σ_{noise}$',r'$4σ_{noise}$',r'$8σ_{noise}$']
                        cs = axch.contourf(LL,BB,data,levels=[-2*sigma,2*sigma,4*sigma,8*sigma], cmap=cm, alpha=0.9)
                    elif galname[-1:] == '4':
                        ticks = [r'$-2σ_{noise}$',r'$2σ_{noise}$',r'$4σ_{noise}$']
                        cs = axch.contourf(LL,BB,data,levels=[-2*sigma,2*sigma,4*sigma], cmap=cm, alpha=0.9)
                    axch.imshow(data, cmap=cm)
                    axch.contour(cs, colors='k')
                    axch.axis('off')
                    axch.set_title(r'${}[km/s]$'.format(np.round(realgal_vel[ch],2)),fontsize=19)
                    return cs,ticks

                if galname == 'G10RC' or galname == 'G10RD': channels = [20,25,35,40,45]
                else: channels = [15,25,35,40,50]
                for indx,ch in enumerate(channels):
                    realgal_chan = realgal_unmasked[ch]
                    axch = fig.add_subplot(gs2[1,indx])
                    cs,ticks = plot_chan(axch,ch,realgal_chan)
            
                for indx,ch in enumerate(channels):
                    indx += 5
                    modelgal_chan = modelgal_unmasked[ch]
                    axch = fig.add_subplot(gs2[1,indx])
                    cs,ticks = plot_chan(axch,ch,modelgal_chan)

                cbar = fig.colorbar(cs,orientation='horizontal',cax=fig.add_axes([0.35,.2,.3,.04]))
                cbar.ax.tick_params(labelsize=20)
                cbar.set_ticklabels(ticks)
                plt.tight_layout()
                plt.savefig(path_to_dir + "/" + galname + "_comp1.pdf")
                plt.show()
                plt.close(fig)
                
                
                # Plot body of results comparison
                matplotlib.rcParams.update({'font.size': 15})
                BBData = np.genfromtxt(path_to_dir + '/output0/ringlog1.txt')
                '''Add surface density later on '''
                fig = plt.figure(figsize=(20, 15))
                gs3 = gridspec.GridSpec(nrows=4, ncols=1)

                # Vrot plot
                ax0 = fig.add_subplot(gs3[0,:])
                ax0.plot(TData['radii'],TData['vrot'],'b',label='Real value')
                ax0.errorbar(BBData[:,1],BBData[:,2],yerr=[[abs(item) for indx,item in enumerate(BBData[:,13])],[abs(item) for indx,item in enumerate(BBData[:,14])]],color='red',marker='s', ls='none',capsize=5, capthick=1, ecolor='red', label=r'$^{3D}Barolo$ fit')
                #ax0.scatter(BBData[:,1],BBData[:,2],marker='s',color='red', label=r'$^{3D}Barolo$ fit')
                ax0.errorbar(np.array(MData['radii'])-112.5,MData['vrot'],yerr=[[abs(item[0]-MData['vrot'][indx]) for indx,item in enumerate(MData['sigmavrot'])],[abs(item[1]-MData['vrot'][indx]) for indx,item in enumerate(MData['sigmavrot'])]],color='green',marker='s', ls='none',capsize=5, capthick=2, ecolor='green', label='Nested sampling')
                #ax0.set_xlabel('Radius [arcseconds]')
                ax0.set_ylabel(r'$V(R)$ [km/s]')
                ax0.grid()
                ax0.legend()

                # Vdisp plot
                ax1 = fig.add_subplot(gs3[1,:], sharex=ax0)
                plt.setp(ax0.get_xticklabels(),visible=False)
                ax1.plot(TData['radii'],TData['vdisp'],'b',label='Real value')
                #ax1.scatter(BBData[:,1],BBData[:,3],marker='s',color='red', label=r'$^{3D}Barolo$ fit')
                ax1.errorbar(BBData[:,1],BBData[:,3],yerr=[[abs(item) for indx,item in enumerate(BBData[:,15])],[abs(item) for indx,item in enumerate(BBData[:,16])]],color='red',marker='s', ls='none',capsize=5, capthick=1, ecolor='red', label=r'$^{3D}Barolo$ fit')
                ax1.errorbar(np.array(MData['radii'])-112.5,MData['vdisp'],yerr=[[abs(item[0]-MData['vdisp'][indx]) for indx,item in enumerate(MData['sigmavdisp'])],[abs(item[1]-MData['vdisp'][indx]) for indx,item in enumerate(MData['sigmavdisp'])]],color='green', marker='s', ls='none',capsize=5, capthick=2, ecolor='green', label='Nested sampling')
                #ax1.set_xlabel('Radius [arcseconds]')
                ax1.set_ylabel(r'$\sigma_{gas}$ [km/s]')
                ax1.grid()
                ax1.legend()

                # inclination plot
                ax2 = fig.add_subplot(gs3[2,:], sharex=ax1)
                plt.setp(ax1.get_xticklabels(),visible=False)
                ax2.plot(TData['radii'],np.repeat(TData['inc'],len(TData['radii'])),'b',label='Real value')
                try:
                    ax2.errorbar(np.array(MData['radii'])-112.5,np.repeat(MData['inc'],len(MData['radii'])),yerr=[np.repeat(abs(MData['sigmainc'][0][0]-MData['inc']),len(MData['radii'])),np.repeat(abs(MData['sigmainc'][0][1]-MData['inc']),len(MData['radii']))],color='green',marker='s', ls='none',capsize=5, capthick=2, ecolor='green', label='Nested sampling')
                except:
                    pass
                #ax2.set_xlabel('Radius [arcseconds]')
                ax2.set_ylabel(r'$i$ [degrees]')
                ax2.set_ylim([MData['inc']-5,MData['inc']+5])
                ax2.grid()
                ax2.legend()

                # position angle plot, comment if not relevant
                ax3 = fig.add_subplot(gs3[3,:], sharex=ax2)
                plt.setp(ax2.get_xticklabels(),visible=False)
                ax3.plot(TData['radii'],np.repeat(TData['phi'],len(TData['radii'])),'b',label='Real value')
                try:
                    ax3.errorbar(np.array(MData['radii'])-112.5,np.repeat(MData['phi'],len(MData['radii'])),yerr=[np.repeat(abs(MData['sigmaphi'][0][0]-MData['phi']),len(MData['radii'])),np.repeat(abs(MData['sigmaphi'][0][1]-MData['phi']),len(MData['radii']))],color='green', ls='none', marker='s', capsize=5, capthick=1, ecolor='green', label='Nested sampling')
                except:
                    pass
                ax3.set_ylabel(r'$\phi$ [degrees]')
                ax3.set_xlabel('Radius [arcseconds]')
                ax3.set_ylim([MData['phi']-5,MData['phi']+5])
                ax3.grid()
                ax3.legend()

                plt.subplots_adjust(hspace=.1)
                plt.savefig(path_to_dir + '/' + galname + '_comp2.pdf')
                plt.show()

                # GLOBAL PROFILES #
                ###################
                datavel_emission = [np.sum(realgal_unmasked[i]) for i in range(len(realgal))]
                datavel_emission = datavel_emission/(np.max(datavel_emission))# Normalized
                modelvel_emission = [np.sum(modelgal_unmasked[i]) for i in range(len(modelgal))]
                modelvel_emission = modelvel_emission/(np.max(modelvel_emission))# Normalized

                plt.plot(realgal_vel,datavel_emission, color='blue',label='Data')
                plt.plot(modelgal_vel,modelvel_emission,'-.', color='red', label='Model')
                plt.xlabel(r'$V_{LOS} [km/s]$')
                plt.ylabel('Normalized Intensity')
                plt.title('Observed HI line emission')
                plt.legend()
                plt.savefig(path_to_dir + '/' + galname + '_globalprofile.png')
                plt.show()



def plot_pv(galname,center):
    path_to_dir = os.path.dirname(os.path.abspath(__file__)) + "/Comparison"
    fdatacube = "/final_G10.fits"#"/final_{}.fits".format(galname)
    header = fits.getheader(path_to_dir + fdatacube)
    mywcs = wcs.WCS(header)
    arcsecperpix = header['cdelt1']*3600

    #ra = degperpix * center[0]; dec = degperpix*center[1]
    pix_coords = [(0,76,0), (150,76,0), (76,76,0)]
    world_coords = mywcs.wcs_pix2world(pix_coords, 0)
    #print(world_coords)
    ra,dec = world_coords[0][:2]
    #print(ra,dec)
    cube = SC.read(path_to_dir + fdatacube)
    MData = json.load(open(path_to_dir + '/params_model.json'))
    vel = (cube.spectral_axis * u.s / u.km)
    #print(vel)
    #g = Galactic(30*u.deg, 50*u.deg)
    path = Path([(0,76),(150,76)], width=1) #PathFromCenter(center=g, length=15*u.arcmin, angle=header['bpa']*u.deg, width=100*u.arcsec, sample =100) # beam orientation fixed to semimajor axis orientation!
    slice1 = extract_pv_slice(cube, path)

    # tryout
    mpl.rc('xtick',direction='in') 
    mpl.rc('ytick',direction='in') 
    mpl.rcParams['contour.negative_linestyle'] = 'solid' 
    plt.rc('font',family='sans-serif',serif='Helvetica',size=10)  
    params = {'text.usetex': False, 'mathtext.fontset': 'cm', 'mathtext.default': 'regular', 'errorbar.capsize': 0} 
    plt.rcParams.update(params) 
    crpixpv = header['crpix1']
    cdeltpv = header['cdelt1']
    crvalpv = header['crval1']
    xminpv,xmaxpv = np.floor(crpixpv-1-83), np.ceil(crpixpv-1+83)
    if xminpv == 0: xminpv = 0
    if xmaxpv >= header['naxis1']: xmaxpv=header['naxis1']-1
    zmin = 0
    zmax = len(vel) - 1
    data_maj = slice1.data[zmin:zmax+1,int(xminpv):int(xmaxpv)+1] 
    #xmin_wcs = -1*MData['radii'][-1]#*cdeltpv*3600
    #print(cdeltpv)
    #xmax_wcs = MData['radii'][-1]#*cdeltpv*3600
    xmin_wcs = ((xminpv+1-crpixpv)*cdeltpv+crvalpv)
    xmax_wcs = ((xmaxpv+1-crpixpv)*cdeltpv+crvalpv)

    zmin_wcs, zmax_wcs = 289.611, -24.7135

    rad = np.array(MData['radii'])
    radius = np.concatenate((-rad,rad))
    vrot, incl, vsystem, phi = np.array(MData['vrot']),MData['inc'],MData['vsys'],MData['phi']
    vlos1 = vrot*np.sin(np.deg2rad(incl))+vsystem 
    vlos2 = vsystem-vrot*np.sin(np.deg2rad(incl)) # reverse
    vlos = np.concatenate((vlos1,vlos2)) 
    xcen, ycen= [np.nanmean(MData['xpos'])-0,np.nanmean(MData['ypos'])-0] 
    vsysmean, pamean = np.nanmean(vsystem), np.nanmean(phi) 
    ext = [(114.248-xmin_wcs)*3600,(114.248-xmax_wcs)*3600,zmin_wcs-vsysmean,zmax_wcs-vsysmean]
    print(ext)

    cont = MData['std']
    v = np.array([1,2,4,8,16,32,64])*cont
    labsize=15
    fig,axis=plt.subplots(1,1);
    axis.tick_params(which='major',length=8, labelsize=labsize)
    axis.set_xlabel('Offset (arcsec)',fontsize=labsize+2)
    axis.set_ylabel('$\mathrm{\Delta V_{LOS}}$ (km/s)',fontsize=labsize+2) 
    axis.text(1, 1.02,'$\phi = $90$^\circ$',ha='right',transform=axis.transAxes,fontsize=labsize+4) 
    axis2 = axis.twinx() 
    axis2.set_xlim([ext[0],ext[1]]) 
    axis2.set_ylim([ext[2]+vsysmean,ext[3]+vsysmean]) 
    axis2.tick_params(which='major',length=8, labelsize=labsize) 
    axis2.set_ylabel('$\mathrm{V_{LOS}}$ (km/s)',fontsize=labsize+2) 
    axis.imshow(slice1.data,origin='lower',cmap=mpl.cm.Greys,extent=ext,aspect='auto') 
    axis.contour(slice1.data,v,origin='lower',linewidths=0.7,colors='#00008B',extent=ext) 
    #axis.contour(toplot[0],-cont,origin='lower',linewidths=0.1,colors='gray',extent=ext) 
    axis.axhline(y=0,color='black') 
    axis.axvline(x=0,color='black') 
    axis2.plot(radius,vlos,'yo')
    #plt.savefig(path_to_dir + '/test.png')
    plt.show()
    plt.close()

    '''
    vel = vel - MData['vsys']
    print(vel)
    # Image stuff

    l = len(slice1.data[0]); b = len(slice1.data)
    LL, BB = np.meshgrid(np.arange(0,l),np.arange(0,b))
    cm = 'coolwarm'
    sigma = MData['std']
    incl = MData['inc']
    fig, ax = plt.subplots()
    if galname[-2:] == '10':
        ticks = [r'$-2σ$',0,r'$2σ$',r'$4σ$',r'$8σ$',r'$16σ$']
        levels=[-2*sigma,0,2*sigma,4*sigma,8*sigma,16*sigma]
    elif galname[-1:] == '6':
        ticks = [r'$-2σ$',0,r'$2σ$',r'$4σ$',r'$8σ$']
        levels=[-2*sigma,0,2*sigma,4*sigma,8*sigma]
    elif galname[-1:] == '4':
        ticks = [r'$-2σ$',0,r'$2σ$',r'$4σ$']
        levels=[-2*sigma,0,2*sigma,4*sigma]
    ax.imshow(slice1.data, cmap=cm,extent=[-900,900,289.611,-24.7135], aspect='auto')
    #ax.set_yticks(vel)
    ax.contour(slice1.data,levels, colors='k',extent=[-900,900,289.611,-24.7135], aspect='auto')
    ax.plot(np.array(MData['radii']),np.array(MData['vrot'])*np.sin(np.deg2rad(incl))+MData['vsys'],'ro')
    #plt.axis('off')
    plt.show() 

    #slice1.writeto(path_to_dir + '/pv.fits',overwrite=True)          
    '''

def plot_ngc(galname):
    '''Plot the velocity field by using results obtained with Bbarolo'''
    if os.path.isdir(os.path.dirname(os.path.abspath(__file__)) + "/Comparison") == False:
        os.mkdir(os.path.dirname(os.path.abspath(__file__)) + "/Comparison")
        sys.exit("Put all relevant files in the directory: {}".format(os.path.dirname(os.path.abspath(__file__))+ "/Comparison"))

    # relevant paths
    path_to_dir = os.path.dirname(os.path.abspath(__file__)) + "/Comparison"
    path_to_mask = path_to_dir + "/mask.fits"
    # Load params
    TData = np.genfromtxt(path_to_dir + "/ringlog1.txt")
    dens = np.genfromtxt(path_to_dir + "/densprof.txt")[:,10]
    galaxy1 = path_to_dir + '/' + galname + '.fits'

    vsys = TData[:,11][0]
    radius = TData[:,1]

    # FIX UNITS
    fits.setval(galaxy1, 'CUNIT3', value='km/s')
    with fits.open(galaxy1) as ff:
        with fits.open(path_to_mask) as mk:
            # Refine the data
            realgal = ff[0].data
            maskgal = mk[0].data

            maskmap = np.nansum(maskgal.data,axis=0).astype(np.float)
            maskmap[maskmap>0]  = 1
            maskmap[maskmap==0] = np.nan

            imggal = realgal*maskgal

            realgal = np.array([[[col if maskgal[indx1,indx2,indx3] != 0 else 0 for indx3,col in enumerate(row)] for indx2,row in enumerate(channel)] for indx1,channel in enumerate(realgal)])
            ff[0].data = realgal
            realgal = SC.read(ff)
                    
            maskgal = np.sum(maskgal, axis=0)
            realgal_disp = fits.open(path_to_dir + '/' + galname + "_2mom.fits")[0].data

            realgal_unmasked = realgal.unmasked_data[:,:,:]/realgal.unit

            realgal_vel = realgal.spectral_axis * u.s / u.km

            realgal_m1 = realgal.moment(order=1)
            realgal_m1.write(path_to_dir + '/temp_realgal.fits',overwrite=True)
            realgal_m1 = fits.open(path_to_dir + '/temp_realgal.fits')[0].data

            realgal_pv = np.array([[col-vsys if maskgal[indx1,indx2] != 0 else 0 for indx2,col in enumerate(row)] for indx1,row in enumerate(realgal_m1)])
                        
            # Convert pixels to arcseconds
            header = fits.getheader(os.path.dirname(os.path.abspath(__file__)) + '/Comparison/' + galname + '.fits')
            pixel2arcsec = lambda pixels,xpos: (pixels-xpos)*(header['cdelt']*3600)
            arcsecperpix = header['cdelt1']*3600

    matplotlib.rcParams.update({'font.size': 16})
    # Velocity field plot
    cmap = 'RdBu_r'

    z1 = realgal_pv

    fig, ax1 = plt.subplots()
    ticks = np.linspace(-140,140,15)
    img1 = ax1.imshow(z1,origin='lower', cmap=cmap,extent=[-radius[-1],radius[-1],-radius[-1],radius[-1]],zorder=0)
    # add beam
    cs = ax1.contourf(z1,levels=ticks[::1],cmap=cmap, alpha=0.9,zorder=1,extent=[-radius[-1],radius[-1],-radius[-1],radius[-1]])
    ax1.contour(z1,ticks,colors='k',zorder=2,extent=[-radius[-1],radius[-1],-radius[-1],radius[-1]])
    cbar = fig.colorbar(cs,ax=ax1, ticks=ticks)
    cbar.set_ticklabels(np.round(ticks).astype(int))
    cbar.set_label(r'$V(R)$ [km/s]')
    #ax1.axis('off')
    ax1.set_xlabel('x [arcseconds]')
    ax1.set_ylabel('y [arcseconds]')
    ax1.set_title("Data")
    plt.savefig(path_to_dir + "/Velocityfield.png",dpi=500)
    plt.show()
    plt.close()

    # Dispersion field plot
    cmap = 'coolwarm'

    z1 = realgal_disp

    fig, ax = plt.subplots()
    ticks = np.linspace(0,70,10)
    img1 = ax.imshow(z1,origin='lower', cmap=cmap,extent=[-radius[-1],radius[-1],-radius[-1],radius[-1]],zorder=0)
    # add beam
    cs = ax.contourf(z1,levels=ticks[::1],cmap=cmap, alpha=0.9,zorder=1,extent=[-radius[-1],radius[-1],-radius[-1],radius[-1]])
    cbar = fig.colorbar(cs,ax=ax, ticks=ticks)
    cbar.set_ticklabels(np.round(ticks).astype(int))
    cbar.set_label(r'$\sigma(R)$ [km/s]')
    #ax1.axis('off')
    ax.set_xlabel('x [arcseconds]')
    ax.set_ylabel('y [arcseconds]')
    ax.set_title("Data")
    plt.savefig(path_to_dir + "/dispersionfield.png")
    plt.show()
    plt.close(fig)

    # Channel maps
    matplotlib.rcParams.update({'font.size': 15})
    def plot_chan(galname,axch,sigma,ch,data, vel):
        cm = 'coolwarm'
        ticks = [r'$-2σ$',r'16σ',r'$32σ$',r'$64σ$',r'$128σ$']
        axch.imshow(data, cmap=cm)
        cs = axch.contourf(data,origin='lower',levels=[-2*sigma,16*sigma,32*sigma,64*sigma,128*sigma],cmap=cm)
        axch.contour(data,origin='lower', colors='k')
        axch.axis('off')
        axch.set_title(r'%.2f [km/s]' %(np.round(vel[ch],2)))
        return cs,ticks

    fig = plt.figure(figsize=(10,15))
    gs = fig.add_gridspec(1, 5, wspace=0.01, hspace=0.01)
    sigma = getσ(fits.open(galaxy1)[0].data,50)
    print(sigma)
    vel = realgal_vel
    for indx,ch in enumerate([15,25,35,40,50]):
        img = imggal[ch]#realgal_unmasked[ch]
        axch = fig.add_subplot(gs[0,indx])
        cs,ticks = plot_chan(galname,axch,sigma,ch,img,vel)
            
    cbar = fig.colorbar(cs,orientation='horizontal',cax=fig.add_axes([0.35,.3,.35,.03]))
    cbar.set_ticklabels(ticks)
    plt.tight_layout()
    plt.savefig(path_to_dir + "/channelngc2403.pdf")
    plt.show()
    plt.close(fig)

    # surface density curve
    plt.figure()
    plt.plot(radius,dens/np.max(dens),'k-',color='gray')
    plt.xlabel(r"Radius $[arcseconds]$")
    plt.ylabel(r"$\frac{\Sigma_{HI}}{\Sigma_{HI,max}}$")
    plt.title("Normalized surface density")
    plt.savefig(path_to_dir + "/surfdens.png",dpi=500)
    plt.show()

def plot_densprof():
    matplotlib.rcParams.update({'font.size': 13})
    Data = np.genfromtxt(os.path.dirname(os.path.abspath(__file__)) + "/Comparison/defdensprof.txt")
    defdensprof = Data[:,10]/np.max(Data[:,10])
    defradius = Data[:,0]
    radius = np.linspace(15,885,30)
    RHI_0 =  989
    RHI_1 = 522
    RHI_2 = 781
    custdensprof = lambda radius,RHI: (10*np.exp(-(radius - 0.39*RHI)**2 /(2*(0.35*RHI)**2)))/np.max(10*np.exp(-(radius - 0.39*RHI)**2 /(2*(0.35*RHI)**2)))
    plt.plot(defradius, defdensprof, color='yellow', label=r"Default profile retrieved with $^{3D}Barolo$")
    plt.plot(radius, custdensprof(radius,RHI_0), color='blue', label=r"Custom profile with $M=3\times 10^9 M_{\odot}$")
    plt.plot(radius, custdensprof(radius,RHI_1), color='cyan', label=r"Custom profile with $M=1\times 10^9 M_{\odot}$")
    plt.plot(radius, custdensprof(radius,RHI_2), color='purple', label=r"Custom profile with $M=2\times 10^9 M_{\odot}$")
    plt.fill_between(defradius, defdensprof, alpha=0.2)
    plt.fill_between(radius, custdensprof(radius,RHI_0), alpha=0.2)
    plt.fill_between(radius, custdensprof(radius,RHI_1), alpha=0.2)
    plt.fill_between(radius, custdensprof(radius,RHI_2), alpha=0.2)
    plt.title("Comparison plot")
    plt.xlabel("Radius [arcseconds]")
    plt.ylabel(r"Normalized surface density $\Sigma_{HI}$")
    plt.legend()
    plt.savefig(os.path.dirname(os.path.abspath(__file__)) + "/Comparison" + "/Comparison_densprof.png",dpi=500)
    plt.show()

def plot_rotcurve():
    matplotlib.rcParams.update({'font.size': 13})
    radius = np.linspace(15,885,30)
    flatcurve = lambda radius: 2/np.pi * 120 * np.arctan(radius/70)
    risingcurve = lambda radius: 2/np.pi * 90*(np.arctan(radius/50) + radius/1500)

    plt.plot(radius, flatcurve(radius), color='green',label="Flat rotational curve")
    plt.plot(radius, risingcurve(radius), color='blue',label="Rising rotational curve")
    plt.title("Rotational curves")
    plt.xlabel("Radius [arcseconds]")
    plt.ylabel(r"$V(R) [km/s]$")
    plt.legend()
    plt.savefig(os.path.dirname(os.path.abspath(__file__)) + "/Comparison" + "/Comparison_rotcurve.png",dpi=500)
    plt.show()

def plot_dispcurve():
    matplotlib.rcParams.update({'font.size': 13})
    radius = np.linspace(15,885,30)
    dispcurve = lambda radius: 8*np.exp(-radius/(300)) + 7

    plt.plot(radius, dispcurve(radius), color='green')
    plt.title("Velocity dispersion curve")
    plt.xlabel("Radius [arcseconds]")
    plt.ylabel(r"$\sigma(R) [km/s]$")
    plt.savefig(os.path.dirname(os.path.abspath(__file__)) + "/Comparison" + "/Comparison_dispcurve.png",dpi=500)
    plt.show()

def plot_channelmaps(galname1,galname2,SNR):
    path_to_dir = os.path.dirname(os.path.abspath(__file__)) + "/Comparison"
    # Load params
    Data1 = json.load(open(path_to_dir + '/params_' + galname1 + '.json'))
    Data2 = json.load(open(path_to_dir + '/params_' + galname2 + '.json'))

    sigma1 = Data1['std']
    sigma2 = Data2['std']

    rawgalaxy = path_to_dir + '/raw_' + galname1 + '.fits'
    final1galaxy = path_to_dir + '/final_' + galname1 + '.fits'
    final2galaxy = path_to_dir + '/final_' + galname2 + '.fits'
    mask = path_to_dir + '/mask.fits'

    # FIX UNITS
    with fits.open(rawgalaxy) as ff:
        with fits.open(final1galaxy) as fff:
            with fits.open(final1galaxy) as ffff:
                with fits.open(mask) as mk:
                # Refine the data
                    rawgal = SC.read(ff)
                    final1gal = SC.read(fff)
                    final2gal = SC.read(ffff)
                    maskgal = mk[0].data

                    rawgal_unmasked = rawgal.unmasked_data[:,:,:]/rawgal.unit
                    final1gal_unmasked = final1gal.unmasked_data[:,:,:]/final1gal.unit
                    final2gal_unmasked = final2gal.unmasked_data[:,:,:]/final2gal.unit

                    vel1 = (final1gal.spectral_axis * u.s / u.km)
                    vel2 = (final2gal.spectral_axis * u.s / u.km)
    # Plot channelmap
    def plot_chan(galname,axch,sigma,ch,data, vel):
        cm = 'coolwarm'
        if galname[-2:] == '10':
            ticks = [r'$-2σ$',r'$2σ$',r'$4σ$',r'$8σ$',r'$16σ$']
            cs = axch.contourf(data,levels=[-2*sigma,2*sigma,4*sigma,8*sigma,16*sigma], cmap=cm, alpha=0.9)
        elif galname[-1:] == '6':
            ticks = [r'$-2σ$',r'$2σ$',r'$4σ$',r'$8σ$']
            cs = axch.contourf(data,levels=[-2*sigma,0,2*sigma,4*sigma,8*sigma], cmap=cm, alpha=0.9)
        elif galname[-1:] == '4':
            ticks = [r'$-2σ$',r'$2σ$',r'$4σ$',r'$6σ$']
            cs = axch.contourf(data,levels=[-2*sigma,2*sigma,4*sigma,6*sigma], cmap=cm, alpha=0.9)
        axch.imshow(data, cmap=cm)
        axch.contour(cs, colors='k')
        axch.axis('off')
        axch.set_title('{} [km/s]'.format(np.round(vel[ch],2)))
        return cs,ticks

    def plot_rawchan(galname,axch,sigma,ch,data, vel):
        cm = 'coolwarm'
        if galname[-2:] == '10':
            ticks = [r'$-2σ$',r'$2σ$',r'$4σ$',r'$8σ$',r'$16σ$']
            levels=[-2*sigma,2*sigma,4*sigma,8*sigma,16*sigma]
        elif galname[-1:] == '6':
            ticks = [r'$-2σ$',r'$2σ$',r'$4σ$',r'$8σ$']
            levels=[-2*sigma,2*sigma,4*sigma,8*sigma]
        elif galname[-1:] == '4':
            ticks = [r'$-2σ$',r'$2σ$',r'$4σ$',r'$6σ$']
            levels=[-2*sigma,2*sigma,4*sigma,6*sigma]
        axch.imshow(data, cmap=cm)
        #axch.contour(data,levels=levels, cmap=cm)
        axch.axis('off')
        axch.set_title('{} [km/s]'.format(np.round(vel[ch],2)))



    #fig = plt.figure(figsize=(20,20))
    #gs = fig.add_gridspec(2, 10, wspace=0.01, hspace=0.01)
    output = [path_to_dir + "/noisecomparisonsnr10.pdf",path_to_dir + "/noisecomparisonsnr4.pdf"]
    for i in [0,1]:        
        fig = plt.figure(figsize=(10,10))
        gs = fig.add_gridspec(2, 5)
        if i == 0:
            sigma = sigma1
            galname = galname1
            gal_chan = [rawgal_unmasked,final1gal_unmasked]
            pos = [0.27,0.55,0.5,0.03]
            vel = vel1
            title = 'SNR = 10'
        else:
            sigma = sigma2
            galname = galname2
            gal_chan = [rawgal_unmasked,final2gal_unmasked]
            pos = [0.33,.18,.35,.03]
            vel= vel2
            title = 'SNR = 4'

        for indx,ch in enumerate([15,25,35,40,50]):
            img = gal_chan[0][ch]
            axch = fig.add_subplot(gs[0,indx])
            plot_rawchan(galname,axch,sigma,ch,img,vel)
            
            
        for indx,ch in enumerate([15,25,35,40,50]):
            img = gal_chan[1][ch]
            axch = fig.add_subplot(gs[1,indx])
            cs,ticks = plot_chan(galname,axch,sigma,ch,img,vel)

        #cframe = fig.add_subplot(gs[1,1:4])
        #cframe.set_visible(False)
        #divider = make_axes_locatable(cframe)
        #cax = divider.append_axes("bottom", size="5%", pad=0.7)
        #cax = divider.new_vertical(size="5%", pad=0.7, pack_start=True)
        cax = fig.add_axes([0.35,0.05,0.35,0.03])
        cbar = fig.colorbar(cs,orientation='horizontal',cax=cax)
        cbar.set_ticklabels(ticks)

        if i == 0:
            fig.text(0.5, .95, r'$SNR = 10$',
                {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center',
                'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
        else:
            fig.text(0.5, 0.95, r'$SNR = 4$',
            {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center',
            'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})

        plt.tight_layout()
        plt.savefig(output[i])
        plt.show()
        plt.close(fig)

def plot_diagnostic(title):
    path_to_dir = os.path.dirname(os.path.abspath(__file__)) + "/Comparison"
    # Load params
    nlive = [50,100,150,180,200,220,250,300,350,400,500]
    # Metropolis-Hastings
    MH7_eff = []
    MH7_ncalls = []
    MH7_KL = []
    MH8_eff = []
    MH8_ncalls = []
    MH8_KL = []
    MH9_eff = []
    MH9_ncalls = []
    MH9_KL = []

    # Uniform
    U7_eff = []
    U7_ncalls = []
    U7_KL = []
    U8_eff = []
    U8_ncalls = []
    U8_KL = []
    U9_eff = []
    U9_ncalls = []
    U9_KL = []

    for i in range(3):
        if i == 0:
            for i in nlive:
                Data1 = json.load(open(path_to_dir + '/result_nested_hastings7_P{}'.format(i) + '.json'))
                Data2 = json.load(open(path_to_dir + '/result_nested_uniform7_P{}'.format(i) + '.json'))

                MH7_eff.append(Data1['eff']);U7_eff.append(Data2['eff'])
                MH7_ncalls.append(np.sum(Data1['ncall']));U7_ncalls.append(np.sum(Data2['ncall']))
                MH7_KL.append(Data1['KLval']);U7_KL.append(Data2['KLval'])
        if i == 1:
            for i in nlive:
                Data1 = json.load(open(path_to_dir + '/result_nested_hastings8_P{}'.format(i) + '.json'))
                Data2 = json.load(open(path_to_dir + '/result_nested_uniform8_P{}'.format(i) + '.json'))

                MH8_eff.append(Data1['eff']);U8_eff.append(Data2['eff'])
                MH8_ncalls.append(np.sum(Data1['ncall']));U8_ncalls.append(np.sum(Data2['ncall']))
                MH8_KL.append(Data1['KLval']);U8_KL.append(Data2['KLval'])
        if i == 2:
            for i in nlive:
                Data1 = json.load(open(path_to_dir + '/result_nested_hastings9_P{}'.format(i) + '.json'))
                Data2 = json.load(open(path_to_dir + '/result_nested_uniform9_P{}'.format(i) + '.json'))

                MH9_eff.append(Data1['eff']);U9_eff.append(Data2['eff'])
                MH9_ncalls.append(np.sum(Data1['ncall']));U9_ncalls.append(np.sum(Data2['ncall']))
                MH9_KL.append(Data1['KLval']);U9_KL.append(Data2['KLval'])

    fig, axes = plt.subplots(nrows=2, ncols=3 ,figsize=(15, 10))
    rows = ["Information loss","Effeciency [%]"]
    col = ["Update proposal at 0.7xnlive-th iteration","Update proposal at 0.8xnlive-th iteration","Update proposal at 0.9xnlive-th iteration"]
    fig.suptitle("Nested sampler diagnostic")

    # make the plots
    for i in range(3):
        if i == 0:
            for j in range(2):
                if j== 0:
                    axes[j,i].set_title("Information loss")
                    axes[j,i].set_ylabel("KL-divergence")
                    axes[j,i].set_xlabel("nlive")
                    axes[j,i].plot(nlive,MH7_KL,linestyle='none',marker='o',color='blue',alpha=0.3, label='Metropolis-Hastings')
                    axes[j,i].plot(nlive,U7_KL,linestyle='none',marker='o',color='green',alpha=0.3, label='Uniform')

                if j == 1:
                    axes[j,i].set_title("Computation time")
                    axes[j,i].set_ylabel("efficiency [%]")
                    axes[j,i].set_xlabel("ncalls")
                    axes[j,i].plot(MH7_ncalls,MH7_eff,linestyle='none',marker='o',color='blue',alpha=0.3, label='Metropolis-Hastings')
                    axes[j,i].plot(U7_ncalls,U7_eff,linestyle='none',marker='o',color='green',alpha=0.3, label='Uniform')

                for ii, txt in enumerate(nlive):
                    axes[j,i].annotate(txt, (MH7_ncalls[ii],MH7_eff[ii]))

                for ii, txt in enumerate(nlive):
                    axes[j,i].annotate(txt, (U7_ncalls[ii],U7_eff[ii]))

                axes[j,i].legend()
                axes[j,i].legend()

        if i == 1:
            for j in range(2):
                if j== 0:
                    axes[j,i].set_title("Information loss")
                    axes[j,i].set_ylabel("KL-divergence")
                    axes[j,i].set_xlabel("nlive")
                    axes[j,i].plot(nlive,MH8_KL,linestyle='none',marker='o',color='blue',alpha=0.3, label='Metropolis-Hastings')
                    axes[j,i].plot(nlive,U8_KL,linestyle='none',marker='o',color='green',alpha=0.3, label='Uniform')

                if j == 1:
                    axes[j,i].set_title("Computation time")
                    axes[j,i].set_ylabel("efficiency [%]")
                    axes[j,i].set_xlabel("ncalls")
                    axes[j,i].plot(MH8_ncalls,MH8_eff,linestyle='none',marker='o',color='blue',alpha=0.3, label='Metropolis-Hastings')
                    axes[j,i].plot(U8_ncalls,U8_eff,linestyle='none',marker='o',color='green',alpha=0.3, label='Uniform')

                for ii, txt in enumerate(nlive):
                    axes[j,i].annotate(txt, (MH8_ncalls[ii],MH8_eff[ii]))

                for ii, txt in enumerate(nlive):
                    axes[j,i].annotate(txt, (U8_ncalls[ii],U8_eff[ii]))

                axes[j,i].legend()
                axes[j,i].legend()

        if i == 2:
            for j in range(2):
                if j== 0:
                    axes[j,i].set_title("Information loss")
                    axes[j,i].set_ylabel("KL-divergence")
                    axes[j,i].set_xlabel("nlive")
                    axes[j,i].plot(nlive,MH9_KL,linestyle='none',marker='o',color='blue',alpha=0.3, label='Metropolis-Hastings')
                    axes[j,i].plot(nlive,U9_KL,linestyle='none',marker='o',color='green',alpha=0.3, label='Uniform')

                if j == 1:
                    axes[j,i].set_title("Computation time")
                    axes[j,i].set_ylabel("efficiency [%]")
                    axes[j,i].set_xlabel("ncalls")
                    axes[j,i].plot(MH9_ncalls,MH9_eff,linestyle='none',marker='o',color='blue',alpha=0.3, label='Metropolis-Hastings')
                    axes[j,i].plot(U9_ncalls,U9_eff,linestyle='none',marker='o',color='green',alpha=0.3, label='Uniform')

                for ii, txt in enumerate(nlive):
                    axes[j,i].annotate(txt, (MH9_ncalls[ii],MH9_eff[ii]))

                for ii, txt in enumerate(nlive):
                    axes[j,i].annotate(txt, (U9_ncalls[ii],U9_eff[ii]))

                axes[j,i].legend()
                axes[j,i].legend()
    plt.savefig(path_to_dir + "/diagnostic.png")
    plt.show()


def plot_comp2(galname):
    path_to_dir = os.path.dirname(os.path.abspath(__file__)) + "/Comparison"
    TData = json.load(open(path_to_dir + '/params_' + galname + '.json'))
    MData = json.load(open(path_to_dir + '/params_model.json'))
    # Plot body of results comparison
    matplotlib.rcParams.update({'font.size': 15})
    BBData = np.genfromtxt(path_to_dir +'/output/' + galname +  '/ringlog1.txt')
    ''''Add surface density later on '''
    fig = plt.figure(figsize=(15, 10))
    gs3 = gridspec.GridSpec(nrows=4, ncols=1)
    # Vrot plot
    ax0 = fig.add_subplot(gs3[0,:])
    ax0.plot(TData['radii'],TData['vrot'],'b',label='Real value')
    ax0.errorbar(BBData[:,1],BBData[:,2],yerr=[[abs(i) for i in BBData[:,13]],[abs(j) for j in BBData[:,14]]],color='red',marker='o', ls='none',capsize=5, capthick=1, ecolor='black', label=r'$^{3D}Barolo$ fit')
    ax0.errorbar(MData['radii'],MData['vrot'],yerr=[[abs(item[0]-MData['vrot'][indx]) for indx,item in enumerate(MData['sigmavrot'])],[abs(item[1]-MData['vrot'][indx]) for indx,item in enumerate(MData['sigmavrot'])]],color='green',marker='s', ls='none',capsize=5, capthick=1, ecolor='black', label='Nested sampling')
    #ax0.set_xlabel('Radius [arcseconds]')
    ax0.set_ylabel(r'$V(R)$ [km/s]')
    ax0.grid()
    ax0.legend()

    # Vdisp plot
    ax1 = fig.add_subplot(gs3[1,:], sharex=ax0)
    plt.setp(ax0.get_xticklabels(),visible=False)
    ax1.plot(TData['radii'],TData['vdisp'],'b',label='Real value')
    ax1.errorbar(BBData[:,1],BBData[:,3],yerr=[[abs(i) for i in BBData[:,15]],[abs(j) for j in BBData[:,16]]],color='red',marker='o', ls='none',capsize=5, capthick=1, ecolor='black', label=r'$^{3D}Barolo$ fit')
    ax1.errorbar(MData['radii'],MData['vdisp'],yerr=[[abs(item[0]-MData['vdisp'][indx]) for indx,item in enumerate(MData['sigmavdisp'])],[abs(item[1]-MData['vdisp'][indx]) for indx,item in enumerate(MData['sigmavdisp'])]],color='green', marker='s', ls='none',capsize=5, capthick=1, ecolor='black', label='Nested sampling')
    #ax1.set_xlabel('Radius [arcseconds]')
    ax1.set_ylabel(r'$\sigma_{gas}$ [km/s]')
    ax1.grid()
    ax1.legend()

    # inclination plot
    ax2 = fig.add_subplot(gs3[2,:], sharex=ax1)
    plt.setp(ax1.get_xticklabels(),visible=False)
    ax2.plot(TData['radii'],np.repeat(TData['inc'],len(TData['radii'])),'b',label='Real value')
    try:
        ax2.errorbar(MData['radii'],np.repeat(MData['inc'],len(MData['radii'])),yerr=[np.repeat(abs(MData['sigmainc'][0][0]-MData['inc']),len(MData['radii'])),np.repeat(abs(MData['sigmainc'][0][1]-MData['inc']),len(MData['radii']))],color='green',marker='s', ls='none',capsize=5, capthick=1, ecolor='black', label='Nested sampling')
    except:
        pass
    #ax2.set_xlabel('Radius [arcseconds]')
    ax2.set_ylabel(r'$i$ [degrees]')
    ax2.set_ylim([MData['inc']-5,MData['inc']+5])
    ax2.grid()
    ax2.legend()

    # position angle plot, comment if not relevant
    ax3 = fig.add_subplot(gs3[3,:], sharex=ax2)
    plt.setp(ax2.get_xticklabels(),visible=False)
    ax3.plot(TData['radii'],np.repeat(TData['phi'],len(TData['radii'])),'b',label='Real value')
    try:
        ax3.errorbar(MData['radii'],np.repeat(MData['phi'],len(MData['radii'])),yerr=[np.repeat(abs(MData['sigmaphi'][0][0]-MData['phi']),len(MData['radii'])),np.repeat(abs(MData['sigmaphi'][0][1]-MData['phi']),len(MData['radii']))],color='green', ls='none', marker='s', capsize=5, capthick=1, ecolor='black', label='Nested sampling')
    except:
        pass
    ax3.set_ylabel(r'$\phi$ [degrees]')
    ax3.set_xlabel('Radius [arcseconds]')
    ax3.set_ylim([MData['phi']-5,MData['phi']+5])
    ax3.grid()
    ax3.legend()

    plt.subplots_adjust(hspace=.1)
    plt.savefig(path_to_dir + '/' + galname + '_comp2.pdf')
    plt.show()    

def plot_velfieldngc7793():
    ###########################
    ### Velocity field plot ###
    ###########################
    path_to_dir = os.path.dirname(os.path.abspath(__file__)) + "/Comparison"
    galaxy = fits.open(path_to_dir + "/NGC7793.fits")[0].data
    galaxy = SC.read(path_to_dir + "/NGC7793.fits")
    mom1 = galaxy.moment(order=1)
    cmap = 'RdBu_r' #plt.get_cmap('RdBu_r',25)

    maskgal = fits.open(path_to_dir + "/mask.fits")[0].data
    img_mas = maskgal
    maskmap = np.nansum(img_mas,axis=0).astype(np.float)
    maskmap[maskmap>0]  = 1
    maskmap[maskmap==0] = np.nan

    z1 = (mom1.data*maskmap - 227000)/1000

    LL,BB = np.meshgrid(np.arange(0,len(z1)),np.arange(0,len(z1)))
    matplotlib.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(20, 20))
    gs2 = gridspec.GridSpec(nrows=1, ncols=1)
    ax1 = fig.add_subplot(gs2[0,0:5])
    realgal_vel = galaxy.spectral_axis * u.s / u.km

    ticks = np.linspace(-100,100,14)
    print(ticks)
    img1 = ax1.imshow(z1, cmap=cmap,interpolation='nearest')
    cs = ax1.contourf(LL,BB,z1,levels=ticks[::1],cmap=cmap, alpha=0.9)
    ax1.contour(cs, colors='k')
    cbar = fig.colorbar(cs,ax=ax1, ticks=ticks)
    cbar.set_ticklabels(np.round(ticks).astype(int))
    cbar.set_label(r'$V(R)$ [km/s]')
    ax1.axis('off')
    ax1.set_title("The velocity field of NGC7793")
    plt.savefig(path_to_dir + "/vfieldngc7793.pdf")
    plt.show()
    plt.close()


galaxy1 = os.path.dirname(os.path.abspath(__file__)) + '/Comparison/ngc2403.fits'
galaxy3 = os.path.dirname(os.path.abspath(__file__)) + '/Comparison/NGC7793.fits'
galaxy2 = os.path.dirname(os.path.abspath(__file__)) + '/model_Galaxy_10.fits'
galaxy4 = os.path.dirname(os.path.abspath(__file__)) + '/final_G10FD.fits'
skeletonfile = os.path.dirname(os.path.abspath(__file__)) + '/examples/ngc2403.fits'
mask = os.path.dirname(os.path.abspath(__file__)) + '/mask_Galaxy_10.fits'
save = os.path.dirname(os.path.abspath(__file__)) + '/Comparison/ngc2403'
save2 = os.path.dirname(os.path.abspath(__file__)) + '/Comparison/ngc7793'
save3 = os.path.dirname(os.path.abspath(__file__)) + '/Comparison'
paramfile = os.path.dirname(os.path.abspath(__file__)) + '/params_Galaxy_10.json'
#save2 = os.path.dirname(os.path.abspath(__file__)) + '/raw_moment.png'
angular_plist = ["incl = 50, PA = 90","incl = 60, PA = 90","incl = 50, PA = 120"]

######## PLOTS #############
#plot_global(galaxy4, save=save3)
#plot_moments(file1, save1)
#plot_moments(galaxy3, save2)
#channel_maps(file1, save1)
#channel_maps(file2, save2)
#plot_moments(galaxy1, save)
#plot_ngc('ngc2403')
#plot_densprof()
#plot_rotcurve()
#plot_dispcurve()
#plot_channelmaps("Galaxy_10","Galaxy_4",10)
#plot_diagnostic("Diagnostic")
#plot_pv('G10',(76,76))
plot_comparison('G10FC')
#plot_bbarolo(angular_plist,"",'flat')
#plot_comp2("G10FD")
#plot_velfieldngc7793()