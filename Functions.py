############ IMPORTS #################
import os
import json
import gc
import time
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

from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from multiprocessing import Pool


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            try:
                return obj.tolist()
            except:
                return None
        if isinstance(obj, object):
            return None
        return json.JSONEncoder.default(self, obj)

def quickimage(image):
    '''Used for inspecting region '''
    assert len(image.shape) == 2, 'Only 2D arrays'

    plt.imshow(image)
    plt.show()

def createmodelgal(path,skeletonfile,params):
    # Start model
    model = GM(skeletonfile)
    # Initialize model
    model.init(radii=params['radii'],xpos=params['xpos'],ypos=params['ypos'],vsys=params['vsys'],vrot=params['vrot'], vdisp=params['vdisp'],vrad=params['vrad'],
                    z0=params['z0'],inc=params['inc'],phi=params['phi'],dens=np.array([0.8976,0.983,0.7064,0.3329]))

    model.set_options(ltype=params['ltype'])
        
    # Compute
    mymodel = model.compute()

    temp_gal = SC.read(mymodel)
    beam = rb.Beam(major=params['radii'][-1]*params['beamfactor'][0]*u.arcsec, minor=params['radii'][-1]*params['beamfactor'][1]*u.arcsec, pa=params['phi']*u.deg) # fitted radius must be last one
    temp_gal = temp_gal.convolve_to(beam)
    temp_gal.write(path,overwrite=True)
    
def pickregion(rawfile, beam):
    '''Show the observed image'''
    with fits.open(rawfile) as ff:
        realcube_data = ff[0].data
        row = realcube_data[0].shape[0] 
        width = realcube_data[0].shape[1]

        # Add the noise        
        for ch in range(len(realcube_data)):
            if np.max(realcube_data[ch]) == 0:
                sigma = 0.05 # doesnt matter for now
                realcube_data[ch] += np.random.normal(0,sigma,(row,width))#np.random.choice(np.array([-3,-2,-1,0,1,2,3])*sigma, (row,width), p=[1/7,1/7,1/7,1/7,1/7,1/7,1/7]) # np.random.normal(0,sigma,(row,width))
            else:
                sigma = 0.05 # doesnt matter for now #mean_ch/self.signal_to_noise #(flux_mean/self.signal_to_noise)/pixels_w_signal
                realcube_data[ch] += np.random.normal(0,sigma,(row,width))#np.random.choice(np.array([-3,-2,-1,0,1,2,3])*sigma, (row,width), p=[1/7,1/7,1/7,1/7,1/7,1/7,1/7]) #np.random.normal(0,sigma,(row,width))
    
        ff[0].data = realcube_data
        cube = SC.read(ff)
    cube = cube.convolve_to(beam)
    mdata = cube.hdu.data

    plt.imshow(mdata[40])
    plt.show()


def pixelcount(maskfile):
    '''Count amount of pixels with signal '''
    with fits.open(maskfile) as ff:
        cube = ff[0].data

    Npixels = np.array([np.sum(cube[ch]) for ch in range(len(cube))])
    Nch = np.array([ch for ch in range(len(Npixels)) if Npixels[ch] != 0])
    return (Npixels,Nch)

def getSNR(region,noise_amplifier,SNR_user,rawfile, beam, noisefile):
    '''Used to estimate SNR '''
    # Tweak the meanflux to take into account the convolution
    with fits.open(rawfile) as ff:
        realcube_data = ff[0].data

        meanflux = np.mean(realcube_data[40][region[0]:region[1],region[2]:region[3]])
        row = realcube_data[0].shape[0] 
        width = realcube_data[0].shape[1]

        # Add the noise        
        for ch in range(len(realcube_data)):
            if np.max(realcube_data[ch]) == 0:
                sigma = meanflux/SNR_user * noise_amplifier
                realcube_data[ch] += np.random.normal(0,sigma,(row,width))#np.random.choice(np.array([-3,-2,-1,0,1,2,3])*sigma, (row,width), p=[1/7,1/7,1/7,1/7,1/7,1/7,1/7]) # np.random.normal(0,sigma,(row,width))
            else:
                #pixels_w_signal = len(realcube_data[ch].flatten()) - len(np.where(realcube_data[ch] == 0)[0])
                #mean_ch = np.max(realcube_data[ch][np.where(realcube_data[ch] != 0)])
                sigma = meanflux/SNR_user * noise_amplifier #mean_ch/self.signal_to_noise #(flux_mean/self.signal_to_noise)/pixels_w_signal
                realcube_data[ch] += np.random.normal(0,sigma,(row,width))#np.random.choice(np.array([-3,-2,-1,0,1,2,3])*sigma, (row,width), p=[1/7,1/7,1/7,1/7,1/7,1/7,1/7]) #np.random.normal(0,sigma,(row,width))
    
        ff[0].data = realcube_data
        cube = SC.read(ff)

    # convolve cube
    cube = cube.convolve_to(beam)
    # Write noise to FITS
    cube.write(noisefile,overwrite=True)
    mdata = cube.hdu.data

    meanflux = np.mean(mdata[40][region[0]:region[1],region[2]:region[3]])
    noise_img = np.std(mdata[40][:30,:].flatten())
    SNR = meanflux/noise_img
    #os.path.dirname(os.path.abspath(__file__))
    return (SNR,noise_img)

def getσ(data,ch):
    '''Gives the uncertainty of an image '''
    noise_img = np.std(data[40][:30,:].flatten())
    return noise_img

def getμ(data, region, ch):
    '''Gives the mean signal of an image '''
    μflux = np.mean(data[40][region[0]:region[1],region[2]:region[3]])
    return μflux

def beampixels(galname):
    TData = json.load(open(os.path.dirname(os.path.abspath(__file__)) + '/params_' + galname + '.json'))

    # Load beam size
    Bmaj = TData['beamfactor'][0] * TData['radii'][-1]
    Bmin = TData['beamfactor'][1] * TData['radii'][-1]

    # Convert pixels to WCS
    header = fits.getheader(os.path.dirname(os.path.abspath(__file__)) + '/final_' + galname + '.fits')
    mywcs = wcs.WCS(header)

    pix_coords = [(20,30,0), (21,30,0), (20,31,0)]
    world_coords = mywcs.wcs_pix2world(pix_coords, 0)

    dra = abs(world_coords[1][1] - world_coords[2][1])*3600 # In arcseconds
    ddec = abs(world_coords[0][0] - world_coords[1][0])*3600 # In arcseconds

    # Compute surface areas
    Surface_area_beam = np.pi*Bmaj*Bmin
    Surface_area_pixel = dra*ddec

    # Compute amount of pixels in beam
    Npixels_beam = Surface_area_beam/Surface_area_pixel # round?
    return Npixels_beam

def runMCMC(path,ndim,p,loglike,ptform,galname,**pdict):
    pdict = pdict['pdict']
    start = time.time(); pdict['start'] = start

    if ndim == 8:
        nparams='_8P'

        sampler = NestedSampler(loglike, ptform, ndim=ndim, nlive=250,sample='unif',bound='multi', logl_kwargs=pdict,update_interval=0.8,dlogz=0.5 ,first_update={'min_ncall': 300, 'min_eff': 50.}, pool=p)
        sampler.run_nested(maxiter=15000, maxcall=50000) 
        res1 = sampler.results

        with open(path + '/result_nested_P' + '{}'.format(ndim) + '.json','w') as ff:
            ff.write(json.dumps(res1,cls=NumpyEncoder))


        lnz_truth = 10 * -np.log(2 * 30.) 
        fig, axes = dyplot.runplot(res1, lnz_truth=lnz_truth)  
        plt.savefig(path + '/runplot_' + galname + nparams + '.png')
        plt.close() 

        fig, axes = dyplot.traceplot(res1, truths=np.array([pdict['vrot'][0],pdict['vrot'][1],pdict['vrot'][2],pdict['vrot'][3],pdict['vdisp'][0],pdict['vdisp'][1],pdict['vdisp'][2],pdict['vdisp'][3]]), 
                                    truth_color='black', show_titles=True,
                                    trace_cmap='viridis', connect=True,smooth=0.02,
                                    connect_highlight=range(8), labels=[r'$v_{rot,225}$',r'$v_{rot,450}$',r'$v_{rot,675}$',r'$v_{rot,900}$',r'$\sigma_{225}$',r'$\sigma_{450}$',r'$\sigma_{675}$',r'$\sigma_{900}$'])

        plt.savefig(path + '/traceplot_' + galname + nparams + '.png')
        plt.close()

        
        # plot 6 snapshots over the course of the run
        for i, a in enumerate(axes.flatten()):
            it = int((i+1)*res1.niter/8.)
            # overplot the result onto each subplot
            temp = dyplot.boundplot(res1, dims=(0, 1), it=it,
                                    prior_transform=ptform,
                                    max_n_ticks=5, show_live=True,
                                    span=[(70, 150), (70, 150)],
                                    fig=(fig, a))
            a.set_title('Iteration {0}'.format(it), fontsize=26)
        fig.tight_layout()
        plt.savefig(path + '/boundplot_' + galname + nparams + '.png')
        plt.close()

        matplotlib.rcParams.update({'font.size': 16})
        fig,axes = dyplot.cornerplot(res1, color='blue',
                            truths=np.array([pdict['vrot'][0],pdict['vrot'][1],pdict['vrot'][2],pdict['vrot'][3],pdict['vdisp'][0],pdict['vdisp'][1],pdict['vdisp'][2],pdict['vdisp'][3],pdict['inc'],pdict['phi']]),
                            truth_color='black', show_titles=True,smooth=0.02,
                            max_n_ticks=5, quantiles=[0.16,0.5,0.84], labels=[r'$V_{225}[km/s]$',r'$V_{450}[km/s]$',r'$V_{675}[km/s]$',r'$V_{900}[km/s]$',r'$\sigma_{gas,225}[km/s]$',r'$\sigma_{gas,450}[km/s]$',r'$\sigma_{gas,675}[km/s]$',r'$\sigma_{gas,900}[km/s]$',r'$i[deg]$',r'$\phi[deg]$'])

        # Save the model data
        samples, weights = res1.samples, np.exp(res1.logwt - res1.logz[-1])
        mean, cov = dyfunc.mean_and_cov(samples, weights)
        MaP = res1['samples'][res1['logl'].tolist().index(max(res1['logl'].tolist()))] 
        quantiles = [dyfunc.quantile(samps, [0.16,0.5,0.84], weights=weights) for samps in samples.T]
        labels=[r'$V_{225}$',r'$V_{450}$',r'$V_{675}$',r'$V_{900}$',r'$\sigma_{gas,225}$',r'$\sigma_{gas,450}$',r'$\sigma_{gas,675}$',r'$\sigma_{gas,900}$',r'$i$',r'$\phi$']
        units = [' [km/s]',' [km/s]',' [km/s]',' [km/s]',' [km/s]',' [km/s]',' [km/s]',' [km/s]',' [deg]',' [deg]']
        for i in range(ndim):
            ax = axes[i, i]
            q5 = np.round(quantiles[i][1],2)
            q14 = np.round(quantiles[i][0],2)
            q84 = np.round(quantiles[i][2],2)
            ax.set_title(r"$%.2f_{%.2f}^{+%.2f}$"%(q5,-1*abs(q5-q14),abs(q5-q84)) + units[i])

        # Loop over the histograms
        for yi in range(ndim):
            axes[yi,0].set_ylabel(labels[yi]+units[yi],labelpad=30,fontsize=20)
            axes[-1,yi].set_xlabel(labels[yi]+units[yi],labelpad=30,fontsize=20)
            axes[yi,0].tick_params(axis='y', which='major', labelsize=14)
            axes[-1,yi].tick_params(axis='x', which='major', labelsize=14)

        
        fig.tight_layout()
        plt.savefig(path + '/cornerplot_' + galname + nparams + '.pdf')
        plt.close()

        with open(path + '/' + galname + '.txt','w+') as f:
            f.write('Running took: {} hours'.format((time.time() - start)/3600))
    elif ndim == 9:
        nparams='_9P'
        
        sampler = NestedSampler(loglike, ptform, ndim=ndim, nlive=250,sample='unif',bound='multi',logl_kwargs=pdict,update_interval=0.8,dlogz=0.5 ,first_update={'min_ncall': 300, 'min_eff': 50.}, pool=p)
        sampler.run_nested(maxiter=15000, maxcall=50000) 
        res1 = sampler.results

        with open(path + '/result_nested_P' + '{}'.format(ndim) + '.json','w') as ff:
            ff.write(json.dumps(res1,cls=NumpyEncoder))
        
        lnz_truth = 10 * -np.log(2 * 30.) 
        fig, axes = dyplot.runplot(res1, lnz_truth=lnz_truth)  
        plt.savefig(path + '/runplot_' + galname + nparams + '.png')
        plt.close() 

        fig, axes = dyplot.traceplot(res1, truths=np.array([pdict['vrot'][0],pdict['vrot'][1],pdict['vrot'][2],pdict['vrot'][3],pdict['vdisp'][0],pdict['vdisp'][1],pdict['vdisp'][2],pdict['vdisp'][3],pdict['inc']]), 
                                    truth_color='black', show_titles=True,
                                    trace_cmap='viridis', connect=True,smooth=0.02,
                                    connect_highlight=range(8), labels=[r'$v_{rot,225}$',r'$v_{rot,450}$',r'$v_{rot,675}$',r'$v_{rot,900}$',r'$\sigma_{225}$',r'$\sigma_{450}$',r'$\sigma_{675}$',r'$\sigma_{900}$',r'$i$'])

        plt.savefig(path + '/traceplot_' + galname + nparams + '.png')
        plt.close()
        # initialize figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # plot 6 snapshots over the course of the run
        for i, a in enumerate(axes.flatten()):
            it = int((i+1)*res1.niter/8.)
            # overplot the result onto each subplot
            temp = dyplot.boundplot(res1, dims=(0, 1), it=it,
                                    prior_transform=ptform,
                                    max_n_ticks=3, show_live=True,
                                    span=[(70, 150), (70, 150)],
                                    fig=(fig, a))
            a.set_title('Iteration {0}'.format(it), fontsize=26)
        fig.tight_layout()
        plt.savefig(path + '/boundplot_' + galname + nparams + '.png')
        plt.close()

        matplotlib.rcParams.update({'font.size': 16})
        fig,axes = dyplot.cornerplot(res1, color='blue',
                            truths=np.array([pdict['vrot'][0],pdict['vrot'][1],pdict['vrot'][2],pdict['vrot'][3],pdict['vdisp'][0],pdict['vdisp'][1],pdict['vdisp'][2],pdict['vdisp'][3],pdict['inc']]),
                            truth_color='black', show_titles=True,smooth=0.02,
                            max_n_ticks=5, quantiles=[0.16,0.5,0.84], labels=[r'$V_{225}[km/s]$',r'$V_{450}[km/s]$',r'$V_{675}[km/s]$',r'$V_{900}[km/s]$',r'$\sigma_{gas,225}[km/s]$',r'$\sigma_{gas,450}[km/s]$',r'$\sigma_{gas,675}[km/s]$',r'$\sigma_{gas,900}[km/s]$',r'$i[deg]$'])

        # Save the model data
        samples, weights = res1.samples, np.exp(res1.logwt - res1.logz[-1])
        mean, cov = dyfunc.mean_and_cov(samples, weights)
        MaP = res1['samples'][res1['logl'].tolist().index(max(res1['logl'].tolist()))] 
        quantiles = [dyfunc.quantile(samps, [0.16,0.5,0.84], weights=weights) for samps in samples.T]
        labels=[r'$V_{225}$',r'$V_{450}$',r'$V_{675}$',r'$V_{900}$',r'$\sigma_{gas,225}$',r'$\sigma_{gas,450}$',r'$\sigma_{gas,675}$',r'$\sigma_{gas,900}$',r'$i$',r'$\phi$']
        units = [' [km/s]',' [km/s]',' [km/s]',' [km/s]',' [km/s]',' [km/s]',' [km/s]',' [km/s]',' [deg]',' [deg]']
        for i in range(ndim):
            ax = axes[i, i]
            q5 = np.round(quantiles[i][1],2)
            q14 = np.round(quantiles[i][0],2)
            q84 = np.round(quantiles[i][2],2)
            ax.set_title(r"$%.2f_{%.2f}^{+%.2f}$"%(q5,-1*abs(q5-q14),abs(q5-q84)) + units[i])

        # Loop over the histograms
        for yi in range(ndim):
            axes[yi,0].set_ylabel(labels[yi]+units[yi],labelpad=30,fontsize=20)
            axes[-1,yi].set_xlabel(labels[yi]+units[yi],labelpad=30,fontsize=20)
            axes[yi,0].tick_params(axis='y', which='major', labelsize=14)
            axes[-1,yi].tick_params(axis='x', which='major', labelsize=14)

        
        fig.tight_layout()
        plt.savefig(path + '/cornerplot_' + galname + nparams + '.pdf')
        plt.close()

        with open(path + '/' + galname + '.txt','w+') as f:
            f.write('Running took: {} hours'.format((time.time() - start)/3600))

    elif ndim == 10:
        nparams='_10P'
        
        sampler = NestedSampler(loglike, ptform, ndim=ndim, nlive=250,sample='unif',bound='multi' ,logl_kwargs=pdict,update_interval=.8,dlogz=0.5 ,first_update={'min_ncall': 300, 'min_eff': 50.}, pool=p)
        sampler.run_nested(maxiter=15000, maxcall=50000) 
        res1 = sampler.results

        with open(path + '/result_nested_P' + '{}'.format(ndim) + '.json','w') as ff:
            ff.write(json.dumps(res1,cls=NumpyEncoder))

        lnz_truth = 10 * -np.log(2 * 30.) 
        fig, axes = dyplot.runplot(res1, lnz_truth=lnz_truth)  
        plt.savefig(path + '/runplot_' + galname + nparams + '.png')
        plt.close() 

        fig, axes = dyplot.traceplot(res1, truths=np.array([pdict['vrot'][0],pdict['vrot'][1],pdict['vrot'][2],pdict['vrot'][3],pdict['vdisp'][0],pdict['vdisp'][1],pdict['vdisp'][2],pdict['vdisp'][3],pdict['inc'],pdict['phi']]), 
                                    truth_color='black', show_titles=True,
                                    trace_cmap='viridis', connect=True,smooth=0.02,
                                    connect_highlight=range(8), labels=[r'$v_{rot,225}$',r'$v_{rot,450}$',r'$v_{rot,675}$',r'$v_{rot,900}$',r'$\sigma_{225}$',r'$\sigma_{450}$',r'$\sigma_{675}$',r'$\sigma_{900}$',r'$i$',r'$\phi$'])

        plt.savefig(path + '/traceplot_' + galname + nparams + '.png')
        plt.close()

        # initialize figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # plot 6 snapshots over the course of the run
        for i, a in enumerate(axes.flatten()):
            it = int((i+1)*res1.niter/8.)
            # overplot the result onto each subplot
            temp = dyplot.boundplot(res1, dims=(0, 1), it=it,
                                    prior_transform=ptform,
                                    max_n_ticks=3, show_live=True,
                                    span=[(70, 150), (70, 150)],
                                    fig=(fig, a))
            a.set_title('Iteration {0}'.format(it), fontsize=26)
        fig.tight_layout()
        plt.savefig(path + '/boundplot_' + galname + nparams + '.png')
        plt.close()

        matplotlib.rcParams.update({'font.size': 16})
        fig,axes = dyplot.cornerplot(res1, color='blue',
                            truths=np.array([pdict['vrot'][0],pdict['vrot'][1],pdict['vrot'][2],pdict['vrot'][3],pdict['vdisp'][0],pdict['vdisp'][1],pdict['vdisp'][2],pdict['vdisp'][3],pdict['inc'],pdict['phi']]),
                            truth_color='black', show_titles=True,smooth=0.02,
                            max_n_ticks=5, quantiles=[0.16,0.5,0.84], labels=[r'$V_{225}[km/s]$',r'$V_{450}[km/s]$',r'$V_{675}[km/s]$',r'$V_{900}[km/s]$',r'$\sigma_{gas,225}[km/s]$',r'$\sigma_{gas,450}[km/s]$',r'$\sigma_{gas,675}[km/s]$',r'$\sigma_{gas,900}[km/s]$',r'$i[deg]$',r'$\phi[deg]$'])

        # Save the model data
        samples, weights = res1.samples, np.exp(res1.logwt - res1.logz[-1])
        mean, cov = dyfunc.mean_and_cov(samples, weights)
        MaP = res1['samples'][res1['logl'].tolist().index(max(res1['logl'].tolist()))] 
        quantiles = [dyfunc.quantile(samps, [0.16,0.5,0.84], weights=weights) for samps in samples.T]
        labels=[r'$V_{225}$',r'$V_{450}$',r'$V_{675}$',r'$V_{900}$',r'$\sigma_{gas,225}$',r'$\sigma_{gas,450}$',r'$\sigma_{gas,675}$',r'$\sigma_{gas,900}$',r'$i$',r'$\phi$']
        units = [' [km/s]',' [km/s]',' [km/s]',' [km/s]',' [km/s]',' [km/s]',' [km/s]',' [km/s]',' [deg]',' [deg]']
        for i in range(ndim):
            ax = axes[i, i]
            q5 = np.round(quantiles[i][1],2)
            q14 = np.round(quantiles[i][0],2)
            q84 = np.round(quantiles[i][2],2)
            ax.set_title(r"$%.2f_{%.2f}^{+%.2f}$"%(q5,-1*abs(q5-q14),abs(q5-q84)) + units[i])

        # Loop over the histograms
        for yi in range(ndim):
            axes[yi,0].set_ylabel(labels[yi]+units[yi],labelpad=30,fontsize=20)
            axes[-1,yi].set_xlabel(labels[yi]+units[yi],labelpad=30,fontsize=20)
            axes[yi,0].tick_params(axis='y', which='major', labelsize=14)
            axes[-1,yi].tick_params(axis='x', which='major', labelsize=14)

        
        fig.tight_layout()
        plt.savefig(path + '/cornerplot_' + galname + nparams + '.pdf')
        plt.close()


        with open(path + '/' + galname + '.txt','w+') as f:
            f.write('Running took: {} hours'.format((time.time() - start)/3600))

   # Save the model data
    samples, weights = res1.samples, np.exp(res1.logwt - res1.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    MaP = res1['samples'][res1['logl'].tolist().index(max(res1['logl'].tolist()))] 
    quantiles = [dyfunc.quantile(samps, [0.16,0.5,0.84], weights=weights) for samps in samples.T]

    pdict['sigmavrot'] = [(quantiles[0][0],quantiles[0][2]),(quantiles[1][0],quantiles[1][2]),(quantiles[2][0],quantiles[2][2]),(quantiles[3][0],quantiles[3][2])]
    pdict['sigmavdisp'] = [(quantiles[4][0],quantiles[4][2]),(quantiles[5][0],quantiles[5][2]),(quantiles[6][0],quantiles[6][2]),(quantiles[7][0],quantiles[7][2])]
    pdict['vrot'] = [quantiles[0][1],quantiles[1][1],quantiles[2][1],quantiles[3][1]]
    pdict['vdisp'] = [quantiles[4][1],quantiles[5][1],quantiles[6][1],quantiles[7][1]]

    if len(quantiles) == 9:
        pdict['inc'] = quantiles[8][1]
        pdict['sigmainc'] = [(quantiles[8][0],quantiles[8][2])]

    if len(quantiles) == 10:
        pdict['inc'] = quantiles[8][1]
        pdict['sigmainc'] = [(quantiles[8][0],quantiles[8][2])]
        pdict['phi'] = quantiles[9][1]
        pdict['sigmaphi'] = [(quantiles[9][0],quantiles[9][2])]

    # We don't need data entry, waste of space
    pdict['Data'] = None
    with open(path + '/params_model.json','w') as f:
        f.write(json.dumps(pdict,cls=NumpyEncoder))
def MCMC_diagnostic(path,ndim,p,loglike,ptform,galname,nlive,**pdict):
    pdict = pdict['pdict']
    start = time.time(); pdict['start'] = start

    if ndim == 10:
        nparams='_10P'

        sampler = NestedSampler(loglike, ptform, ndim=ndim, nlive=nlive,sample='unif',bound='multi', logl_kwargs=pdict,update_interval=.8,dlogz=0.5 ,first_update={'min_ncall': nlive, 'min_eff': 50.}, pool=p)
        sampler.run_nested(maxiter=15000, maxcall=50000) 
        res1 = sampler.results

        # Save nested data
        # obtain KL divergence
        klds = []
        for i in range(500):
            kld = dyfunc.kld_error(res1, error='simulate')
            klds.append(kld[-1])
        print(np.mean(klds))
        res1['KLval'] = np.mean(klds)
        with open(path + '/result_nested_P' + '{}'.format(nlive) + '.json','w') as ff:
            ff.write(json.dumps(res1,cls=NumpyEncoder))


        lnz_truth = 10 * -np.log(2 * 30.) 
        fig, axes = dyplot.runplot(res1, lnz_truth=lnz_truth)  
        plt.savefig(path + '/runplot_' + galname + nparams + '.png')
        plt.close() 

        fig, axes = dyplot.traceplot(res1, truths=np.array([pdict['vrot'][0],pdict['vrot'][1],pdict['vrot'][2],pdict['vrot'][3],pdict['vdisp'][0],pdict['vdisp'][1],pdict['vdisp'][2],pdict['vdisp'][3],pdict['inc'],pdict['phi']]), 
                                    truth_color='black', show_titles=True,
                                    trace_cmap='viridis', connect=True,smooth=0.02,
                                    connect_highlight=range(8), labels=[r'$v_{rot,225}$',r'$v_{rot,450}$',r'$v_{rot,675}$',r'$v_{rot,900}$',r'$\sigma_{225}$',r'$\sigma_{450}$',r'$\sigma_{675}$',r'$\sigma_{900}$',r'$i$',r'$\phi$'])

        plt.savefig(path + '/traceplot_' + galname + nparams + '.png')
        plt.close()
        # initialize figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # plot 6 snapshots over the course of the run
        for i, a in enumerate(axes.flatten()):
            it = int((i+1)*res1.niter/8.)
            # overplot the result onto each subplot
            temp = dyplot.boundplot(res1, dims=(0, 1), it=it,
                                    prior_transform=ptform,
                                    max_n_ticks=3, show_live=True,
                                    span=[(70, 150), (70, 150)],
                                    fig=(fig, a))
            a.set_title('Iteration {0}'.format(it), fontsize=26)
        fig.tight_layout()
        plt.savefig(path + '/boundplot_' + galname + nparams + '.png')
        plt.close()

        fg, ax = dyplot.cornerplot(res1, color='blue', truths=np.array([pdict['vrot'][0],pdict['vrot'][1],pdict['vrot'][2],pdict['vrot'][3],pdict['vdisp'][0],pdict['vdisp'][1],pdict['vdisp'][2],pdict['vdisp'][3],pdict['inc'],pdict['phi']]), # 91.8,98.3,8.88,6.5,60,60
                                truth_color='black', show_titles=True,smooth=0.02,
                                max_n_ticks=5, quantiles=None, labels=[r'$v_{rot,225}$',r'$v_{rot,450}$',r'$v_{rot,675}$',r'$v_{rot,900}$',r'$\sigma_{225}$',r'$\sigma_{450}$',r'$\sigma_{675}$',r'$\sigma_{900}$',r'$i$',r'$\phi$'])

        plt.savefig(path + '/cornerplot_' + galname + nparams + '.png')
        plt.close()

        with open(path + '/' + galname + '.txt','w+') as f:
            f.write('Running took: {} hours'.format((time.time() - start)/3600))

    # Save the model data
    samples, weights = res1.samples, np.exp(res1.logwt - res1.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    MaP = res1['samples'][res1['logl'].tolist().index(max(res1['logl'].tolist()))]  
    quantiles = [dyfunc.quantile(samps, [0.025,0.5,0.975], weights=weights) for samps in samples.T]
    print(quantiles)

    # vrotsigma
    sigmavrot1_l = [i for i in samples[:,0] if (i - MaP[0])<0]; sigmavrot1_r = [i for i in samples[:,0] if (i - MaP[0])>0]
    sigmavrot2_l = [i for i in samples[:,1] if (i - MaP[1])<0]; sigmavrot2_r = [i for i in samples[:,1] if (i - MaP[1])>0]
    sigmavrot3_l = [i for i in samples[:,2] if (i - MaP[2])<0]; sigmavrot3_r = [i for i in samples[:,2] if (i - MaP[2])>0]
    sigmavrot4_l = [i for i in samples[:,3] if (i - MaP[3])<0]; sigmavrot4_r = [i for i in samples[:,3] if (i - MaP[3])>0]

    if len(sigmavrot1_l) == 0: sigmavrot1_l.append(0);
    if len(sigmavrot1_r) == 0: sigmavrot1_r.append(0);
    if len(sigmavrot2_l) == 0: sigmavrot2_l.append(0);
    if len(sigmavrot2_r) == 0: sigmavrot2_r.append(0);
    if len(sigmavrot3_l) == 0: sigmavrot3_l.append(0);
    if len(sigmavrot3_r) == 0: sigmavrot3_r.append(0);
    if len(sigmavrot4_l) == 0: sigmavrot4_l.append(0);
    if len(sigmavrot4_r) == 0: sigmavrot4_r.append(0);

    # vdispsigma
    sigmavdisp1_l = [i for i in samples[:,4] if (i - MaP[4])<0]; sigmavdisp1_r = [i for i in samples[:,4] if (i - MaP[4])>0]
    sigmavdisp2_l = [i for i in samples[:,5] if (i - MaP[5])<0]; sigmavdisp2_r = [i for i in samples[:,5] if (i - MaP[5])>0]
    sigmavdisp3_l = [i for i in samples[:,6] if (i - MaP[6])<0]; sigmavdisp3_r = [i for i in samples[:,6] if (i - MaP[6])>0]
    sigmavdisp4_l = [i for i in samples[:,7] if (i - MaP[7])<0]; sigmavdisp4_r = [i for i in samples[:,7] if (i - MaP[7])>0]

    if len(sigmavdisp1_l) == 0: sigmavdisp1_l.append(0);
    if len(sigmavdisp1_r) == 0: sigmavdisp1_r.append(0);
    if len(sigmavdisp2_l) == 0: sigmavdisp2_l.append(0);
    if len(sigmavdisp2_r) == 0: sigmavdisp2_r.append(0);
    if len(sigmavdisp3_l) == 0: sigmavdisp3_l.append(0);
    if len(sigmavdisp3_r) == 0: sigmavdisp3_r.append(0);
    if len(sigmavdisp4_l) == 0: sigmavdisp4_l.append(0);
    if len(sigmavdisp4_r) == 0: sigmavdisp4_r.append(0);

    pdict['sigmavrot'] = [(np.std(sigmavrot1_l),np.std(sigmavrot1_r)),(np.std(sigmavrot2_l),np.std(sigmavrot2_r)),(np.std(sigmavrot3_l),np.std(sigmavrot3_r)),(np.std(sigmavrot4_l),np.std(sigmavrot4_r))]
    pdict['sigmavdisp'] = [(np.std(sigmavdisp1_l),np.std(sigmavdisp1_r)),(np.std(sigmavdisp2_l),np.std(sigmavdisp2_r)),(np.std(sigmavdisp3_l),np.std(sigmavdisp3_r)),(np.std(sigmavdisp4_l),np.std(sigmavdisp4_r))]

    if len(MaP) == 8:
        pdict['vrot'] = MaP[0:4]
        pdict['vdisp'] = MaP[4:8]

    if len(MaP) == 9:
        pdict['vrot'] = MaP[0:4]
        pdict['vdisp'] = MaP[4:8]
        pdict['inc'] = MaP[8]
        # inc
        sigmainc_l = [i for i in samples[:,8] if (i - MaP[8])<0]; sigmainc_r = [i for i in samples[:,8] if (i - MaP[8])>0]
        if len(sigmainc_l) == 0: sigmainc_l.append(0);
        if len(sigmainc_r) == 0: sigmainc_r.append(0);
        pdict['sigmainc'] = [(np.std(sigmainc_l),np.std(sigmainc_r))]

    if len(MaP) == 10:
        pdict['vrot'] = MaP[0:4]
        pdict['vdisp'] = MaP[4:8]
        pdict['inc'] = MaP[8]
        pdict['phi'] = MaP[9]

        # inc
        sigmainc_l = [i for i in samples[:,8] if (i - MaP[8])<0]; sigmainc_r = [i for i in samples[:,8] if (i - MaP[8])>0]
        if len(sigmainc_l) == 0: sigmainc_l.append(0);
        if len(sigmainc_r) == 0: sigmainc_r.append(0);
        pdict['sigmainc'] = [(np.std(sigmainc_l),np.std(sigmainc_r))]

        # phi
        sigmaphi_l = [i for i in samples[:,9] if (i - MaP[9])<0]; sigmaphi_r = [i for i in samples[:,9] if (i - MaP[9])>0]
        if len(sigmaphi_l) == 0: sigmaphi_l.append(0);
        if len(sigmaphi_r) == 0: sigmaphi_r.append(0);
        pdict['sigmaphi'] = [(np.std(sigmaphi_l),np.std(sigmaphi_r))]

    # We don't need data entry
    pdict['Data'] = None
    with open(path + '/params_model.json','w') as f:
        f.write(json.dumps(pdict,cls=NumpyEncoder))
