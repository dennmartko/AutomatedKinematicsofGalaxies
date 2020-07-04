import os
import json
import gc
import time
import shutil
import scipy.stats
import sys

import numpy as np
import radio_beam as rb
import matplotlib.gridspec as gridspec
import seaborn as sns

from pyBBarolo import GalMod as GM
from pyBBarolo import FitMod3D as Fit3D
from spectral_cube import SpectralCube as SC
from astropy import units as u
from astropy.io import fits
from matplotlib import pyplot as plt

plt.switch_backend('agg')

from astropy import wcs
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot
from multiprocessing import Pool

from Functions import pixelcount,runMCMC,NumpyEncoder

def ptform(u):
    """Transforms the uniform random variable `u ~ Unif[0., 1.)`
    to the parameter of interest."""

    x = np.array(u)
    # Vrot
    x[0:4] = x[0:4]*150

    # Vdisp
    x[4:8] = x[4:8] * 30

    if len(x) == 9:
        x[8] = x[8]*90
    if len(x) == 10:
        x[8] = x[8]*90
        x[9] = x[9]*360
    return x

def loglike(x,**params):
   
    model = GM(os.path.dirname(os.path.abspath(__file__)) + '/examples/ngc2403.fits')

    if len(x) == 8:
        model.init(radii=params['radii'],xpos=params['xpos'],ypos=params['ypos'],vsys=params['vsys'],vrot=x[0:4],vrad=params['vrad'],
                        vdisp=x[4:8],z0=params['z0'],inc=params['inc'],phi=params['phi'])
    if len(x) == 9:
        model.init(radii=params['radii'],xpos=params['xpos'],ypos=params['ypos'],vsys=params['vsys'],vrot=x[0:4],vrad=params['vrad'],
                        vdisp=x[4:8],z0=params['z0'],inc=x[8],phi=params['phi'])
    if len(x) == 10:
        model.init(radii=params['radii'],xpos=params['xpos'],ypos=params['ypos'],vsys=params['vsys'],vrot=x[0:4],vrad=params['vrad'],
                        vdisp=x[4:8],z0=params['z0'],inc=x[8],phi=x[9])

    model.set_options(ltype=params['ltype'])
    mymodel = model.compute()

    temp_gal = SC.read(mymodel)
    beam = rb.Beam(major=params['radii'][-1]*params['beamfactor'][0]*u.arcsec, minor=params['radii'][-1]*params['beamfactor'][1]*u.arcsec, pa=params['phi']*u.deg) # fitted radius must be last one
    temp_gal = temp_gal.convolve_to(beam)
    mdata = temp_gal.unmasked_data[:,:,:]/temp_gal.unit
    
    if np.round((time.time() - params['start'])/3600,2) in np.arange(0,10,0.1):  
        gc.collect()

    return -0.5*np.sum([np.sum(np.abs(params['data'][ch] - mdata[ch]))/(params['std']*params['Nsignal_pix'][ch]) for ch in params['nch']])   # -len(data)*np.log(np.sqrt(2*np.pi*params['std']**2))-0.5*residual(data, mdata)/params['std']

# First extract the available galaxies and ngc2403 skeleton file
pathAGs = os.path.dirname(os.path.abspath(__file__)) + '/AGs'
galaxynames = [d for d in os.listdir(pathAGs) if os.path.isdir(os.path.join(pathAGs, d)) and d[0] != '.']
pathngc = os.path.dirname(os.path.abspath(__file__)) + '/examples/ngc2403.fits'

# FITS FILE PROCESSING
fits.setval(os.path.dirname(os.path.abspath(__file__)) + '/examples/ngc2403.fits', 'CUNIT3', value='km/s')

# Parameters specified in command line
ndim = int(sys.argv[1])
curvetype = sys.argv[3]
for galname in galaxynames:
    # Clean memory
    gc.collect()
    path = pathAGs + '/' + galname
    # Amount of cores
    cores = int(sys.argv[2])
    p1 = Pool(cores)
    p1.size = cores
    # Extract relevant parameters
    TData = json.load(open(path + '/params_' + galname + '.json')) # open to retrieve fixed params
    with fits.open(path + '/final_' + galname + '.fits') as ff:
        data = SC.read(ff)
        data = data.unmasked_data[:,:,:]/data.unit
        row = len(data[0][0])
        width = len(data[0][:,0])
        size = row*width
        nchannels = len(data)

    maskfile = path + '/output0/' + 'mask' + '.fits'
    assert os.path.isfile(maskfile), 'mask file needs to be called: galaxyname_mask.fits, current file is called: {}'.format(maskfile)
    Nsignal_pixels, Channel_w_signal = pixelcount(maskfile)

    ''' 
    Modify pdict for fixing certain parameters. fill 'vrot' and 'vdisp' with zeroes according to the amount of free parameters per ring for each ring
    '''
    if curvetype == 'flat':
        pdict = {'radii':[225,450,675,900], 'xpos':TData['xpos'], 'ypos':TData['ypos'], 'vsys':TData['vsys'], 'vrad':TData['vrad'], 'vrot':[96.958,108.211,112.106,114.070], 'vdisp':[13.224,10.731,9.554,8.998], 'z0':TData['z0'], 'inc':TData['inc'], 'phi':TData['phi'],
             'ltype':TData['ltype'], 'std':TData['std'], 'beamfactor':TData['beamfactor'], 'bpa':TData['phi'], 'row':row, 'width':width, 'size':size, 'nch':Channel_w_signal, 'Nsignal_pix':Nsignal_pixels, 'data':data}
    else:
        pdict = {'radii':[225,450,675,900], 'xpos':TData['xpos'], 'ypos':TData['ypos'], 'vsys':TData['vsys'], 'vrad':TData['vrad'], 'vrot':[86.066,100.849,111.547,121.198], 'vdisp':[13.224,10.731,9.554,8.998], 'z0':TData['z0'], 'inc':TData['inc'], 'phi':TData['phi'],
            'ltype':TData['ltype'], 'std':TData['std'], 'beamfactor':TData['beamfactor'], 'bpa':TData['phi'], 'row':row, 'width':width, 'size':size, 'nch':Channel_w_signal, 'Nsignal_pix':Nsignal_pixels, 'data':data}

    # Start the nested sampler
    runMCMC(path,ndim,p1,loglike,ptform,galname,pdict=pdict)
    p1.terminate()

