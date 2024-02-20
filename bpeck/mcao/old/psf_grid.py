#Once all the A_keck_scao_lgs.conf files have been created and properly updated
#A_keck_scao_lgs.conf files should be called: A_keck_scao_lgs_2000.conf, A_keck_scao_lgs_3000.conf, etc

#IMPORT PACKAGES
import os
import glob
import subprocess
import shutil
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import interp1d
from paarti.utils import maos_utils
from paarti import psfs, psf_plots
from paarti.psf_metrics import metrics
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import colors
from matplotlib.colorbar import Colorbar
from astropy.io import fits

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def return_psf_metrics_x0y0(rad_start, rad_stop, rad_step, directory='./', oversamp=3, seed=1, cut_radius=20):
    """
    Print some PSF metrics for a central PSF computed by MAOS
    at an arbitrary number of wavelengths.
    """
    radius = np.arange(rad_start, rad_stop, rad_step)
    
    for radii in radius: 
        fits_files = glob.glob(directory + f'evlpsfcl_{seed}_x{radii}_y0.fits')
    
        psf_all_wvls = fits.open(fits_files[0])
 
        nwvl = len(psf_all_wvls)

        wavelengths = np.zeros(nwvl)
        strehl_values = np.zeros(nwvl)
        fwhm_gaus_values = np.zeros(nwvl)
        fwhm_emp_values = np.zeros(nwvl)
        r_ee80_values = np.zeros(nwvl)
 
        print(f'{"Wavelength":10s} {"Strehl":>6s} {"FWHM_gaus":>10s} {"FWHM_emp":>10s} {"r_EE80":>6s}')
        print(f'{"(microns)":10s} {"":>6s} {"(mas)":>10s} {"(mas)":>10s} {"(mas)":>6s}')
    
        for pp in range(nwvl):
            psf = psf_all_wvls[pp].data
            hdr = psf_all_wvls[pp].header
            mets = metrics.calc_psf_metrics_single(psf, hdr['DP'], oversamp=oversamp)
            wavelengths[pp] = hdr["WVL"] * 1e6
            strehl_values[pp] = mets["strehl"]
            fwhm_gaus_values[pp] = mets["emp_fwhm"] * 1e3
            fwhm_emp_values[pp] = mets["fwhm"] * 1e3
            r_ee80_values[pp] = mets["ee80"] * 1e3

            sout  = f'{hdr["WVL"]*1e6:10.3f} '
            sout += f'{mets["strehl"]:6.2f} '
            sout += f'{mets["emp_fwhm"]*1e3:10.1f} ' 
            sout += f'{mets["fwhm"]*1e3:10.1f} ' 
            sout += f'{mets["ee80"]*1e3:6.1f}' 
            print(sout)

        psf_all_wvls.close()
 
    return