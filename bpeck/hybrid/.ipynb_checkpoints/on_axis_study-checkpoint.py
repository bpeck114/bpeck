#IMPORT PACKAGES
import os
import glob
import subprocess
import shutil
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from paarti.utils import maos_utils
import readbin
from astropy.io import fits
from paarti.psf_metrics import metrics
from bpeck.mcao import utils

def get_wfe_metrics(directory='./', seed=1):
    """
    COPIED FROM PAARTI
    
    Function to print various wave-front error (WFE) metrics 
    to terminal.

    Inputs:
    ------------
    directory      : string, default is current directory
        Path to directory where simulation results live

    seed           : int, default=10
        Seed with which simulation was run

    Outputs:
    ------------
    open_mean_nm   : array, len=3, dtype=float
        Array containing WFE metrics for open-loop MAOS results

    closed_mean_nm : array, len=3, dtype=float
        Array containing WFE metrics for closed-loop MAOS results
    """
    results_file = f'{directory}Res_{seed}.bin'
    results = readbin.readbin(results_file)
    #print("Looking in directory:", directory)

    # Open-loop WFE (nm): Piston removed, TT only, Piston+TT removed
    open_mean_nm = np.sqrt(results[0].mean(axis=0)) * 1.0e9

    # Closed-loop WFE (nm): Piston removed, TT only, Piston+TT removed
    clos_mean_nm = np.sqrt(results[2].mean(axis=0)) * 1.0e9

    #print('---------------------')
    #print('WaveFront Error (nm): [note, piston removed from all]')
    #print('---------------------')
    #print(f'{"      ":<7s}  {"Total":>11s}  {"High_Order":>11s}  {"TT":>11s}')
    #print(f'{"Open  ":<7s}  {open_mean_nm[0]:11.1f}  {open_mean_nm[2]:11.1f}  {open_mean_nm[1]:11.1f}')
    #print(f'{"Closed":<7s}  {clos_mean_nm[0]:11.1f}  {clos_mean_nm[2]:11.1f}  {clos_mean_nm[1]:11.1f}')

    return open_mean_nm, clos_mean_nm

def get_psf_metrics_x0y0(directory='./', oversamp=3, seed=1):
    """
    COPIED FROM PAARTI
    
    Print some PSF metrics for a central PSF computed by MAOS
    at an arbitrary number of wavelengths. Closed-loop.

    Inputs:
    ------------
    directory        : string, default is current directory
        Directory where MAOS simulation results live

    oversamp         : int, default=3

    seed             : int, default=10
        Simulation seed (seed value for which MAOS simulation was run)

    Outputs:
    ------------
    wavelengths      : array, dtype=float
        Array of wavelengths for which MAOS simulation was run and for
        which output metrics were calculated

    strehl_values    : array, dtype=float
        Array of Strehl values for each wavelength

    fwhm_gaus_values : array, dtype=float
        Array of FWHM values for Gaussians fit to each MAOS PSF at
        each wavelength

    fwhm_emp_values  : array, dtype=float
        Array of empirical FWHM values for each MAOS PSF. Empirical
        FWHM is calculated by locating the pixel with the largest flux,
        dividing that flux by 2, finding the nearest pixel with this halved 
        flux value, and computing the distance between them. This quantity
        is then converted to micro-arcsec (mas) using the MAOS pixel scale
        (arcsec/px) from the MAOS PSF header

    r_ee80_values    : array, dtype=float
        Array of radii for each MAOS PSF. At each wavelength, a radius
        is computed on the MAOS PSF, within which 80% of the total
        image flux is contained.
    """
    #print("Looking in %s for simulation results..." % directory)  
    fits_files = glob.glob(directory + f'evlpsfcl_{seed}_x0_y0.fits')
    psf_all_wvls = fits.open(fits_files[0])
    nwvl = len(psf_all_wvls)

    wavelengths = np.zeros(nwvl)
    strehl_values = np.zeros(nwvl)
    fwhm_gaus_values = np.zeros(nwvl)
    fwhm_emp_values = np.zeros(nwvl)
    r_ee80_values = np.zeros(nwvl)
 
    #print(f'{"Wavelength":10s} {"Strehl":>6s} {"FWHM_gaus":>10s} {"FWHM_emp":>10s} {"r_EE80":>6s}')
    #print(f'{"(microns)":10s} {"":>6s} {"(mas)":>10s} {"(mas)":>10s} {"(mas)":>6s}')
    
    for pp in range(nwvl):
        psf = psf_all_wvls[pp].data
        hdr = psf_all_wvls[pp].header
        mets = metrics.calc_psf_metrics_single(psf, hdr['DP'], 
                                               oversamp=oversamp)
        wavelengths[pp] = hdr["WVL"] * 1.0e6
        strehl_values[pp] = mets["strehl"]
        fwhm_gaus_values[pp] = mets["emp_fwhm"] * 1.0e3
        fwhm_emp_values[pp] = mets["fwhm"] * 1.0e3
        r_ee80_values[pp] = mets["ee80"] * 1.0e3

        sout  = f'{hdr["WVL"]*1e6:10.3f} '
        sout += f'{mets["strehl"]:6.2f} '
        sout += f'{mets["emp_fwhm"]*1e3:10.1f} ' 
        sout += f'{mets["fwhm"]*1e3:10.1f} ' 
        sout += f'{mets["ee80"]*1e3:6.1f}' 
        #print(sout)

    psf_all_wvls.close()
    return wavelengths, strehl_values, fwhm_gaus_values, fwhm_emp_values, r_ee80_values

def print_burn_wfe_metrics(directory='./', seed=1, start_index=100):
    results_file = f'{directory}Res_{seed}.bin'
    results = readbin.readbin(results_file)
    print("Looking in directory:", directory)

    # Open-loop WFE (nm): Piston removed, TT only, Piston+TT removed
    burn_open_mean_nm = np.sqrt(results[0][start_index:].mean(axis=0)) * 1.0e9

    # Closed-loop WFE (nm): Piston removed, TT only, Piston+TT removed
    burn_clos_mean_nm = np.sqrt(results[2][start_index:].mean(axis=0)) * 1.0e9

    print('---------------------')
    print('WaveFront Error (nm): [note, piston removed from all]')
    print('---------------------')
    print(f'{"      ":<7s}  {"Total":>11s}  {"High_Order":>11s}  {"TT":>11s}')
    print(f'{"Open  ":<7s}  {burn_open_mean_nm[0]:11.1f}  {burn_open_mean_nm[2]:11.1f}  {burn_open_mean_nm[1]:11.1f}')
    print(f'{"Closed":<7s}  {burn_clos_mean_nm[0]:11.1f}  {burn_clos_mean_nm[2]:11.1f}  {burn_clos_mean_nm[1]:11.1f}')

    return burn_open_mean_nm, burn_clos_mean_nm

def get_burn_wfe_metrics(directory='./', seed=1, start_index=100):
    results_file = f'{directory}Res_{seed}.bin'
    results = readbin.readbin(results_file)
    #print("Looking in directory:", directory)

    # Open-loop WFE (nm): Piston removed, TT only, Piston+TT removed
    burn_open_mean_nm = np.sqrt(results[0][start_index:].mean(axis=0)) * 1.0e9

    # Closed-loop WFE (nm): Piston removed, TT only, Piston+TT removed
    burn_clos_mean_nm = np.sqrt(results[2][start_index:].mean(axis=0)) * 1.0e9

    #print('---------------------')
    #print('WaveFront Error (nm): [note, piston removed from all]')
    #print('---------------------')
    #print(f'{"      ":<7s}  {"Total":>11s}  {"High_Order":>11s}  {"TT":>11s}')
    #print(f'{"Open  ":<7s}  {burn_open_mean_nm[0]:11.1f}  {burn_open_mean_nm[2]:11.1f}  {burn_open_mean_nm[1]:11.1f}')
    #print(f'{"Closed":<7s}  {burn_clos_mean_nm[0]:11.1f}  {burn_clos_mean_nm[2]:11.1f}  {burn_clos_mean_nm[1]:11.1f}')

    return burn_open_mean_nm, burn_clos_mean_nm

def get__burn_wfe_metrics_over_field(start_index=100, directory='./', seed=1):
    """
    Function to print various wave-front error (WFE) metrics 
    to terminal.

    Inputs:
    ------------
    directory      : string, default is current directory
        Path to directory where simulation results live

    seed           : int, default=10
        Seed with which simulation was run

    Outputs:
    ------------
    open_mean_nm   : array, len=3, dtype=float
        Array containing WFE metrics for open-loop MAOS results
        averaged over all the PSF evalution locations.

    closed_mean_nm : array, len=3, dtype=float
        Array containing WFE metrics for closed-loop MAOS results
        averaged over all the PSF evalution locations.

    open_xx_mean_nm   : array, shape=[N,3], dtype=float
        Array containing WFE metrics for open-loop MAOS results
        evaluated at each PSF location. Shape is [N, 3] where
        N is the number of PSF locations. Will return None if
        only a single PSF location. 

    closed_xx_mean_nm : array, shape=[N,3], dtype=float
        Array containing WFE metrics for closed-loop MAOS results
        evaluated at each PSF location. Shape is [N, 3] where
        N is the number of PSF locations. Will return None if
        only a single PSF location.
    
    """
    # Field averaged results
    results_file = f'{directory}Res_{seed}.bin'
    results = readbin.readbin(results_file)
    #print("Looking in directory:", directory)

    # Open-loop WFE (nm): Piston removed, TT only, Piston+TT removed
    burn_open_mean_nm = np.sqrt(results[0][start_index:].mean(axis=0)) * 1.0e9

    # Closed-loop WFE (nm): Piston removed, TT only, Piston+TT removed
    burn_clos_mean_nm = np.sqrt(results[2][start_index:].mean(axis=0)) * 1.0e9

    # Field-dependent resutls
    # Determine if we have a field-dependent WFE results file in extra/
    results_xx_file = f'{directory}/extra/Resp_{seed}.bin'
    if os.path.exists(results_xx_file):
        results_xx = readbin.readbin(results_xx_file)

        burn_open_xx_mean_nm = np.zeros((results_xx[2].shape[0], 3), dtype=float)
        burn_clos_xx_mean_nm = np.zeros((results_xx[3].shape[0], 3), dtype=float)

        # Loop through PSF positions and get RMS WFE in nm
        for xx in range(burn_open_xx_mean_nm.shape[0]):
            
            # Open-loop WFE (nm): Piston removed, TT only, Piston+TT removed
            burn_open_xx_mean_nm[xx] = np.sqrt(results_xx[2][xx][start_index:].mean(axis=0)) * 1.0e9
            
            # Closed-loop WFE (nm): Piston removed, TT only, Piston+TT removed
            burn_clos_xx_mean_nm[xx] = np.sqrt(results_xx[3][xx][start_index:].mean(axis=0)) * 1.0e9

    else:
        results_xx = None
        burn_open_xx_mean_nm = None
        burn_clos_xx_mean_nm = None

    return burn_open_mean_nm, burn_clos_mean_nm, burn_open_xx_mean_nm, burn_clos_xx_mean_nm

def get_psf_metrics_over_field(directory='./', oversamp=3, seed=10):
    """
    Print some PSF metrics vs. wavelength and field position for PSFs
    computed by MAOS. Closed-loop.

    Inputs:
    ------------
    directory        : string, default is current directory
        Directory where MAOS simulation results live

    oversamp         : int, default=3

    seed             : int, default=10
        Simulation seed (seed value for which MAOS simulation was run)

    Outputs:
    ------------
    wavelengths      : array, dtype=float
        Array of wavelengths for which MAOS simulation was run and for
        which output metrics were calculated

    strehl_values    : array, dtype=float
        Array of Strehl values for each wavelength

    fwhm_gaus_values : array, dtype=float
        Array of FWHM values for Gaussians fit to each MAOS PSF at
        each wavelength

    fwhm_emp_values  : array, dtype=float
        Array of empirical FWHM values for each MAOS PSF. Empirical
        FWHM is calculated by locating the pixel with the largest flux,
        dividing that flux by 2, finding the nearest pixel with this halved 
        flux value, and computing the distance between them. This quantity
        is then converted to micro-arcsec (mas) using the MAOS pixel scale
        (arcsec/px) from the MAOS PSF header

    r_ee80_values    : array, dtype=float
        Array of radii for each MAOS PSF. At each wavelength, a radius
        is computed on the MAOS PSF, within which 80% of the total
        image flux is contained.
    """
    #print("Looking in %s for simulation results..." % directory)  
    fits_files = glob.glob(directory + f'evlpsfcl_{seed}_x*_y*.fits')
    psf_all_wvls = fits.open(fits_files[0])
    nwvl = len(psf_all_wvls)
    npos = len(fits_files)

    psf_all_wvls.close()

    xpos = np.zeros((npos, nwvl), dtype=int)
    ypos = np.zeros((npos, nwvl), dtype=int)
    wavelengths = np.zeros((npos, nwvl), dtype=float)
    strehl_values = np.zeros((npos, nwvl), dtype=float)
    fwhm_gaus_values = np.zeros((npos, nwvl), dtype=float)
    fwhm_emp_values = np.zeros((npos, nwvl), dtype=float)
    r_ee50_values = np.zeros((npos, nwvl), dtype=float)
    r_ee80_values = np.zeros((npos, nwvl), dtype=float)
 
    for xx in range(npos):
        psf_all_wvls = fits.open(fits_files[xx])

        file_name = fits_files[xx].split('/')[-1]
        file_root = file_name.split('.')[0]
        tmp = file_root.split('_')
        tmpx = int(tmp[2][1:])
        tmpy = int(tmp[3][1:])
        #print('xx = ', tmpx, 'yy = ', tmpy)

        for pp in range(nwvl):
            xpos[xx, pp] = tmpx
            ypos[xx, pp] = tmpy
            
            psf = psf_all_wvls[pp].data
            hdr = psf_all_wvls[pp].header
            mets = metrics.calc_psf_metrics_single(psf, hdr['DP'], 
                                                   oversamp=oversamp)
            wavelengths[xx, pp] = hdr["WVL"] * 1.0e6
            strehl_values[xx, pp] = mets["strehl"]
            fwhm_gaus_values[xx, pp] = mets["emp_fwhm"] * 1.0e3
            fwhm_emp_values[xx, pp] = mets["fwhm"] * 1.0e3
            r_ee50_values[xx, pp] = mets["ee50"] * 1.0e3
            r_ee80_values[xx, pp] = mets["ee80"] * 1.0e3


        psf_all_wvls.close()
        
    return xpos, ypos, wavelengths, strehl_values, fwhm_gaus_values, fwhm_emp_values, r_ee50_values, r_ee80_values