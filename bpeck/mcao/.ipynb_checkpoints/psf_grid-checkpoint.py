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
from astropy.io import fits
from paarti.psf_metrics import metrics

"""
List of Functions
----------
plot_psf_wvl(radius_start, radius_stop, radius_step, mag, directory_results_path='A_keck_scao_lgs')
    - Plots the wavelengths for PSF grid for a single magnitude

plot_psf_mag_5_to_8_tt(tt_count, radius_start, radius_stop, radius_step, directory_results_path='A_keck_scao_lgs')

"""


def plot_psf_wvl(radius_start, radius_stop, radius_step, mag, directory_results_path='A_keck_scao_lgs'):
    strehl_2200 = []
    strehl_1673 = []
    strehl_1248 = []
    strehl_1020 = []
    strehl_0877 = []
    strehl_0810 = []
    strehl_0652 = []
    strehl_0544 = []
    strehl_0432 = []

    fwhm_2200 = []
    fwhm_1673 = []
    fwhm_1248 = []
    fwhm_1020 = []
    fwhm_0877 = []
    fwhm_0810 = []
    fwhm_0652 = []
    fwhm_0544 = []
    fwhm_0432 = []
    
    radius = np.arange(grid_start, grid_stop, grid_step)

    os.chdir
    
    radii_directory = f"{directory_results_path}/{mag}mag"
    os.chdir(radii_directory)

    for i, radii in enumeratre(radius):
        fits_files = glob.glob(directory + f'evlpsfcl_{seed}_x{radii}_y0.fits')
        psf_all_wvls = fits.open(fits_files[0])
        nwvl = len(psf_all_wvls)

        strehl_values = np.zeros(nwvl)
        fwhm_emp_values = np.zeros(nwvl)
        
    
        for pp in range(nwvl):
            psf = psf_all_wvls[pp].data
            hdr = psf_all_wvls[pp].header
            mets = metrics.calc_psf_metrics_single(psf, hdr['DP'], oversamp=oversamp)
            strehl_values[pp] = mets["strehl"]
            fwhm_emp_values[pp] = mets["fwhm"] * 1e3

            sout += f'{mets["strehl"]:6.2f} '
            sout += f'{mets["emp_fwhm"]*1e3:10.1f} ' 

        psf_all_wvls.close()

        strehl_2200.append(np.around(strehl_values[8], decimals=2))
        strehl_1673.append(np.around(strehl_values[7], decimals=2))
        strehl_1248.append(np.around(strehl_values[6], decimals=2))
        strehl_1020.append(np.around(strehl_values[5], decimals=2))
        strehl_0877.append(np.around(strehl_values[4], decimals=2))
        strehl_0810.append(np.around(strehl_values[3], decimals=2))
        strehl_0652.append(np.around(strehl_values[2], decimals=2))
        strehl_0544.append(np.around(strehl_values[1], decimals=2))
        strehl_0432.append(np.around(strehl_values[0], decimals=2))

        fwhm_2200.append(np.around(fwhm_emp_values[8], decimals=1))
        fwhm_1673.append(np.around(fwhm_emp_values[7], decimals=1))
        fwhm_1248.append(np.around(fwhm_emp_values[6], decimals=1))
        fwhm_1020.append(np.around(fwhm_emp_values[5], decimals=1))
        fwhm_0877.append(np.around(fwhm_emp_values[4], decimals=1))
        fwhm_0810.append(np.around(fwhm_emp_values[3], decimals=1))
        fwhm_0652.append(np.around(fwhm_emp_values[2], decimals=1))
        fwhm_0544.append(np.around(fwhm_emp_values[1], decimals=1))
        fwhm_0432.append(np.around(fwhm_emp_values[0], decimals=1))

    figure, axis = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
        
    axis[0].plot(radius, strehl_2200, color='#FF0000', label="2200 nm")
    axis[0].plot(radius, strehl_1673, color='#FF7F00', label="1673 nm")
    axis[0].plot(radius, strehl_1248, color='#FFFF00', label="1248 nm")
    axis[0].plot(radius, strehl_1020, color='#7FFF00', label="1020 nm")
    axis[0].plot(radius, strehl_0877, color='#00FF00', label="877 nm")
    axis[0].plot(radius, strehl_0810, color='#00FF7F', label="810 nm")
    axis[0].plot(radius, strehl_0652, color='#00FFFF', label="652 nm")
    axis[0].plot(radius, strehl_0544, color='#007FFF', label="544 nm")
    axis[0].plot(radius, strehl_0432, color='#0000FF', label="432 nm")
    #axis[0].set_xlabel("Number of LGS (n)")
    #axis[0].set_ylabel("Strehl Ratio")
    #axis[0].set_title("Strehl Ratio vs. LGS Count (7 mag LGS)")

    axis[1].plot(radius, fwhm_2200, color='#FF0000', label="2200 nm")
    axis[1].plot(radius, fwhm_1673, color='#FF7F00', label="1673 nm")
    axis[1].plot(radius, fwhm_1248, color='#FFFF00', label="1248 nm")
    axis[1].plot(radius, fwhm_1020, color='#7FFF00', label="1020 nm")
    axis[1].plot(radius, fwhm_0877, color='#00FF00', label="877 nm")
    axis[1].plot(radius, fwhm_0810, color='#00FF7F', label="810 nm")
    axis[1].plot(radius, fwhm_0652, color='#00FFFF', label="652 nm")
    axis[1].plot(radius, fwhm_0544, color='#007FFF', label="544 nm")
    axis[1].plot(radius, fwhm_0432, color='#0000FF', label="432 nm")
    #axis[1].legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3)
    axis[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)
    #axis[1].set_xlabel("Number of LGS (n)")
    #axis[1].set_ylabel("Full-Width Half Max (mas)")
    #axis[1].set_title("Empirical FWHM vs. LGS Count (7 mag LGS)")

    #plt.savefig('l.png')
    plt.show()

def plot_psf_mag_5_to_8_tt(tt_count, radius_start=0, radius_stop=65, radius_step=5, mag_start=5, mag_stop=9, mag_step=1, directory_results_path='A_keck_scao_lgs',seed=1,oversamp=3):
    strehl_1000_mag8 = []
    strehl_1000_mag7 = []
    strehl_1000_mag6 = []
    strehl_1000_mag5 = []

    fwhm_1000_mag8 = []
    fwhm_1000_mag8 = [] 
    fwhm_1000_mag8 = []
    fwhm_1000_mag8 = []
    
    radius = np.arange(radius_start, radius_stop, radius_step)
    magnitude = np.arange(mag_start, mag_stop, mag_step)

    for i, mag in enumerate(magnitude):
        tt_mag_directory = f"{directory_results_path}/{tt_count}tt/{mag}mag"
        os.chdir(tt_mag_directory)

        cwd = os.getcwd()
        print(cwd)
        
        for i, rad in enumerate(radius):
            fits_files = glob.glob(f'evlpsfcl_{seed}_x{rad}_y0.fits')
            print(fits_files)
            psf_all_wvls = fits.open(fits_files[0])
            nwvl = len(psf_all_wvls)

            strehl_values = np.zeros(nwvl)
            fwhm_emp_values = np.zeros(nwvl)
        
    
        for pp in range(nwvl):
            psf = psf_all_wvls[pp].data
            hdr = psf_all_wvls[pp].header
            mets = metrics.calc_psf_metrics_single(psf, hdr['DP'], oversamp=oversamp)
            strehl_values[pp] = mets["strehl"]
            fwhm_emp_values[pp] = mets["fwhm"] * 1e3

            sout += f'{mets["strehl"]:6.2f} '
            sout += f'{mets["emp_fwhm"]*1e3:10.1f} ' 

        psf_all_wvls.close()

        strehl_1000_mag[i].append(np.around(strehl_values[5], decimals=2))

        fwhm_1000_mag[i].append(np.around(fwhm_emp_values[5], decimals=1))

    figure, axis = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
        
    axis[0].plot(radius, strehl_1000_mag8, color='red', label="LGS Magnitude 8")
    axis[0].plot(radius, strehl_1000_mag7, color='green', label="LGS Magnitude 7")
    axis[0].plot(radius, strehl_1000_mag6, color='blue', label="LGS Magnitude 6")
    axis[0].plot(radius, strehl_1000_mag5, color='cyan', label="LGS Magnitude 5")
    #axis[0].set_xlabel("Number of LGS (n)")
    #axis[0].set_ylabel("Strehl Ratio")
    #axis[0].set_title("Strehl Ratio vs. LGS Count (7 mag LGS)")

    axis[1].plot(radius, fwhm_1000_mag8, color='red', label="LGS Magnitude 8")
    axis[1].plot(radius, fwhm_1000_mag7, color='green', label="LGS Magnitude 7")
    axis[1].plot(radius, fwhm_1000_mag6, color='blue', label="LGS Magnitude 6")
    axis[1].plot(radius, fwhm_1000_mag5, color='cyan', label="LGS Magnitude 5")
    axis[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)
    #axis[1].set_xlabel("Number of LGS (n)")
    #axis[1].set_ylabel("Full-Width Half Max (mas)")
    #axis[1].set_title("Empirical FWHM vs. LGS Count (7 mag LGS)")

    #plt.savefig('l.png')
    plt.show()
        