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


def psf_metrics_x0y0(directory='./', oversamp=3, seed=10, cut_radius=20):
    """
    Copied from PAARTI, just doesn't print message.
    Print some PSF metrics for a central PSF computed by MAOS
    at an arbitrary number of wavelengths.
    """
    print("Looking in directory:", directory)  
    fits_files = glob.glob(directory + f'evlpsfcl_{seed}_x0_y0.fits')
    
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
        #print(sout)

    psf_all_wvls.close()
 
    return wavelengths, strehl_values, fwhm_gaus_values, fwhm_emp_values, r_ee80_values                                    

def plot_psf_wvls(lgs_start, lgs_stop, lgs_step, mag, directory_results_path='A_keck_scao_lgs'):
    '''
    Plots the strehl ratio and fwhm of the actuat
    '''
    psf_metrics = []
    lgss = np.arange(lgs_start, lgs_stop, lgs_step)

    strehl = []
    fwhm = []

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

    #cwd = os.getcwd()
    #print(cwd)

    for lgs in lgss:
        directory = f"{directory_results_path}/{mag}mag/{lgs}lgs"
        #print(directory)
        os.chdir(directory)

        psf = maos_utils.print_psf_metrics_x0y0(oversamp=3,seed=1)
        psf_metrics.append(psf)

        os.chdir('..')
        os.chdir('..')
        os.chdir('..')

    for psf_value in psf_metrics:
        wavelengths, strehl_values, fwhm_gaus_values, fwhm_emp_values, r_ee80_values = psf_value

        strehl.append(np.around(strehl_values, decimals=2))
        fwhm.append(np.around(fwhm_emp_values, decimals=1))

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
        
    axis[0].plot(lgss, strehl_2200, color='#FF0000', label="2200 nm")
    axis[0].plot(lgss, strehl_1673, color='#FF7F00', label="1673 nm")
    axis[0].plot(lgss, strehl_1248, color='#FFFF00', label="1248 nm")
    axis[0].plot(lgss, strehl_1020, color='#7FFF00', label="1020 nm")
    axis[0].plot(lgss, strehl_0877, color='#00FF00', label="877 nm")
    axis[0].plot(lgss, strehl_0810, color='#00FF7F', label="810 nm")
    axis[0].plot(lgss, strehl_0652, color='#00FFFF', label="652 nm")
    axis[0].plot(lgss, strehl_0544, color='#007FFF', label="544 nm")
    axis[0].plot(lgss, strehl_0432, color='#0000FF', label="432 nm")
    axis[0].set_xlabel("Number of LGS (n)")
    axis[0].set_ylabel("Strehl Ratio")
    axis[0].set_title("Strehl Ratio vs. LGS Count (7 mag LGS)")

    axis[1].plot(lgss, fwhm_2200, color='#FF0000', label="2200 nm")
    axis[1].plot(lgss, fwhm_1673, color='#FF7F00', label="1673 nm")
    axis[1].plot(lgss, fwhm_1248, color='#FFFF00', label="1248 nm")
    axis[1].plot(lgss, fwhm_1020, color='#7FFF00', label="1020 nm")
    axis[1].plot(lgss, fwhm_0877, color='#00FF00', label="877 nm")
    axis[1].plot(lgss, fwhm_0810, color='#00FF7F', label="810 nm")
    axis[1].plot(lgss, fwhm_0652, color='#00FFFF', label="652 nm")
    axis[1].plot(lgss, fwhm_0544, color='#007FFF', label="544 nm")
    axis[1].plot(lgss, fwhm_0432, color='#0000FF', label="432 nm")
    #axis[1].legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3)
    axis[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)
    axis[1].set_xlabel("Number of LGS (n)")
    axis[1].set_ylabel("Full-Width Half Max (mas)")
    axis[1].set_title("Empirical FWHM vs. LGS Count (7 mag LGS)")

    plt.tight_layout()
    plt.savefig('lgs_study_psf_wvl.png')
    plt.show()
    
    cwd = os.getcwd()
    print(f"Current working directory:", cwd)

def plot_psf(lgs_start, lgs_stop, lgs_step, magnitudes, bandwidth=5, directory_results_path='A_keck_scao_lgs'):
    strehl_ = {}
    fwhm_ = {}

    lgss = np.arange(lgs_start, lgs_stop, lgs_step)

    for mag in magnitudes:
        strehl_[mag] = []
        fwhm_[mag] = []

        mag_directory = f"{directory_results_path}/{mag}mag"
        os.chdir(mag_directory)

        for lgs in lgss:
            act_directory = f"{lgs}lgs"
            os.chdir(act_directory)

            psf_metrics = maos_utils.print_psf_metrics_x0y0(oversamp=3, seed=1)
            wavelengths, strehl_values, fwhm_gaus_values, fwhm_emp_values, r_ee80_values = psf_metrics

            strehl_[mag].append(np.round(strehl_values[bandwidth], decimals=2))
            fwhm_[mag].append(np.round(fwhm_emp_values[bandwidth], decimals=1))

            os.chdir('..')

        os.chdir('..')
        os.chdir('..')

    figure, axis = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
    
    for mag, color in zip(magnitudes, ['red', 'orange', 'green', 'blue']):
        axis[0].plot(lgss, strehl_[mag], color=color, label=f"{mag} magnitude")
        axis[1].plot(lgss, fwhm_[mag], color=color, label=f"{mag} magnitude")

    for ax in axis:
        ax.set_xlabel("Number of LGS (n)")

    axis[0].set_ylabel("Strehl Ratio")
    axis[1].set_ylabel("Full-Width at Half Max (mas)")

    axis[0].set_title("Strehl Ratio vs. LGS Count (1000 nm)")
    axis[1].set_title("Empirical FWHM vs. LGS Count (1000 nm)")

    axis[1].legend(bbox_to_anchor=(-.1, -0.15), loc="upper center", ncol=4)
    plt.savefig('lgs_study_psf.png')
    plt.show()


def plot_wfe(lgs_start, lgs_stop, lgs_step, magnitudes, directory_results_path='A_keck_scao_lgs'):
    total_ = {}
    tt_ = {}
    ho_ = {}
    
    lgss = np.arange(lgs_start, lgs_stop, lgs_step)

    for mag in magnitudes:
        total_[mag] = []
        tt_[mag] = []
        ho_[mag] = []
        
        mag_directory = f"{directory_results_path}/{mag}mag"
        os.chdir(mag_directory)

        for lgs in lgss:
            act_directory = f"{lgs}lgs"
            os.chdir(act_directory)

            wfe_metrics = maos_utils.print_wfe_metrics(seed=1)
            open_mean_nm, clos_mean_nm = wfe_metrics

            total_[mag].append(np.round(clos_mean_nm[0], decimals=1))
            tt_[mag].append(np.round(clos_mean_nm[1], decimals=1))
            ho_[mag].append(np.round(clos_mean_nm[2], decimals=1))
            
            os.chdir('..')
            
        os.chdir('..')
        os.chdir('..')

    figure, axis = plt.subplots(nrows=1, ncols=3, figsize=(16,4), sharey=True)
    
    for mag, color in zip(magnitudes, ['red', 'orange', 'green', 'blue']):
        axis[0].plot(lgss, total_[mag], color=color, label=f"{mag} magnitude")
        axis[1].plot(lgss, tt_[mag], color=color, label=f"{mag} magnitude")
        axis[2].plot(lgss, ho_[mag], color=color, label=f"{mag} magnitude")

    for ax in axis:
        ax.set_xlabel("Number of LGS (n)")

    axis[0].set_ylabel("Total Wave-front Error (nm)")
    axis[1].set_ylabel("Tip-Tilt Wave-front Error (nm)")
    axis[2].set_ylabel("High-Order Wave-front Error (nm)")

    axis[0].tick_params(left=True, labelleft=True)
    axis[1].tick_params(left=True, labelleft=True)
    axis[2].tick_params(left=True, labelleft=True)

    axis[0].set_title("Total WFE vs. LGS Count")
    axis[1].set_title("Tip-Tilt WFE vs. LGS Count")
    axis[2].set_title("High-Order WFE vs. LGS Count")

    axis[1].legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=4)
    plt.savefig('act_study_wfe.png')
    plt.show()