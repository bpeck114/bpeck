import os
import glob
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from paarti.utils import maos_utils

def paarti_mag_to_flux(mag_start, mag_stop, mag_step, lgs_mag):
    lgs = []
    truth = []
    tt = []

    lgs_siglev = []
    lgs_bkgrnd = []
    lgs_nearecon = []

    tt_siglev = []
    tt_bkgrnd = []
    tt_nearecon = []

    truth_siglev = []
    truth_bkgrnd = []
    truth_nearecon = []
    
    magnitudes = np.arange(mag_start, mag_stop, mag_step)

    for magnitude in magnitudes:
        lgs_flux = maos_utils.keck_nea_photons(lgs_mag, wfs="LGSWFS", wfs_int_time=1/472)
        tt_flux = maos_utils.keck_nea_photons(magnitude, wfs="STRAP", wfs_int_time = 1/472)
        truth_flux = maos_utils.keck_nea_photons(magnitude, wfs="LBWFS", wfs_int_time = 1000/472)

        lgs.append(lgs_flux)
        tt.append(tt_flux)
        truth.append(truth_flux)

    for lgs_parameter in lgs:
        _, sigma_theta, Np, Nb = lgs_parameter
        lgs_nearecon.append(round(sigma_theta, 3))
        lgs_siglev.append(round(Np, 3))
        lgs_bkgrnd.append(round(Nb, 3))

    for tt_parameter in tt:  
        _, sigma_theta, Np, Nb = tt_parameter
        tt_nearecon.append(round(sigma_theta, 3))
        tt_siglev.append(round(Np, 3))
        tt_bkgrnd.append(round(Nb, 3))

    for truth_parameter in truth:  
        _, sigma_theta, Np, Nb = truth_parameter
        truth_nearecon.append(round(sigma_theta, 3))
        truth_siglev.append(round(Np, 3))
        truth_bkgrnd.append(round(Nb, 3))

    print('--------------------')
    print('SCAO Flux Parameters:')
    print('--------------------') 

    for i, magnitude in enumerate(magnitudes):
        print('#Tip-Tilt Magnitude:', magnitude)
        print('#powfs.siglev = [',lgs_siglev[i], tt_siglev[i], truth_siglev[i], ']')
        print('#powfs.bkgrnd = [', lgs_bkgrnd[i], tt_bkgrnd[i], truth_bkgrnd[i], ']')
        print('#powfs.nearecon = [', lgs_nearecon[i], tt_nearecon[i], truth_nearecon[i], ']')
        print('')
                                                                
    return 

def plot_psf(mag_start, mag_stop, mag_step, directory_results_path='A_keck_scao_lgs'):
    psf_metrics = []
    strehl = []
    fwhm = []
    
    strehl_2200 = []
    strehl_1650 = []
    strehl_1250 = []
    strehl_1000 = []
    strehl_0800 = []

    fwhm_2200 = []
    fwhm_1650 = []
    fwhm_1250 = []
    fwhm_1000 = []
    fwhm_0800 = []

    magnitudes = np.arange(mag_start, mag_stop, mag_step)

    for mag in magnitudes:
        mag_directory = f"{directory_results_path}/{mag}mag"
        os.chdir(mag_directory)

        psf = maos_utils.print_psf_metrics_x0y0(oversamp=3, seed=1)
        psf_metrics.append(psf)

        os.chdir('..')
        os.chdir('..')

    for psf_value in psf_metrics:
        wavelengths, strehl_values, fwhm_gaus_values, fwhm_emp_values, r_ee80_values = psf_value

        strehl_2200.append(np.around(strehl_values[4], decimals=2))
        strehl_1650.append(np.around(strehl_values[3], decimals=2))
        strehl_1250.append(np.around(strehl_values[2], decimals=2))
        strehl_1000.append(np.around(strehl_values[1], decimals=2))
        strehl_0800.append(np.around(strehl_values[0], decimals=2))

        fwhm_2200.append(np.around(fwhm_emp_values[4], decimals=2))
        fwhm_1650.append(np.around(fwhm_emp_values[3], decimals=2))
        fwhm_1250.append(np.around(fwhm_emp_values[2], decimals=2))
        fwhm_1000.append(np.around(fwhm_emp_values[1], decimals=2))
        fwhm_0800.append(np.around(fwhm_emp_values[0], decimals=2))

    figure, axis = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    
    axis[0].plot(magnitudes, strehl_2200, color='#FF0000', label="2200 nm")
    axis[0].plot(magnitudes, strehl_1650, color='#FF7F00', label="1650 nm")
    axis[0].plot(magnitudes, strehl_1250, color='#FFFF00', label="1250 nm")
    axis[0].plot(magnitudes, strehl_1000, color='#7FFF00', label="1000 nm")
    axis[0].plot(magnitudes, strehl_0800, color='#00FF00', label="800 nm")
    axis[0].set_xlabel("Magnitude of Tip-Tilt Star (mag)")
    axis[0].set_ylabel("Strehl Ratio")
    axis[0].set_title("Strehl Ratio vs. Magnitude of TT Star")

    axis[1].plot(magnitudes, fwhm_2200, color='#FF0000', label="2200 nm")
    axis[1].plot(magnitudes, fwhm_1650, color='#FF7F00', label="1650 nm")
    axis[1].plot(magnitudes, fwhm_1250, color='#FFFF00', label="1250 nm")
    axis[1].plot(magnitudes, fwhm_1000, color='#7FFF00', label="1000 nm")
    axis[1].plot(magnitudes, fwhm_0800, color='#00FF00', label="800 nm")
    axis[1].set_xlabel("Magnitude of Tip-Tilt Star (mag)")
    axis[1].set_ylabel("Full-Width Half Max (mas)")
    axis[1].set_title("Empirical FWHM vs. Magnitude of TT Star")

    #axis[1].legend(bbox_to_anchor=(0, -0.15), loc="upper center", ncol=5)

    plt.tight_layout()
    plt.savefig('scao_magnitude.png')
    plt.show()

def plot_wfe(mag_start, mag_stop, mag_step, directory_results_path='A_keck_scao_lgs'):
    total = []
    tt = []
    ho = []

    magnitudes = np.arange(mag_start, mag_stop, mag_step)

    for mag in magnitudes:
        magnitude_directory = f"{directory_results_path}/{mag}mag"
        os.chdir(magnitude_directory)

        wfe_metrics = maos_utils.print_wfe_metrics(seed=1)
        open_mean_nm, clos_mean_nm = wfe_metrics

        total.append(np.round(clos_mean_nm[0], decimals=1))
        tt.append(np.round(clos_mean_nm[1], decimals=1))
        ho.append(np.round(clos_mean_nm[2], decimals=1))
            
        os.chdir('..')
        os.chdir('..')

    figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(4,4))

    axis.plot(magnitudes, total, color="red", label="Total WFE")
    axis.plot(magnitudes, tt, color="blue", label="High-Order WFE")
    axis.plot(magnitudes, ho, color="green", label="Tip-Tilt WFE")
    axis.set_xlabel("Magnitude of Tip-Tilt Star (mag)")
    axis.set_ylabel("Wave-front Error (nm)")
    axis.set_title("WFE vs. Magnitude of TT Star")

    axis.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3)
    
    plt.savefig('scao_wfe.png')
    plt.show()

def plot_psf_wfe(mag_start, mag_stop, mag_step, directory_results_path='A_keck_scao_lgs'):
    psf_metrics = []
    strehl = []
    fwhm = []

    total = []
    tt = []
    ho = []
    
    strehl_2200 = []
    strehl_1650 = []
    strehl_1250 = []
    strehl_1000 = []
    strehl_0800 = []

    fwhm_2200 = []
    fwhm_1650 = []
    fwhm_1250 = []
    fwhm_1000 = []
    fwhm_0800 = []

    magnitudes = np.arange(mag_start, mag_stop, mag_step)

    for mag in magnitudes:
        mag_directory = f"{directory_results_path}/{mag}mag"
        os.chdir(mag_directory)

        psf = maos_utils.print_psf_metrics_x0y0(oversamp=3, seed=1)
        psf_metrics.append(psf)

        wfe_metrics = maos_utils.print_wfe_metrics(seed=1)
        open_mean_nm, clos_mean_nm = wfe_metrics

        total.append(np.round(clos_mean_nm[0], decimals=1))
        tt.append(np.round(clos_mean_nm[1], decimals=1))
        ho.append(np.round(clos_mean_nm[2], decimals=1))

        os.chdir('..')
        os.chdir('..')

    for psf_value in psf_metrics:
        wavelengths, strehl_values, fwhm_gaus_values, fwhm_emp_values, r_ee80_values = psf_value
        
        strehl.append(np.around(strehl_values, decimals=2))
        fwhm.append(np.around(fwhm_emp_values, decimals=1))

        strehl_2200.append(np.around(strehl_values[4], decimals=2))
        strehl_1650.append(np.around(strehl_values[3], decimals=2))
        strehl_1250.append(np.around(strehl_values[2], decimals=2))
        strehl_1000.append(np.around(strehl_values[1], decimals=2))
        strehl_0800.append(np.around(strehl_values[0], decimals=2))

        fwhm_2200.append(np.around(fwhm_emp_values[4], decimals=2))
        fwhm_1650.append(np.around(fwhm_emp_values[3], decimals=2))
        fwhm_1250.append(np.around(fwhm_emp_values[2], decimals=2))
        fwhm_1000.append(np.around(fwhm_emp_values[1], decimals=2))
        fwhm_0800.append(np.around(fwhm_emp_values[0], decimals=2))

    figure, axis = plt.subplots(nrows=1, ncols=3, figsize=(14,4))
    
    axis[0].plot(magnitudes, strehl_2200, color='red', label="2200 nm")
    axis[0].plot(magnitudes, strehl_1650, color='orange', label="1650 nm")
    axis[0].plot(magnitudes, strehl_1250, color='green', label="1250 nm")
    axis[0].plot(magnitudes, strehl_1000, color='cyan', label="1000 nm")
    axis[0].plot(magnitudes, strehl_0800, color='blue', label="800 nm")
    axis[0].set_xlabel("Magnitude of Tip-Tilt Star (mag)")
    axis[0].set_ylabel("Strehl Ratio")
    axis[0].set_title("Strehl Ratio vs. Magnitude of TT Star")

    axis[1].plot(magnitudes, fwhm_2200, color='red', label="2200 nm")
    axis[1].plot(magnitudes, fwhm_1650, color='orange', label="1650 nm")
    axis[1].plot(magnitudes, fwhm_1250, color='green', label="1250 nm")
    axis[1].plot(magnitudes, fwhm_1000, color='cyan', label="1000 nm")
    axis[1].plot(magnitudes, fwhm_0800, color='blue', label="800 nm")
    axis[1].set_xlabel("Magnitude of Tip-Tilt Star (mag)")
    axis[1].set_ylabel("Full-Width Half Max (mas)")
    axis[1].set_title("Empirical FWHM vs. Magnitude of TT Star")

    axis[2].plot(magnitudes, total, color="red", label="Total WFE", linestyle='--')
    axis[2].plot(magnitudes, tt, color="blue", label="Tip-Tilt WFE", linestyle='--')
    axis[2].plot(magnitudes, ho, color="green", label="High-Order WFE", linestyle='--')
    axis[2].set_xlabel("Magnitude of Tip-Tilt Star (mag)")
    axis[2].set_ylabel("Wave-front Error (nm)")
    axis[2].set_title("WFE vs. Magnitude of TT Star") 

    axis[1].legend(bbox_to_anchor=(0, -0.15), loc="upper center", ncol=5, fontsize='small')
    axis[2].legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3, fontsize='small')
    plt.subplots_adjust(wspace=0.4)

    #plt.tight_layout()
    #plt.subplots_adjust(vspace=30)
    plt.savefig('scao_magnitude.png')
    plt.show()