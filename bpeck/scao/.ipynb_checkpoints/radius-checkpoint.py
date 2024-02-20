import os
import glob
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from paarti.utils import maos_utils

def plot_psf_wfe(rad_start, rad_stop, rad_step, directory_results_path='A_keck_scao_lgs'):
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

    radius = np.arange(rad_start, rad_stop, rad_step)

    for rad in radius:
        rad_directory = f"{directory_results_path}/{rad}rad"
        os.chdir(rad_directory)

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
    
    axis[0].plot(radius, strehl_2200, color='red', label="2200 nm")
    axis[0].plot(radius, strehl_1650, color='orange', label="1650 nm")
    axis[0].plot(radius, strehl_1250, color='green', label="1250 nm")
    axis[0].plot(radius, strehl_1000, color='cyan', label="1000 nm")
    axis[0].plot(radius, strehl_0800, color='blue', label="800 nm")
    axis[0].set_xlabel("Radius of Tip-Tilt Star (as)")
    axis[0].set_ylabel("Strehl Ratio")
    axis[0].set_title("Strehl Ratio vs. Radius of TT Star")

    axis[1].plot(radius, fwhm_2200, color='red', label="2200 nm")
    axis[1].plot(radius, fwhm_1650, color='orange', label="1650 nm")
    axis[1].plot(radius, fwhm_1250, color='green', label="1250 nm")
    axis[1].plot(radius, fwhm_1000, color='cyan', label="1000 nm")
    axis[1].plot(radius, fwhm_0800, color='blue', label="800 nm")
    axis[1].set_xlabel("Radius of Tip-Tilt Star (as)")
    axis[1].set_ylabel("Full-Width Half Max (mas)")
    axis[1].set_title("Empirical FWHM vs. Radius of TT Star")

    axis[2].plot(radius, total, color="red", label="Total WFE", linestyle='--')
    axis[2].plot(radius, tt, color="blue", label="High-Order WFE", linestyle='--')
    axis[2].plot(radius, ho, color="green", label="Tip-Tilt WFE", linestyle='--')
    axis[2].set_xlabel("Radius of Tip-Tilt Star (as)")
    axis[2].set_ylabel("Wave-front Error (nm)")
    axis[2].set_title("WFE vs. Radius of TT Star") 

    axis[1].legend(bbox_to_anchor=(0, -0.15), loc="upper center", ncol=5, fontsize='small')
    axis[2].legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3, fontsize='small')
    plt.subplots_adjust(wspace=0.4)

    #plt.tight_layout()
    #plt.subplots_adjust(vspace=30)
    plt.savefig('scao_magnitude.png')
    plt.show()