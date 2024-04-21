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

def plot_wfe(height_start, height_stop, height_step, directory_results_path='A_keck_scao_lgs'):
    total_dm2 = []
    tt_dm2 = []
    ho_dm2 = []

    total_dm3 = []
    tt_dm3 = []
    ho_dm3 = []

    heights = np.arange(height_start, height_stop, height_step)

    dm2_directory = f"{directory_results_path}/dm2"
    os.chdir(dm2_directory)
    
    for height in heights:
        height_directory = f"{height}km"
        os.chdir(height_directory)

        wfe_metrics = maos_utils.print_wfe_metrics(seed=1)
        open_mean_nm, clos_mean_nm = wfe_metrics

        total_dm2.append(np.round(clos_mean_nm[0], decimals=1))
        tt_dm2.append(np.round(clos_mean_nm[1], decimals=1))
        ho_dm2.append(np.round(clos_mean_nm[2], decimals=1))

        os.chdir('..')

    os.chdir('..')

    dm3_directory = f"dm3"
    os.chdir(dm3_directory)

    for height in heights:
        height_directory = f"{height}km"
        os.chdir(height_directory)

        wfe_metrics = maos_utils.print_wfe_metrics(seed=1)
        open_mean_nm, clos_mean_nm = wfe_metrics

        total_dm3.append(np.round(clos_mean_nm[0], decimals=1))
        tt_dm3.append(np.round(clos_mean_nm[1], decimals=1))
        ho_dm3.append(np.round(clos_mean_nm[2], decimals=1))

        os.chdir('..')

    os.chdir('..')
    os.chdir('..')

    figure, axis = plt.subplots(nrows=1, ncols=2, figsize=(12,4))

    axis[0].plot(heights, total_dm2, color="red", label="Total WFE")
    axis[0].plot(heights, tt_dm2, color="blue", label="TT WFE")
    axis[0].plot(heights, ho_dm2, color="green", label="High-Order WFE")
    axis[0].set_xlabel("Altitude of Conjugate High-Order DM2 (km)")
    axis[0].set_ylabel("Wave-front Error (nm)")
    axis[0].set_title("WFE vs. Altitude of HO DM2")

    axis[1].plot(heights, total_dm3, color="red", label="Total WFE")
    axis[1].plot(heights, tt_dm3, color="blue", label="TT WFE")
    axis[1].plot(heights, ho_dm3, color="green", label="High-Order WFE")
    axis[1].set_xlabel("Altitude of Conjugate High-Order DM3 (km)")
    axis[1].set_ylabel("Wave-front Error (nm)")
    axis[1].set_title("WFE vs. Altitude of HO DM3")

    axis[1].legend(bbox_to_anchor=(0, -0.15), loc="upper center", ncol=3)
    
    plt.savefig('scao_wfe.png')
    plt.show()

def plot_psf(height_start, height_stop, height_step, bandwidth=5, directory_results_path='A_keck_scao_lgs'):
    psf_metrics_dm2 = []
    psf_metrics_dm3 = []
    strehl_dm2 = []
    strehl_dm3 = []
    fwhm_dm2 = []
    fwhm_dm3 = []
    
    strehl_dm2_2200 = []
    strehl_dm2_1673 = []
    strehl_dm2_1248 = []
    strehl_dm2_1020 = []
    strehl_dm2_0877 = []
    strehl_dm2_0810 = []
    strehl_dm2_0652 = []
    strehl_dm2_0544 = []
    strehl_dm2_0432 = []

    strehl_dm3_2200 = []
    strehl_dm3_1673 = []
    strehl_dm3_1248 = []
    strehl_dm3_1020 = []
    strehl_dm3_0877 = []
    strehl_dm3_0810 = []
    strehl_dm3_0652 = []
    strehl_dm3_0544 = []
    strehl_dm3_0432 = []

    fwhm_dm2_2200 = []
    fwhm_dm2_1673 = []
    fwhm_dm2_1248 = []
    fwhm_dm2_1020 = []
    fwhm_dm2_0877 = []
    fwhm_dm2_0810 = []
    fwhm_dm2_0652 = []
    fwhm_dm2_0544 = []
    fwhm_dm2_0432 = []

    fwhm_dm3_2200 = []
    fwhm_dm3_1673 = []
    fwhm_dm3_1248 = []
    fwhm_dm3_1020 = []
    fwhm_dm3_0877 = []
    fwhm_dm3_0810 = []
    fwhm_dm3_0652 = []
    fwhm_dm3_0544 = []
    fwhm_dm3_0432 = []

    heights = np.arange(height_start, height_stop, height_step)

    dm2_directory = f"{directory_results_path}/dm2"
    os.chdir(dm2_directory)  

    for height in heights:
        height_directory = f"{height}km"
        os.chdir(height_directory)

        psf_dm2 = maos_utils.print_psf_metrics_x0y0(oversamp=3, seed=1)
        psf_metrics_dm2.append(psf_dm2)

        os.chdir('..')

    for psf_value_dm2 in psf_metrics_dm2:
        wavelengths, strehl_values, fwhm_gaus_values, fwhm_emp_values, r_ee80_values = psf_value_dm2

        strehl_dm2.append(np.around(strehl_values, decimals=2))
        fwhm_dm2.append(np.around(fwhm_emp_values, decimals=1))

        strehl_dm2_2200.append(np.around(strehl_values[8], decimals=2))
        strehl_dm2_1673.append(np.around(strehl_values[7], decimals=2))
        strehl_dm2_1248.append(np.around(strehl_values[6], decimals=2))
        strehl_dm2_1020.append(np.around(strehl_values[5], decimals=2))
        strehl_dm2_0877.append(np.around(strehl_values[4], decimals=2))
        strehl_dm2_0810.append(np.around(strehl_values[3], decimals=2))
        strehl_dm2_0652.append(np.around(strehl_values[2], decimals=2))
        strehl_dm2_0544.append(np.around(strehl_values[1], decimals=2))
        strehl_dm2_0432.append(np.around(strehl_values[0], decimals=2))

        fwhm_dm2_2200.append(np.around(fwhm_emp_values[8], decimals=1))
        fwhm_dm2_1673.append(np.around(fwhm_emp_values[7], decimals=1))
        fwhm_dm2_1248.append(np.around(fwhm_emp_values[6], decimals=1))
        fwhm_dm2_1020.append(np.around(fwhm_emp_values[5], decimals=1))
        fwhm_dm2_0877.append(np.around(fwhm_emp_values[4], decimals=1))
        fwhm_dm2_0810.append(np.around(fwhm_emp_values[3], decimals=1))
        fwhm_dm2_0652.append(np.around(fwhm_emp_values[2], decimals=1))
        fwhm_dm2_0544.append(np.around(fwhm_emp_values[1], decimals=1))
        fwhm_dm2_0432.append(np.around(fwhm_emp_values[0], decimals=1))

    os.chdir('..')

    dm3_directory = f"dm3"
    os.chdir(dm3_directory)

    for height in heights:
        height_directory = f"{height}km"
        os.chdir(height_directory)

        psf_dm3 = maos_utils.print_psf_metrics_x0y0(oversamp=3, seed=1)
        psf_metrics_dm3.append(psf_dm3)

        os.chdir('..')

    for psf_value_dm3 in psf_metrics_dm3:
        wavelengths, strehl_values, fwhm_gaus_values, fwhm_emp_values, r_ee80_values = psf_value_dm3

        strehl_dm3.append(np.around(strehl_values, decimals=2))
        fwhm_dm3.append(np.around(fwhm_emp_values, decimals=1))

        strehl_dm3_2200.append(np.around(strehl_values[8], decimals=2))
        strehl_dm3_1673.append(np.around(strehl_values[7], decimals=2))
        strehl_dm3_1248.append(np.around(strehl_values[6], decimals=2))
        strehl_dm3_1020.append(np.around(strehl_values[5], decimals=2))
        strehl_dm3_0877.append(np.around(strehl_values[4], decimals=2))
        strehl_dm3_0810.append(np.around(strehl_values[3], decimals=2))
        strehl_dm3_0652.append(np.around(strehl_values[2], decimals=2))
        strehl_dm3_0544.append(np.around(strehl_values[1], decimals=2))
        strehl_dm3_0432.append(np.around(strehl_values[0], decimals=2))

        fwhm_dm3_2200.append(np.around(fwhm_emp_values[8], decimals=1))
        fwhm_dm3_1673.append(np.around(fwhm_emp_values[7], decimals=1))
        fwhm_dm3_1248.append(np.around(fwhm_emp_values[6], decimals=1))
        fwhm_dm3_1020.append(np.around(fwhm_emp_values[5], decimals=1))
        fwhm_dm3_0877.append(np.around(fwhm_emp_values[4], decimals=1))
        fwhm_dm3_0810.append(np.around(fwhm_emp_values[3], decimals=1))
        fwhm_dm3_0652.append(np.around(fwhm_emp_values[2], decimals=1))
        fwhm_dm3_0544.append(np.around(fwhm_emp_values[1], decimals=1))
        fwhm_dm3_0432.append(np.around(fwhm_emp_values[0], decimals=1))

    os.chdir('..')
    os.chdir('..')

    figure, axis = plt.subplots(nrows=2, ncols=2, figsize=(12,12))

    axis[0,0].plot(heights, strehl_dm2_2200, color='#FF0000', label="2200 nm")
    axis[0,0].plot(heights, strehl_dm2_1673, color='#FF7F00', label="1673 nm")
    axis[0,0].plot(heights, strehl_dm2_1248, color='#FFFF00', label="1248 nm")
    axis[0,0].plot(heights, strehl_dm2_1020, color='#7FFF00', label="1020 nm")
    axis[0,0].plot(heights, strehl_dm2_0877, color='#00FF00', label="877 nm")
    axis[0,0].plot(heights, strehl_dm2_0810, color='#00FF7F', label="810 nm")
    axis[0,0].plot(heights, strehl_dm2_0652, color='#00FFFF', label="652 nm")
    axis[0,0].plot(heights, strehl_dm2_0544, color='#007FFF', label="544 nm")
    axis[0,0].plot(heights, strehl_dm2_0432, color='#0000FF', label="432 nm")
    axis[0,0].set_xlabel("Altitude of Conjugate High-Order DM2 (km)")
    axis[0,0].set_ylabel("Strehl Ratio")
    axis[0,0].set_title("Strehl Ratio vs. Altitude of HO DM2 (1000 nm)")

    axis[0,1].plot(heights, strehl_dm3_2200, color='#FF0000', label="2200 nm")
    axis[0,1].plot(heights, strehl_dm3_1673, color='#FF7F00', label="1673 nm")
    axis[0,1].plot(heights, strehl_dm3_1248, color='#FFFF00', label="1248 nm")
    axis[0,1].plot(heights, strehl_dm3_1020, color='#7FFF00', label="1020 nm")
    axis[0,1].plot(heights, strehl_dm3_0877, color='#00FF00', label="877 nm")
    axis[0,1].plot(heights, strehl_dm3_0810, color='#00FF7F', label="810 nm")
    axis[0,1].plot(heights, strehl_dm3_0652, color='#00FFFF', label="652 nm")
    axis[0,1].plot(heights, strehl_dm3_0544, color='#007FFF', label="544 nm")
    axis[0,1].plot(heights, strehl_dm3_0432, color='#0000FF', label="432 nm")
    axis[0,1].set_xlabel("Altitude of Conjugate High-Order DM3 (km)")
    axis[0,1].set_ylabel("Strehl Ratio")
    axis[0,1].set_title("Strehl Ratio vs. Altitude of HO DM3 (1000 nm)")

    axis[1,0].plot(heights, fwhm_dm2_2200, color='#FF0000', label="2200 nm")
    axis[1,0].plot(heights, fwhm_dm2_1673, color='#FF7F00', label="1673 nm")
    axis[1,0].plot(heights, fwhm_dm2_1248, color='#FFFF00', label="1248 nm")
    axis[1,0].plot(heights, fwhm_dm2_1020, color='#7FFF00', label="1020 nm")
    axis[1,0].plot(heights, fwhm_dm2_0877, color='#00FF00', label="877 nm")
    axis[1,0].plot(heights, fwhm_dm2_0810, color='#00FF7F', label="810 nm")
    axis[1,0].plot(heights, fwhm_dm2_0652, color='#00FFFF', label="652 nm")
    axis[1,0].plot(heights, fwhm_dm2_0544, color='#007FFF', label="544 nm")
    axis[1,0].plot(heights, fwhm_dm2_0432, color='#0000FF', label="432 nm")
    axis[1,0].set_xlabel("Altitude of Conjugate High-Order DM2 (km)")
    axis[1,0].set_ylabel("Full-Width at Half Max")
    axis[1,0].set_title("Empirical FWHM vs. Altitude of HO DM2 (1000 nm)")

    axis[1,1].plot(heights, fwhm_dm3_2200, color='#FF0000', label="2200 nm")
    axis[1,1].plot(heights, fwhm_dm3_1673, color='#FF7F00', label="1673 nm")
    axis[1,1].plot(heights, fwhm_dm3_1248, color='#FFFF00', label="1248 nm")
    axis[1,1].plot(heights, fwhm_dm3_1020, color='#7FFF00', label="1020 nm")
    axis[1,1].plot(heights, fwhm_dm3_0877, color='#00FF00', label="877 nm")
    axis[1,1].plot(heights, fwhm_dm3_0810, color='#00FF7F', label="810 nm")
    axis[1,1].plot(heights, fwhm_dm3_0652, color='#00FFFF', label="652 nm")
    axis[1,1].plot(heights, fwhm_dm3_0544, color='#007FFF', label="544 nm")
    axis[1,1].plot(heights, fwhm_dm3_0432, color='#0000FF', label="432 nm")
    axis[1,1].set_xlabel("Altitude of Conjugate High-Order DM3 (km)")
    axis[1,1].set_ylabel("Full-Width at Half Max")
    axis[1,1].set_title("Empirical FWHM vs. Altitude of HO DM3 (1000 nm)")

    axis[1,1].legend(bbox_to_anchor=(-0.15, -0.15), loc="upper center", ncol=3)
    
    plt.savefig('dm_psf.png')
    plt.show()