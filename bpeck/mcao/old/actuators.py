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
from scipy.interpolate import interp1d
from paarti.utils import maos_utils

def calculate_side(act_start, act_stop, act_step, sigfigs=3):
    #For vismcao, to change the amount of actuators on the ASM, the value dm.dx needs to be changed
    #This function calcualtes the value of dm.dx 
    #Feeds into "run" function to run actuator vismcao simulations
    side_count=[]
    actuators = np.arange(act_start, act_stop, act_step)

    print(f"Calculating dm.dx value for keck_dm_single.conf & side for keck_nea_photons:")
    
    for i, actuator in enumerate(actuators, start=1):
        side = 11 / (2*((actuator/np.pi)**0.5))
        rounded_side = np.round(side, sigfigs)

        message = f"For {actuator} actuators: {rounded_side} (m)\n"
        print(message)

        side_count.append(rounded_side)
        
    return side_count

def calculate_total_flux(act_start, act_stop, act_step, int_time, lgsmag, ttmag):
    #For converting magnitude to flux for LGWFS-OCAM2K and LBWFS for changing actuators
    #Relies on calculate_side function for actual equation
    #LGS magnitude is equal to LB magnitude
    lgs_all = []
    truth_all = []
    tt_all = []
    
    actuators = np.arange(act_start, act_stop, act_step)
    
    sides = calculate_side(act_start, act_stop, act_step)

    print(f"Calculating dm.dx value for keck_dm_single.conf & side for keck_nea_photons:")
    
    for side, actuator in zip(sides, actuators):
        print(side)
        flux_lgs = maos_utils.keck_nea_photons_any_config(wfs='LGSWFS-OCAM2K', side=side, throughput=0.36 * 0.88, ps=1.0, theta_beta=1.5 *(math.pi/180)/(60*60), band= "R", sigma_e=0.5, pix_per_ap=25, time=int_time, m=lgsmag)
        print(flux_lgs)
        lgs_all.append(flux_lgs)
        print(side)
        flux_truth = maos_utils.keck_nea_photons_any_config(wfs="LBWFS", side=side, throughput=0.03, ps=1.5, theta_beta=0.49 *(math.pi/180)/(60*60), band="R", sigma_e=7.96, pix_per_ap=4, time=10, m=lgsmag)
        print(flux_truth)
        truth_all.append(flux_truth)
        
        flux_tt = maos_utils.keck_nea_photons(m=ttmag, wfs="STRAP", wfs_int_time=int_time)
        tt_all.append(flux_tt)
    
    return lgs_all, truth_all, tt_all

def calculate_parameters(act_start, act_stop, act_step, int_time, lgsmag, ttmag):
    actuators = np.arange(act_start, act_stop, act_step)
    lgs_all, truth_all, tt_all = calculate_total_flux(act_start, act_stop, act_step, int_time, lgsmag, ttmag)
    side_count = calculate_side(act_start, act_stop, act_step)
    
    lgs_nearecon = []
    lgs_siglev = []
    lgs_bkgrnd = []
    truth_nearecon = []
    truth_siglev = []
    truth_bkgrnd = []
    tt_nearecon = []
    tt_siglev = []
    tt_bkgrnd = []

    for results in lgs_all:
        _, sigma_theta, Np, Nb = results
        lgs_nearecon.append(round(sigma_theta, 3))
        lgs_siglev.append(round(Np, 3))
        lgs_bkgrnd.append(round(Nb, 3))

    for results in truth_all:  
        _, sigma_theta, Np, Nb = results
        truth_nearecon.append(round(sigma_theta, 3))
        truth_siglev.append(round(Np, 3))
        truth_bkgrnd.append(round(Nb, 3))

    for results in tt_all:  
        _, sigma_theta, Np, Nb = results
        tt_nearecon.append(round(sigma_theta, 3))
        tt_siglev.append(round(Np, 3))
        tt_bkgrnd.append(round(Nb, 3))

    for i in range(len(lgs_nearecon)):
        lgs_n = lgs_nearecon[i]
        truth_n = truth_nearecon[i]
        tt_n = tt_nearecon[i]
        lgs_s = lgs_siglev[i]
        truth_s = truth_siglev[i]
        tt_s = tt_siglev[i]
        lgs_b = lgs_bkgrnd[i]
        truth_b = truth_bkgrnd[i]
        tt_b = tt_bkgrnd[i]
        
        actuator = actuators[i]
        side = side_count[i]

        message = f"For {actuator} actuators, flux values:\n" 
        message1 = f"powfs.siglev: [{lgs_s} {tt_s} {truth_s}] (m)\n"
        message2 = f"powfs.bkgrnd: [{lgs_b} {tt_b} {truth_b}] (m)\n"
        message3 = f"powfs.nearecon: [{lgs_n} {tt_n} {truth_n}] (m)\n"
        print(message, message1, message2, message3) 

def plot_psf(act_start, act_stop, act_step, directory_results_path='A_keck_scao_lgs', interpolated=False):
    #Makes psf plots for changing acutators
    actuators = np.arange(act_start, act_stop, act_step)
    psf_all = []

    cwd = os.getcwd()
    print(cwd)
    
    for actuator in actuators:
        directory = f"{directory_results_path}/vis{actuator}a"
        os.chdir(directory)

        psf_metrics = maos_utils.print_psf_metrics_x0y0(oversamp=3, seed=1)
        psf_all.append(psf_metrics)

        cwd = os.getcwd()
        print(cwd)
        
        os.chdir("..")
        os.chdir("..")
    
    strehl = []
    g_fwhm = []
    emp_fwhm = []
    ee80 = []

    strehl_2200 = []
    strehl_1650 = []
    strehl_1250 = []
    strehl_1000 = []
    strehl_0800 = []

    g_fwhm_2200 = []
    g_fwhm_1650 = []
    g_fwhm_1250 = []
    g_fwhm_1000 = []
    g_fwhm_0800 = []

    e_fwhm_2200 = []
    e_fwhm_1650 = []
    e_fwhm_1250 = []
    e_fwhm_1000 = []
    e_fwhm_0800 = []

    r_ee80_2200 = []
    r_ee80_1650 = []
    r_ee80_1250 = []
    r_ee80_1000 = []
    r_ee80_0800 = []

    for psf_metrics in psf_all:
        wavelengths, strehl_values, fwhm_gaus_values, fwhm_emp_values, r_ee80_values = psf_metrics

        strehl.append(np.around(strehl_values, decimals=2))
        g_fwhm.append(np.around(fwhm_gaus_values, decimals=1))
        emp_fwhm.append(np.around(fwhm_emp_values, decimals=1))
        ee80.append(np.around(r_ee80_values, decimals=1))

        strehl_2200.append(np.around(strehl_values[4], decimals=2))
        strehl_1650.append(np.around(strehl_values[3], decimals=2))
        strehl_1250.append(np.around(strehl_values[2], decimals=2))
        strehl_1000.append(np.around(strehl_values[1], decimals=2))
        strehl_0800.append(np.around(strehl_values[0], decimals=2))

        g_fwhm_2200.append(np.around(fwhm_gaus_values[4], decimals=1))
        g_fwhm_1650.append(np.around(fwhm_gaus_values[3], decimals=1))
        g_fwhm_1250.append(np.around(fwhm_gaus_values[2], decimals=1))
        g_fwhm_1000.append(np.around(fwhm_gaus_values[1], decimals=1))
        g_fwhm_0800.append(np.around(fwhm_gaus_values[0], decimals=1))

        e_fwhm_2200.append(np.around(fwhm_emp_values[4], decimals=1))
        e_fwhm_1650.append(np.around(fwhm_emp_values[3], decimals=1))
        e_fwhm_1250.append(np.around(fwhm_emp_values[2], decimals=1))
        e_fwhm_1000.append(np.around(fwhm_emp_values[1], decimals=1))
        e_fwhm_0800.append(np.around(fwhm_emp_values[0], decimals=1))

        r_ee80_2200.append(np.around(r_ee80_values[4], decimals=1))
        r_ee80_1650.append(np.around(r_ee80_values[3], decimals=1))
        r_ee80_1250.append(np.around(r_ee80_values[2], decimals=1))
        r_ee80_1000.append(np.around(r_ee80_values[1], decimals=1))
        r_ee80_0800.append(np.around(r_ee80_values[0], decimals=1))

    act_int = np.linspace(min(actuators), max(actuators), 100)

    strehl_int_2200 = interp1d(actuators, strehl_2200, kind='cubic')
    strehl_int_1650 = interp1d(actuators, strehl_1650, kind='cubic')
    strehl_int_1250 = interp1d(actuators, strehl_1250, kind='cubic')
    strehl_int_1000 = interp1d(actuators, strehl_1000, kind='cubic')
    strehl_int_0800 = interp1d(actuators, strehl_0800, kind='cubic')

    g_fwhm_int_2200 = interp1d(actuators, g_fwhm_2200, kind='cubic')
    g_fwhm_int_1650 = interp1d(actuators, g_fwhm_1650, kind='cubic')
    g_fwhm_int_1250 = interp1d(actuators, g_fwhm_1250, kind='cubic')
    g_fwhm_int_1000 = interp1d(actuators, g_fwhm_1000, kind='cubic')
    g_fwhm_int_0800 = interp1d(actuators, g_fwhm_0800, kind='cubic')

    e_fwhm_int_2200 = interp1d(actuators, e_fwhm_2200, kind='cubic')
    e_fwhm_int_1650 = interp1d(actuators, e_fwhm_1650, kind='cubic')
    e_fwhm_int_1250 = interp1d(actuators, e_fwhm_1250, kind='cubic')
    e_fwhm_int_1000 = interp1d(actuators, e_fwhm_1000, kind='cubic')
    e_fwhm_int_0800 = interp1d(actuators, e_fwhm_0800, kind='cubic')

    r_ee80_int_2200 = interp1d(actuators, r_ee80_2200, kind='cubic')
    r_ee80_int_1650 = interp1d(actuators, r_ee80_1650, kind='cubic')
    r_ee80_int_1250 = interp1d(actuators, r_ee80_1250, kind='cubic')
    r_ee80_int_1000 = interp1d(actuators, r_ee80_1000, kind='cubic')
    r_ee80_int_0800 = interp1d(actuators, r_ee80_0800, kind='cubic')

    if interpolated:
        figure, axis = plt.subplots(nrows=1, ncols=4, figsize=(16,4))
        
        axis[0].plot(act_int, strehl_int_2200(act_int), color="red", label="2.200 um")
        axis[0].plot(act_int, strehl_int_1650(act_int), color="orange", label="1.650 um")
        axis[0].plot(act_int, strehl_int_1250(act_int), color="green", label="1.250 um")
        axis[0].plot(act_int, strehl_int_1000(act_int), color="cyan", label= "1.000 um")
        axis[0].plot(act_int, strehl_int_0800(act_int), color="navy", label="0.800 um")
        axis[0].set_xlabel("Actuators on Adaptive Secondary Mirror")
        axis[0].set_ylabel("Strehl Values")
        axis[0].set_title("Strehl vs. Actuators")

        axis[1].plot(act_int, g_fwhm_int_2200(act_int), color="red", label="2.200 um")
        axis[1].plot(act_int, g_fwhm_int_1650(act_int), color="orange", label="1.650 um")
        axis[1].plot(act_int, g_fwhm_int_1250(act_int), color="green", label="1.250 um")
        axis[1].plot(act_int, g_fwhm_int_1000(act_int), color="cyan", label="1.000 um")
        axis[1].plot(act_int, g_fwhm_int_0800(act_int), color="navy", label="0.800 um")
        axis[1].set_xlabel("Actuators on Adaptive Secondary Mirror") 
        axis[1].set_ylabel("Gauss Full-Width Half Max (mas)")
        axis[1].set_title("Gauss FWHM vs. Actuators")

        axis[2].plot(act_int, e_fwhm_int_2200(act_int), color="red", label="2.200 um")
        axis[2].plot(act_int, e_fwhm_int_1650(act_int), color="orange", label="1.650 um")
        axis[2].plot(act_int, e_fwhm_int_1250(act_int), color="green", label="1.250 um")
        axis[2].plot(act_int, e_fwhm_int_1000(act_int), color="cyan", label="1.000 um")
        axis[2].plot(act_int, e_fwhm_int_0800(act_int), color="navy", label="0.800 um")
        axis[2].set_xlabel("Actuators on Adaptive Secondary Mirror")
        axis[2].set_ylabel("Empirical Full-Width Half Max (mas)")
        axis[2].set_title("Empirical FWHM vs. Actuators")

        axis[3].plot(act_int, r_ee80_int_2200(act_int), color="red", label="2.200 um")
        axis[3].plot(act_int, r_ee80_int_1650(act_int), color="orange", label="1.650 um")
        axis[3].plot(act_int, r_ee80_int_1250(act_int), color="green", label="1.250 um")
        axis[3].plot(act_int, r_ee80_int_1000(act_int), color="cyan", label="1.000 um")
        axis[3].plot(act_int, r_ee80_int_0800(act_int), color="navy", label="0.800 um")
        axis[3].set_xlabel("Actuators on Adaptive Secondary Mirror")
        axis[3].set_ylabel("Radius of Encircled Energy at 80% (mas)")
        axis[3].set_title("Radius of EE80 vs. Actuators")
        
        handles, labels = axis[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(labels))
        plt.subplots_adjust(bottom=0.2)
 
    else:
        figure, axis = plt.subplots(nrows=1, ncols=4, figsize=(16,4))
        
        axis[0].plot(actuators, strehl_2200, color="red", label="2.200 um")
        axis[0].plot(actuators, strehl_1650, color="blue", label="1.650 um")
        axis[0].plot(actuators, strehl_1250, color="green", label="1.250 um")
        axis[0].plot(actuators, strehl_1000, color="orange", label= "1.000 um")
        axis[0].plot(actuators, strehl_0800, color="purple", label="0.800 um")
        axis[0].set_xlabel("Actuators on Adaptive Secondary Mirror")
        axis[0].set_ylabel("Strehl Values")
        axis[0].set_title("Strehl vs. Actuators")

        axis[1].plot(actuators, g_fwhm_2200, color="red", label="2.200 um")
        axis[1].plot(actuators, g_fwhm_1650, color="blue", label="1.650 um")
        axis[1].plot(actuators, g_fwhm_1250, color="green", label="1.250 um")
        axis[1].plot(actuators, g_fwhm_1000, color="orange", label="1.000 um")
        axis[1].plot(actuators, g_fwhm_0800, color="purple", label="0.800 um")
        axis[1].set_xlabel("Actuators on Adaptive Secondary Mirror")
        axis[1].set_ylabel("Gaussian Full-Width Half Max (mas)")
        axis[1].set_title("Gaussian FWHM vs. Actuators")

        axis[2].plot(actuators, e_fwhm_2200, color="red", label="2.200 um")
        axis[2].plot(actuators, e_fwhm_1650, color="blue", label="1.650 um")
        axis[2].plot(actuators, e_fwhm_1250, color="green", label="1.250 um")
        axis[2].plot(actuators, e_fwhm_1000, color="orange", label="1.000 um")
        axis[2].plot(actuators, e_fwhm_0800, color="purple", label="0.800 um")
        axis[2].set_xlabel("Actuators on Adaptive Secondary Mirror")
        axis[2].set_ylabel("Empirical Full-Width Half Max (mas)")
        axis[2].set_title("Empirical FWHM vs. Actuators")

        axis[3].plot(actuators, r_ee80_2200, color="red", label="2.200 um")
        axis[3].plot(actuators, r_ee80_1650, color="blue", label="1.650 um")
        axis[3].plot(actuators, r_ee80_1250, color="green", label="1.250 um")
        axis[3].plot(actuators, r_ee80_1000, color="orange", label="1.000 um")
        axis[3].plot(actuators, r_ee80_0800, color="purple", label="0.800 um")
        axis[3].set_xlabel("Actuators on Adaptive Secondary Mirror")
        axis[3].set_ylabel("Radius of Encircled Energy at 80% (mas)")
        axis[3].set_title("Radius of EE80 vs. Actuators")

        handles, labels = axis[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(labels))
        plt.subplots_adjust(bottom=0.2)

    plt.tight_layout()
    os.chdir('..')
    plt.savefig('vis_actuators_psf.png')
    plt.show()
    
    cwd = os.getcwd()
    print(f"Current working directory:", cwd)

def plot_wfe(mag_start, mag_stop, mag_step, directory_results_path='A_keck_scao_lgs', interpolated=False):
    actuators = np.arange(act_start, act_stop, act_step)

    directory = f"{directory_results_path}/vis{actuators[0]}a"
    os.chdir(directory)
    
    vis_directories = [f"vis{actuator}a" for actuator in actuators]

    cwd = os.getcwd()
    print(cwd)
    
    wfe_all = []
    for vis_directory in vis_directories:

        wfe_metrics = maos_utils.print_wfe_metrics(seed=1)
        wfe_all.append(wfe_metrics)

        os.chdir("..")

        next_directory = vis_directory
        os.chdir(next_directory)

        cwd = os.getcwd()
        print(cwd)

    open_wfe = []
    closed_wfe = []

    wfe_open_total = []
    wfe_open_high_order = []
    wfe_open_tt = []

    wfe_closed_total = []
    wfe_closed_high_order = []
    wfe_closed_tt = []

    for wfe_metrics in wfe_all:
        open_mean_nm, clos_mean_nm = wfe_metrics

        wfe_open_total.append(np.around(open_mean_nm[0], decimals=1))
        wfe_open_high_order.append(np.around(open_mean_nm[1], decimals=1))
        wfe_open_tt.append(np.around(open_mean_nm[2], decimals=1))

        wfe_closed_total.append(np.around(clos_mean_nm[0], decimals=1))
        wfe_closed_high_order.append(np.around(clos_mean_nm[2], decimals=1))
        wfe_closed_tt.append(np.around(clos_mean_nm[1], decimals=1))

    act_int = np.linspace(min(actuators), max(actuators), 100)

    wfe_closed_total_int = interp1d(actuators, wfe_closed_total, kind='cubic')
    wfe_closed_high_order_int = interp1d(actuators, wfe_closed_high_order, kind='cubic')
    wfe_closed_tt_int = interp1d(actuators, wfe_closed_tt, kind='cubic')

    if interpolated:
        figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
        
        axis.plot(act_int, wfe_closed_total_int(act_int), color="red", label="Total")
        axis.plot(act_int, wfe_closed_high_order_int(act_int), color="blue", label="High-Order")
        axis.plot(act_int, wfe_closed_tt_int(act_int), color="green", label="Tip-Tilt")
        axis.set_xlabel("Actuators on Adaptive Secondary Mirror")
        axis.set_ylabel("WFE (nm)")
        axis.set_title("WFE vs. Actuators")
        axis.legend(fontsize="small")
          
    else:
        figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
        
        axis.plot(actuators, wfe_closed_total, color="red", label="Total")
        axis.plot(actuators, wfe_closed_high_order, color="blue", label="High-Order")
        axis.plot(actuators, wfe_closed_tt, color="green", label="Tip-Tilt")
        axis.set_xlabel("Actuators on Adaptive Secondary Mirror")
        axis.set_ylabel("WFE (nm)")
        axis.set_title("WFE vs. Actuators")
        axis.legend(fontsize="small")

    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    os.chdir('..')
    os.chdir('..')
    os.chdir('..')
    plt.savefig('vis_actuators_wfe.png')
    plt.show()
    
    cwd = os.getcwd()
    print(f"Current working directory:", cwd)