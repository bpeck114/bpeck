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

def run_sims(act_start, act_stop, act_step, mag):
    actuators = np.arange(act_start, act_stop, act_step)

    for actuator in actuators:
        home_directory = os.path.expanduser('~')
        os.chdir(home_directory)

        working_directory = '/u/bpeck/work/mcao/experiments/act_study/vismcao'
        os.chdir(working_directory)

        cwd = os.getcwd()
        print('Current Working Directory:', cwd)
        
        master_file = f"A_keck_scao_lgs_{mag}mag_{actuator}a.conf"
        print(master_file)

        output_folder = f"A_keck_scao_lgs/{mag}mag/{actuator}a"
        print(output_folder)
        
        maos_command = ["maos", "-o", "A_keck_scao_lgs", "-c", master_file, "plot.all=1", "plot.setup=1", "-O"]
        subprocess.run(maos_command)

        

def calculate_side(act_start, act_stop, act_step, sigfigs=3):
    '''
    - Calculate the distance (in meters) between actuators 
        on the primary mirror.
    - Known as the parameter dm.dx in MAOS.
    - Paramter is needed for flux conversions in PAARTI 
        (keck_nea_photons_any_config under maos_utils)
    '''
    
    sides = []
    actuators = []
    
    amount_actuators = np.arange(act_start, act_stop, act_step)

    ####
    # Loops over list of actuators (2000, 3000, 4000, 5000) and 
    # calculates side value. Appends to a new list.
    ####
    for i, actuator_count in enumerate(amount_actuators, start=1):
        side = 11 / (2 *((actuator_count/np.pi)**0.5))
        rounded_side = np.round(side, sigfigs)

        sides.append(rounded_side)
        actuators.append(actuator_count)

    return sides, actuators

def print_side (act_start, act_stop, act_step, sigfigs=3):
    '''
    - Prints the distance (in meters) between actuators 
        on the primary mirror.
    '''

    sides, actuators = calculate_side(act_start, act_stop, 
                                      act_step, sigfigs=sigfigs)

    print('--------------------')
    print('Distance between actuators relative to primary mirror (m), dm.dx:')
    print('--------------------')

    for side_value, actuator_value in zip(sides, actuators):
        print(f'{actuator_value:11.0f} {side_value:11.3f} \n')

    return

def paarti_mag_to_flux(act_start, act_stop, act_step, 
                                lgs_mag, tt_mag, lgs_bkgrnd_value =0.1, lgs_pixpsa=25, 
                                lgs_pixsize=1, int_time=1/1500):
    '''
    - Convert magnitude values for laser guide star and natural 
        guide start to flux values that MAOS will understand.
    - From PAARTI, use keck_nea_photons_any_config for LGS and 
        LBWFS while keck_nea_photons for tt
    - Works with powfs.siglev, powfs.bkgrnd and powfs.nearecon
    '''

    lgs = []
    truth = []
    tt = []

    sides, actuators = calculate_side(act_start, act_stop, act_step)
                                    
    ####
    # For each dm.dx value and actuator count, calulates magnitude to flux values
    # with PAARTI package.
    ####

    for side_value, actuator_value in zip(sides, actuators):
        lgs_flux = maos_utils.keck_nea_photons_any_config(wfs='LGSWFS-OCAM2K', 
                                                          side = side_value, 
                                                          throughput = 0.36 * 0.88, 
                                                          ps = lgs_pixsize, 
                                                          theta_beta = 1.5 *(math.pi/180)/(60*60), 
                                                          band = "R", sigma_e=0.5, 
                                                          pix_per_ap=lgs_pixpsa, 
                                                          time=int_time, 
                                                          m = lgs_mag)
        tt_flux = maos_utils.keck_nea_photons(m=tt_mag, wfs="STRAP", wfs_int_time=int_time)
        truth_flux = maos_utils.keck_nea_photons_any_config(wfs = 'LBWFS',
                                                            side = 0.563,
                                                            throughput=0.03,
                                                            ps = 1.5,
                                                            theta_beta = 0.49 * (math.pi/180)/(60*60),
                                                            band = "R",
                                                            sigma_e = 7.96,
                                                            pix_per_ap = 4,
                                                            time = 1000000/1500,
                                                            m = tt_mag)
        
        lgs.append(lgs_flux)
        tt.append(tt_flux)
        truth.append(truth_flux)
                                                                
    return lgs, tt, truth, actuators, sides

def calculate_magnitude_to_flux(act_start, act_stop, act_step, 
                                lgs_mag, tt_mag, lgs_bkgrnd_value =0.1, lgs_pixpsa=25, 
                                lgs_pixsize=1, int_time=1/1500):
    
    '''
    - Prints flux values for laser guide star and natural 
        from magnitude inputs. 
    '''
    
    lgs_siglev = []
    lgs_bkgrnd = []
    lgs_nearecon = []

    tt_siglev = []
    tt_bkgrnd = []
    tt_nearecon = []

    truth_siglev = []
    truth_bkgrnd = []
    truth_nearecon = []                                
                                    
    lgs, tt, truth, actuators, sides = paarti_mag_to_flux(act_start, act_stop,
                                                act_step, lgs_mag,
                                                tt_mag, lgs_bkgrnd_value=0.1,lgs_pixpsa=lgs_pixpsa, 
                                        lgs_pixsize=lgs_pixsize, int_time=int_time)
    for lgs_parameter in lgs:
        _, sigma_theta, Np, Nb = lgs_parameter
        lgs_nearecon.append(round(sigma_theta, 3))
        lgs_siglev.append(round(Np, 3))
        lgs_bkgrnd.append(lgs_bkgrnd_value)

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

    return lgs_siglev, lgs_bkgrnd, lgs_nearecon, tt_siglev, tt_bkgrnd, tt_nearecon, truth_siglev, truth_bkgrnd, truth_nearecon, actuators, sides


                                    

def print_magnitude_to_flux(act_start, act_stop, act_step, 
                                lgs_mag, tt_mag, lgs_bkgrnd_value=0.1, lgs_pixpsa=25, 
                                lgs_pixsize=1, int_time=1/1500):

    lgs_siglev, lgs_bkgrnd, lgs_nearecon, tt_siglev, tt_bkgrnd, tt_nearecon, truth_siglev, truth_bkgrnd, truth_nearecon, actuators, sides = calculate_magnitude_to_flux(act_start, act_stop, act_step, lgs_mag, tt_mag, lgs_pixpsa=lgs_pixpsa, lgs_pixsize=lgs_pixsize, int_time=int_time)


    print('--------------------')
    print('MCAO Flux Parameters:')
    print('--------------------')
    print('')
    print('####')
    print(f'#{lgs_mag}mag')
    print('####')

    for i, actuator in enumerate(actuators):
        print('#Actuator Count:', actuator)
        print('#dm.dx = [', sides[i], '.168 .168 ]')
        print('#powfs.siglev = [',lgs_siglev[i], tt_siglev[i], truth_siglev[i], ']')
        print('#powfs.bkgrnd = [', lgs_bkgrnd[i], tt_bkgrnd[i], truth_bkgrnd[i], ']')
        print('#powfs.nearecon = [', lgs_nearecon[i], tt_nearecon[i], truth_nearecon[i], ']')
        print('')
    
    return

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

def plot_psf_wvls(act_start, act_stop, act_step, mag, directory_results_path='A_keck_scao_lgs'):
    '''
    Plots the strehl ratio and fwhm of the actuat
    '''
    psf_metrics = []
    actuators = np.arange(act_start, act_stop, act_step)

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

    for actuator in actuators:
        directory = f"{directory_results_path}/{mag}mag/{actuator}a"
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
        
    axis[0].plot(actuators, strehl_2200, color='#FF0000', label="2200 nm")
    axis[0].plot(actuators, strehl_1673, color='#FF7F00', label="1673 nm")
    axis[0].plot(actuators, strehl_1248, color='#FFFF00', label="1248 nm")
    axis[0].plot(actuators, strehl_1020, color='#7FFF00', label="1020 nm")
    axis[0].plot(actuators, strehl_0877, color='#00FF00', label="877 nm")
    axis[0].plot(actuators, strehl_0810, color='#00FF7F', label="810 nm")
    axis[0].plot(actuators, strehl_0652, color='#00FFFF', label="652 nm")
    axis[0].plot(actuators, strehl_0544, color='#007FFF', label="544 nm")
    axis[0].plot(actuators, strehl_0432, color='#0000FF', label="432 nm")
    axis[0].set_xlabel("Actuator Count on ASM (n)")
    axis[0].set_ylabel("Strehl Ratio")
    axis[0].set_title("Strehl Ratio vs. Actuator Count (7 mag LGS)")

    axis[1].plot(actuators, fwhm_2200, color='#FF0000', label="2200 nm")
    axis[1].plot(actuators, fwhm_1673, color='#FF7F00', label="1673 nm")
    axis[1].plot(actuators, fwhm_1248, color='#FFFF00', label="1248 nm")
    axis[1].plot(actuators, fwhm_1020, color='#7FFF00', label="1020 nm")
    axis[1].plot(actuators, fwhm_0877, color='#00FF00', label="877 nm")
    axis[1].plot(actuators, fwhm_0810, color='#00FF7F', label="810 nm")
    axis[1].plot(actuators, fwhm_0652, color='#00FFFF', label="652 nm")
    axis[1].plot(actuators, fwhm_0544, color='#007FFF', label="544 nm")
    axis[1].plot(actuators, fwhm_0432, color='#0000FF', label="432 nm")
    #axis[1].legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3)
    axis[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)
    axis[1].set_xlabel("Actuator Count on ASM (n)")
    axis[1].set_ylabel("Full-Width Half Max (mas)")
    axis[1].set_title("Empirical FWHM vs. Actuator Count (7 mag LGS)")

    plt.tight_layout()
    plt.savefig('act_study_psf_wvl.png')
    plt.show()
    
    cwd = os.getcwd()
    print(f"Current working directory:", cwd)

def plot_psf(act_start, act_stop, act_step, magnitudes, bandwidth=5, directory_results_path='A_keck_scao_lgs'):
    strehl_ = {}
    fwhm_ = {}

    actuators = np.arange(act_start, act_stop, act_step)

    for mag in magnitudes:
        strehl_[mag] = []
        fwhm_[mag] = []

        mag_directory = f"{directory_results_path}/{mag}mag"
        os.chdir(mag_directory)

        for actuator in actuators:
            act_directory = f"{actuator}a"
            print(f"Directory: {mag}mag/{actuator}a")
            os.chdir(act_directory)

            psf_metrics = maos_utils.print_psf_metrics_x0y0(oversamp=3, seed=1)
            wavelengths, strehl_values, fwhm_gaus_values, fwhm_emp_values, r_ee80_values = psf_metrics

            strehl_[mag].append(np.round(strehl_values[bandwidth], decimals=2))
            fwhm_[mag].append(np.round(fwhm_emp_values[bandwidth], decimals=1))

            os.chdir('..')
            
        print(f"Magnitude: {mag}, Strehl: {strehl_[mag]}")
        print(f"Magnitude:{mag}, Strehl: {fwhm_[mag]}")
        
        os.chdir('..')
        os.chdir('..')

    figure, axis = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
    
    for mag, color in zip(magnitudes, ['red', 'orange', 'green', 'blue']):
        axis[0].plot(actuators, strehl_[mag], color=color, label=f"{mag} magnitude")
        axis[1].plot(actuators, fwhm_[mag], color=color, label=f"{mag} magnitude")

    for ax in axis:
        ax.set_xlabel("Actuator Count on ASM (n)")

    axis[0].set_ylabel("Strehl Ratio")
    axis[1].set_ylabel("Full-Width at Half Max (mas)")

    axis[0].set_title("Strehl Ratio vs. Actuator Count (1000 nm)")
    axis[1].set_title("Empirical FWHM vs. Actuator Count (1000 nm)")

    axis[1].legend(bbox_to_anchor=(-.1, -0.15), loc="upper center", ncol=4)
    plt.savefig('act_study_psf.png')
    plt.show()


def plot_wfe(mag_start, mag_stop, mag_step, sigrecons, directory_results_path='A_keck_scao_lgs'):
    total_ = {}
    tt_ = {}
    ho_ = {}
    
    magnitudes = np.arange(mag_start, mag_stop, mag_step)

    for sigrecon in sigrecons:
        total_[sigrecon] = []
        tt_[sigrecon] = []
        ho_[sigrecon] = []
        
        sigrecon_directory = f"{directory_results_path}/{sigrecon}sigrecon"
        print(sigrecon_directory)
        os.chdir(sigrecon_directory)

        for mag in magnitudes:
            mag_directory = f"{mag}mag"
            os.chdir(mag_directory)

            wfe_metrics = maos_utils.print_wfe_metrics(seed=1)
            open_mean_nm, clos_mean_nm = wfe_metrics

            total_[sigrecon].append(np.round(clos_mean_nm[0], decimals=1))
            tt_[sigrecon].append(np.round(clos_mean_nm[1], decimals=1))
            ho_[sigrecon].append(np.round(clos_mean_nm[2], decimals=1))
            
            os.chdir('..')
            
        os.chdir('..')
        os.chdir('..')

    figure, axis = plt.subplots(nrows=1, ncols=3, figsize=(16,4), sharey=True)
    
    for sigrecon, color in zip(sigrecons, ['red', 'orange', 'green', 'blue']):
        print(sigrecons, sigrecon, color)
        axis[0].plot(magnitudes, total_[sigrecon], color=color, label=f"1/{sigrecon} sigrecon")
        axis[1].plot(magnitudes, tt_[sigrecon], color=color, label=f"1/{sigrecon} sigrecon")
        axis[2].plot(magnitudes, ho_[sigrecon], color=color, label=f"1/{sigrecon} sigrecon")

    for ax in axis:
        ax.set_xlabel("Magnitude of LGS Constellation (mag)")
    
    axis[0].set_ylabel("Total Wave-front Error (nm)")
    axis[1].set_ylabel("Tip-Tilt Wave-front Error (nm)")
    axis[2].set_ylabel("High-Order Wave-front Error (nm)")

    axis[0].tick_params(left=True, labelleft=True)
    axis[1].tick_params(left=True, labelleft=True)
    axis[2].tick_params(left=True, labelleft=True)

    axis[0].set_title("Total WFE vs. LGS Magntiude")
    axis[1].set_title("Tip-Tilt WFE vs. LGS Magnitude")
    axis[2].set_title("High-Order WFE vs. LGS Magnitude")

    axis[1].legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=4)
    plt.savefig('act_study_wfe.png')
    plt.show()