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

"""
Constants for VisAO (Last Updated: 4/20/2024)
Refer to PAARTI for further documentation.
"""
#LGS WFS PAARTI Constants
LGS_WFS               = 'LGSWFS-OCAM2K'             #Wave Front Sensor
LGS_THROUGHPUT        = 0.36 * 0.88                 #Throughput (w QE)
LGS_PIXEL_SIZE        = 1                           #Plate Scale (w QE)
LGS_THETA_BETA        = 1.5 *(math.pi/180)/(60*60)  #Spot Size Diameter
LGS_BAND              = 'R'                         #Filter
LGS_SIGMA_E           = 0.5                         #Read Noise
LGS_PIXPERSA          = 25                          #Pixels Per Subaperture
LGS_INTEGRATION       = 1/1500                      #Integration Time

#LGS WFS MAOS Constants 
LGS_BKGRND            = 0.1 #powfs.bkgrnd

#TT WFS PAARTI Constants
TT_WFS                = 'TRICK-H'
TT_INTEGRATION        = 1/1500

#LBWFS Constants
LBWFS_WFS             = 'LBWFS'
LBWFS_INTEGRATION     = 1/1500 #Needs to happen infrequently

#Output directory for psf metrics and WFE
OUTPUT_DIRECTORY = '/u/bpeck/work/mcao/experiments/act_study/vismcao/A_keck_mcao_lgs/'


def calc_side(act_start, act_stop, act_step, sigfigs=3):
    '''
    Calculate the distance (in meters) between actuators 
    on the primary mirror for Keck. Known as dm.dx in MAOS, needed for 
    updated keck_nea_photons_any_config. Assumes 11 meter diameter
    telescope. 

    Inputs:
    -------
        act_start(float): Least number of actuators
        act_stop(float) : Most number of actuators, exclusive
        act_step(float) : Steps between each calculation
        sigfigs(float)  : Number of sigfigs to be included (optional)
    Outputs:
    --------
        sides(list)     : Calculated dm.dx value
        actuators(list): Associated actuator values
    '''
    # Generate list actuators
    actuators = list(np.arange(act_start, act_stop, act_step))

    # Calculate the side length per suberapture based on actuator count
    sides = [np.round(11 / (2 *((actuator_count/np.pi)**0.5)), sigfigs)
             for actuator_count in actuators]
    return sides, actuators

def print_calc_side(act_start, act_stop, act_step, sigfigs=3):
    '''
    Prints the distance (in meters) between actuators 
    on the primary mirror.
    Inputs:
    -------
        act_start(float): Least number of actuators
        act_stop(float) : Most number of actuators, exclusive
        act_step(float) : Steps between each calculation
        sigfigs(float)  : Number of sigfigs to be included (default = 3)
    Outputs:
    --------
        sides(list)     : Calculated dm.dx value
        actuators(list): Associated actuator values
    '''

    #Define side of subaperature and associated actuator count
    sides, actuators = calc_side(act_start, act_stop, act_step, sigfigs=sigfigs)

    # Print header
    print('--------------------')
    print('Distance between actuators relative to primary mirror (m), dm.dx:')
    print('--------------------')

    # Print formatting
    for side_value, actuator_value in zip(sides, actuators):
        print(f'{actuator_value:11.0f} {side_value:11.3f} \n')

    print(sides)

    return sides, actuators

def lgs_mag_to_flux(act_start, act_stop, act_step, lgs_mag, tt_mag):
    '''
    Converts magnitude to flux for LGS WFS.
    Inputs:
    -------
        act_start(float)          : Least number of actuators
        act_stop(float)           : Most number of actuators, exclusive
        act_step(float)           : Steps between each calculation
        lgs_mag(float)            : Magnitude of laser guide stars
        tt_mag(float)             : Magnitude of tip-tilt guide stars
    Outputs:
    --------
        lgs_flux_values(list)     : Laser Guide Star PAARTI FLux Values
    '''
    
    sides, actuators = calc_side(act_start, act_stop, act_step)
    lgs_flux_values = [maos_utils.keck_nea_photons_any_config(wfs=LGS_WFS, 
                                              side=side_value, 
                                              throughput=LGS_THROUGHPUT, 
                                              ps=LGS_PIXEL_SIZE, 
                                              theta_beta=LGS_THETA_BETA, 
                                              band=LGS_BAND, 
                                              sigma_e=LGS_SIGMA_E, 
                                              pix_per_ap=LGS_PIXPERSA, 
                                              time=LGS_INTEGRATION, 
                                              m=lgs_mag)
           for side_value in sides]
    return lgs_flux_values

def tt_mag_to_flux(act_start, act_stop, act_step, lgs_mag, tt_mag):
    '''
    Converts magnitude to flux for TT WFS.
    Inputs:
    -------
        act_start(float)          : Least number of actuators
        act_stop(float)           : Most number of actuators, exclusive
        act_step(float)           : Steps between each calculation
        lgs_mag(float)            : Magnitude of laser guide stars
        tt_mag(float)             : Magnitude of tip-tilt guide stars
    Outputs:
    --------
        tt_flux_values(list)      : Laser Guide Star PAARTI FLux Values
    '''
    sides, actuators = calc_side(act_start, act_stop, act_step)
    tt_flux_values = [maos_utils.keck_nea_photons(m=tt_mag,
                                      wfs=TT_WFS,
                                      wfs_int_time=TT_INTEGRATION)
          for _ in sides]

    return tt_flux_values

def truth_mag_to_flux(act_start, act_stop, act_step, lgs_mag, tt_mag):
    '''
    Converts magnitude to flux for LBWFS (truth) WFS.
    Inputs:
    -------
        act_start(float)          : Least number of actuators
        act_stop(float)           : Most number of actuators, exclusive
        act_step(float)           : Steps between each calculation
        lgs_mag(float)            : Magnitude of laser guide stars
        tt_mag(float)             : Magnitude of tip-tilt guide stars
    Outputs:
    --------
        truth_flux_values(list)   : Laser Guide Star PAARTI FLux Values
    '''
    sides, actuators = calc_side(act_start, act_stop, act_step)
    truth_flux_values = [maos_utils.keck_nea_photons(m=tt_mag,
                                        wfs=LBWFS_WFS,
                                        wfs_int_time=LBWFS_INTEGRATION)
             for _ in sides]
    
    return truth_flux_values

def lgs_flux_params(act_start, act_stop, act_step, lgs_mag, tt_mag):
    '''
    Takes PAARTI flux values and converts them to LGS parameters
    that can be called upon in MAOS: powfs.siglev,
    powfs.bkgrnd and powfs.nearecon.
    
    Inputs:
    -------
        act_start(float)          : Least number of actuators
        act_stop(float)           : Most number of actuators, exclusive
        act_step(float)           : Steps between each calculation
        lgs_mag(float)            : Magnitude of laser guide stars
        tt_mag(float)             : Magnitude of tip-tilt guide stars
    Outputs
    --------
        lgs_siglev(list)          : Laser Guide Star powfs.siglev
        lgs_bkgrnd(list)          : Laser Guide Star powfs.bkgrnd
        lgs_nearecon(list)        : Laser Guide Star powfs.nearecon
    '''
    lgs_flux_values = lgs_mag_to_flux(act_start, act_stop, act_step, lgs_mag, tt_mag)
    lgs_siglev, lgs_bkgrnd, lgs_nearecon = zip(*[(round(lgs_parameter[2], 3),
                                                  LGS_BKGRND,
                                                  round(lgs_parameter[1], 3))
                                                 for lgs_parameter in lgs_flux_values])
    return (list(lgs_siglev), list(lgs_bkgrnd), list(lgs_nearecon))

def tt_flux_params(act_start, act_stop, act_step, lgs_mag, tt_mag):
    '''
    Takes PAARTI flux values and converts them to TT parameters
    that can be called upon in MAOS: powfs.siglev,
    powfs.bkgrnd and powfs.nearecon.
    
    Inputs:
    -------
        act_start(float)          : Least number of actuators
        act_stop(float)           : Most number of actuators, exclusive
        act_step(float)           : Steps between each calculation
        lgs_mag(float)            : Magnitude of laser guide stars
        tt_mag(float)             : Magnitude of tip-tilt guide stars
    Outputs
    --------
        tt_siglev(list)           : Tip-Tilt Star powfs.siglev
        tt_bkgrnd(list)           : Tip-Tilt Star powfs.bkgrnd
        tt_nearecon(list)         : Tip-Tilt Star powfs.nearecon
    '''
    tt_flux_values = tt_mag_to_flux(act_start, act_stop, act_step, lgs_mag, tt_mag)
    tt_siglev, tt_bkgrnd, tt_nearecon = zip(*[(round(tt_parameter[2], 3),
                                                round(tt_parameter[3], 3),
                                                round(tt_parameter[1], 3))
                                               for tt_parameter in tt_flux_values])
    return list(tt_siglev), list(tt_bkgrnd), list(tt_nearecon)

def truth_flux_params(act_start, act_stop, act_step, lgs_mag, tt_mag):
    '''
    Takes PAARTI flux values and converts them to LBWFS parameters
    that can be called upon in MAOS: powfs.siglev,
    powfs.bkgrnd and powfs.nearecon.
    
    Inputs:
    -------
        act_start(float)          : Least number of actuators
        act_stop(float)           : Most number of actuators, exclusive
        act_step(float)           : Steps between each calculation
        lgs_mag(float)            : Magnitude of laser guide stars
        tt_mag(float)             : Magnitude of tip-tilt guide stars
    Outputs
    --------
        truth_siglev(list)        : LBWFS powfs.siglev
        truth_bkgrnd(list)        : LBWFS powfs.bkgrnd
        truth_nearecon(list)      : LBWFS powfs.nearecon
       
    '''
    truth_flux_values = truth_mag_to_flux(act_start, act_stop, act_step, lgs_mag, tt_mag)
    truth_siglev, truth_bkgrnd, truth_nearecon = zip(*[(round(truth_parameter[2], 3),
                                                        round(truth_parameter[3], 3),
                                                        round(truth_parameter[1], 3))
                                                       for truth_parameter in truth_flux_values])
    return list(truth_siglev), list(truth_bkgrnd), list(truth_nearecon)
                           
def print_mag_to_flux(act_start, act_stop, act_step, 
                                lgs_mag, tt_mag):
    '''
    Converts magnitude to flux for MAOS. Makes use of wrapper
    package PAARTI to convert values. Ready for copy and paste into 
    master file for MAOS runs. 
    
    Inputs:
    -------
        act_start(float)          : Least number of actuators
        act_stop(float)           : Most number of actuators, exclusive
        act_step(float)           : Steps between each calculation
        lgs_mag(float)            : Magnitude of laser guide stars
        tt_mag(float)             : Magnitude of tip-tilt guide stars
    Outputs:
    --------
    Printed output for master level file (e.g. A_keck_mcao_lgs.conf)
    '''
                                    
    # Calculate the magnitude to flux parameters for LGS/TT/Truth
    lgs_siglev, lgs_bkgrnd, lgs_nearecon = lgs_flux_params(act_start, act_stop, act_step, lgs_mag, tt_mag)
    print(lgs_siglev)
    tt_siglev, tt_bkgrnd, tt_nearecon = tt_flux_params(act_start, act_stop, act_step, lgs_mag, tt_mag)
    print(tt_siglev)
    truth_siglev, truth_bkgrnd, truth_nearecon = truth_flux_params(act_start, act_stop, act_step, lgs_mag, tt_mag)
    actuators, sides = calc_side(act_start, act_stop, act_step, sigfigs=3)

    # Print header
    print('--------------------')
    print('VisMCAO Magnitude-to-Flux Parameters:')
    print('--------------------')
    print('')

    print(f'####\n#{lgs_mag}mag LGS ({tt_mag}mag TT)\n####')

    # Iterate over actuators and print parameters
    for i, (actuator, side) in enumerate(zip(sides, actuators)):
        print('#Actuator Count:', actuator)
        print('#dm.dx = [', actuators[i], '.168 .168 ]')
        print('#powfs.siglev = [',lgs_siglev[i], tt_siglev[i], truth_siglev[i], ']')
        print('#powfs.bkgrnd = [', lgs_bkgrnd[i], tt_bkgrnd[i], truth_bkgrnd[i], ']')
        print('#powfs.nearecon = [', lgs_nearecon[i], tt_nearecon[i], truth_nearecon[i], ']')
        print('')
    
    return

def calc_psf_metrics(act_start, act_stop, act_step, lgs_mag):
    actuators = np.arange(act_start, act_stop, act_step)
    psf_metrics = []

    for act in actuators:
        home_directory = os.path.expanduser('~')
        os.chdir(home_directory)

        main_output_directory = OUTPUT_DIRECTORY
        os.chdir(main_output_directory)

        cwd1 = os.getcwd()
        #print('Current Working Directory:', cwd1)

        output_directory = f"{lgs_mag}mag/{act}act"
        os.chdir(output_directory)

        cwd2 = os.getcwd()
        #print(cwd2)

        psf_metrics.append(maos_utils.print_psf_metrics_x0y0(oversamp=3, seed=1))

    return psf_metrics

def calc_wfe(act_start, act_stop, act_step, lgs_mag):
    actuators = np.arange(act_start, act_stop, act_step)
    wfe_metrics = []

    for act in actuators:
        home_directory = os.path.expanduser('~')
        os.chdir(home_directory)

        main_output_directory = OUTPUT_DIRECTORY
        os.chdir(main_output_directory)

        cwd1 = os.getcwd()
        #print('Current Working Directory:', cwd1)

        output_directory = f"{lgs_mag}mag/{act}act"
        os.chdir(output_directory)

        cwd2 = os.getcwd()
        #print(cwd2)

        wfe_metrics.append(maos_utils.print_wfe_metrics( seed=1))

    return wfe_metrics

def extract_closed_wfe(act_start, act_stop, act_step, lgs_mag, position):
    wfe_metrics = calc_wfe(act_start, act_stop, act_step, lgs_mag)
    closed_loop_results = [result[3][position] for result in wfe_metrics]
    return closed_loop_results

def extract_total_wfe(act_start, act_stop, act_step, lgs_mag, position):
    closed_loop_results = extract_closed_wfe(act_start, act_stop, act_step, lgs_mag, position)
    total_wfe  = [result[0] for result in closed_loop_results]
    return total_wfe

def extract_ho_wfe(act_start, act_stop, act_step, lgs_mag, position):
    closed_loop_results = extract_closed_wfe(act_start, act_stop, act_step, lgs_mag, position)
    ho_wfe  = [result[2] for result in closed_loop_results]
    return ho_wfe

def extract_tt_wfe(act_start, act_stop, act_step, lgs_mag, position):
    closed_loop_results = extract_closed_wfe(act_start, act_stop, act_step, lgs_mag, position)
    tt_wfe  = [result[1] for result in closed_loop_results]
    return tt_wfe

def plot_closed_wfe(act_start, act_stop, act_step, lgs_mag, position):
    actuators = np.arange(act_start, act_stop, act_step)
    total_wfe = extract_total_wfe(act_start, act_stop, act_step, lgs_mag, position)
    ho_wfe = extract_ho_wfe(act_start, act_stop, act_step, lgs_mag, position)
    tt_wfe = extract_tt_wfe(act_start, act_stop, act_step, lgs_mag, position)

    plt.plot(actuators, total_wfe, color = 'red', label = "Total WFE")
    plt.plot(actuators, ho_wfe, color = 'blue', label = "High-Order WFE")
    plt.plot(actuators, tt_wfe, color = 'green', label = "TT WFE")

    plt.title(f"WFE vs. Actuator Count ({lgs_mag} LGS mag)")
    plt.xlabel("Actuator Count")
    plt.ylabel("WFE")
    plt.legend()
    plt.show()

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


def plot_wfe(act_start, act_stop, act_step, magnitudes, directory_results_path='A_keck_scao_lgs'):
    total_ = {}
    tt_ = {}
    ho_ = {}
    
    actuators = np.arange(act_start, act_stop, act_step)

    for mag in magnitudes:
        total_[mag] = []
        tt_[mag] = []
        ho_[mag] = []
        
        mag_directory = f"{directory_results_path}/{mag}mag"
        os.chdir(mag_directory)

        for actuator in actuators:
            act_directory = f"{actuator}a"
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
        axis[0].plot(actuators, total_[mag], color=color, label=f"{mag} magnitude")
        axis[1].plot(actuators, tt_[mag], color=color, label=f"{mag} magnitude")
        axis[2].plot(actuators, ho_[mag], color=color, label=f"{mag} magnitude")

    for ax in axis:
        ax.set_xlabel("Actuator Count on ASM (n)")

    axis[0].set_ylabel("Total Wave-front Error (nm)")
    axis[1].set_ylabel("Tip-Tilt Wave-front Error (nm)")
    axis[2].set_ylabel("High-Order Wave-front Error (nm)")

    axis[0].tick_params(left=True, labelleft=True)
    axis[1].tick_params(left=True, labelleft=True)
    axis[2].tick_params(left=True, labelleft=True)

    axis[0].set_title("Total WFE vs. Actuator Count")
    axis[1].set_title("Tip-Tilt WFE vs. Actuator Count")
    axis[2].set_title("High-Order WFE vs. Actuator Count")

    axis[1].legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=4)
    plt.savefig('act_study_wfe.png')
    plt.show()

def get_wfe_metrics(directory='./', seed=10):
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
    open_mean_nm = np.sqrt(results[0].mean(axis=0)) * 1.0e9

    # Closed-loop WFE (nm): Piston removed, TT only, Piston+TT removed
    clos_mean_nm = np.sqrt(results[2].mean(axis=0)) * 1.0e9

    # Field-dependent resutls
    # Determine if we have a field-dependent WFE results file in extra/
    results_xx_file = f'{directory}/extra/Resp_{seed}.bin'
    if os.path.exists(results_xx_file):
        results_xx = readbin.readbin(results_xx_file)

        open_xx_mean_nm = np.zeros((results_xx[2].shape[0], 3), dtype=float)
        clos_xx_mean_nm = np.zeros((results_xx[3].shape[0], 3), dtype=float)

        # Loop through PSF positions and get RMS WFE in nm
        for xx in range(open_xx_mean_nm.shape[0]):
            
            # Open-loop WFE (nm): Piston removed, TT only, Piston+TT removed
            open_xx_mean_nm[xx] = np.sqrt(results_xx[2][xx].mean(axis=0)) * 1.0e9
            
            # Closed-loop WFE (nm): Piston removed, TT only, Piston+TT removed
            clos_xx_mean_nm[xx] = np.sqrt(results_xx[3][xx].mean(axis=0)) * 1.0e9

    else:
        results_xx = None
        open_xx_mean_nm = None
        clos_xx_mean_nm = None

    return open_mean_nm, clos_mean_nm, open_xx_mean_nm, clos_xx_mean_nm