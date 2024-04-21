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
LBWFS_INTEGRATION     = 1/1500 

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

