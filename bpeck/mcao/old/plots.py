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


#MCAO Metrics
psf_x = [0, 5, 10, 15, 20, 25, 30, 35, 45, 60]
psf_y = [0, 0,  0,  0,  0,  0,  0,  0,  0,  0]
wfe = ["Total WFE", "Tip-Tilt", "High-Order"]
colors = ["r", "g", "b"]

def plot_wfe_metrics(directory='./', seed=1):
    """
    Function to plot various wave-front error (WFE) metrics.

    Inputs:
    ------------
    directory      : string, default is current directory
        Path to directory where simulation results live

    seed           : int, default=10
        Seed with which simulation was run

    Outputs:
    ------------
    """
    # Field-dependent results from MAOS outputs
    results_xx_file = f'{directory}/extra/Resp_{seed}.bin'
    results_xx = readbin.readbin(results_xx_file)

    #Number of PSF positions
    n_psf = clos_mean_nm.shape[0]

    #Generate subplots (contains 2 rows and 5 columns for 10 PSFs)
    fig, ax = plt.subplots(2, n_psf // 2, figsize= (32,8), gridspec_kw={'hspace': 0.3, 'wspace': 0.4})

    #Loop through PSF position and WFE metrics
    for i in range(n_psf):
    row_index = i // (n_psf // 2)
    col_index = (i % (n_psf // 2))
    clos_psf_results = results_xx[3][i]
    for j in range(clos_psf_results.shape[1]):
        ax[row_index,col_index].plot(np.sqrt(clos_psf_results[:,j])*1.0e9, color=colors[j], label=wfe[j], linewidth=0.5)
        ax[row_index,col_index].set_xlabel('Time Step')
        ax[row_index,col_index].set_ylabel('Wavefront Error (nm)')
        ax[row_index,col_index].set_title(f'PSF {psf_x[i]}')
    
    return