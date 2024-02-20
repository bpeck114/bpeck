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
from matplotlib import colors
from scipy.interpolate import interp1d
from paarti.utils import maos_utils
from paarti import psfs, psf_plots
from paarti.psf_metrics import metrics
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import colors
from matplotlib.colorbar import Colorbar
from astropy.io import fits

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def setup_colorbar(wvl_start, wvl_stop, act_start, act_stop, lgs_magnitude, wvl_set=True):
    wvls = np.arange(wvl_start, wvl_stop, 1)
    actuators = np.arange(act_start, act_stop, 1000)
    
    if wvl_set == True:
        wvl_values = np.array([800, 1000, 1250, 1650, 2200])

    min_actuator = np.min(actuators)
    max_actuator = np.max(actuators)

    min_wvl = np.min(wvls)
    max_wvl = np.max(wvls)

    min_directory = f"vismcao/A_keck_scao_lgs/actuators_{lgs_magnitude}mag/{min_actuator}a"
    os.chdir(min_directory)

    min_psf = psfs.MAOS_PSF_stack(isgrid=True, bandpass=min_wvl)
    min_n = min_psf.psfs.shape[0]
    min_side = int(np.sqrt(min_n))
    min_n_side = int(np.sqrt(min_n))
    min_pos = np.linspace(0, (min_n_side - 1), min_side)
    min_pos = min_pos.astype(int)
    #min_zoom = 0.5
    min_zoom_px = round(min_psf.psfs.shape[1] / 2)
    min_zoom_px_min = (min_psf.psfs.shape[1] / 2) - min_zoom_px
    min_zoom_px_max = (min_psf.psfs.shape[1] / 2) + min_zoom_px
    min_psf_color = min_psf.psfs[:, int(min_zoom_px_min): int(min_zoom_px_max), int(min_zoom_px_min): int(min_zoom_px_max)]
    cmap_name = 'hot'
    min_psf_color_min = np.min(min_psf_color)
    min_psf_color_max = np.max(min_psf_color)

    os.chdir("..")

    max_directory = f"{max_actuator}a"
    os.chdir(max_directory)

    psf_max = psfs.MAOS_PSF_stack(isgrid=True, bandpass=max_wvl)
    max_n = psf_max.psfs.shape[0]
    max_side = int(np.sqrt(max_n))
    max_n_side = int(np.sqrt(max_n))
    max_pos = np.linspace(0, (max_n_side - 1), max_side)
    max_pos = max_pos.astype(int)
    #max_zoom = 0.5
    max_zoom_px = round(psf_max.psfs.shape[1] / 2)
    max_zoom_px_min = (psf_max.psfs.shape[1] / 2) - max_zoom_px
    max_zoom_px_max = (psf_max.psfs.shape[1] / 2) + max_zoom_px
    max_psf_color = psf_max.psfs[:, int(max_zoom_px_min): int(max_zoom_px_max), int(max_zoom_px_min): int(max_zoom_px_max)]
    cmap_name = 'hot'
    max_psf_color_min = np.min(max_psf_color)
    max_psf_color_max = np.max(max_psf_color)

    os.chdir("..")
    os.chdir("..")
    os.chdir("..")
    os.chdir("..")

    return min_psf_color_min, min_zoom_px, max_psf_color_max, max_zoom_px
    
def return_psf_metrics_x0y0(directory='./', oversamp=3, seed=1, cut_radius=20):
    """
    Print some PSF metrics for a central PSF computed by MAOS
    at an arbitrary number of wavelengths.
    """
    #print("Looking in directory:", directory)  
    fits_files = glob.glob(directory + f'evlpsfcl_{seed}_x0_y0.fits')
    
    psf_all_wvls = fits.open(fits_files[0])
 
    nwvl = len(psf_all_wvls)

    wavelengths = np.zeros(nwvl)
    strehl_values = np.zeros(nwvl)
    fwhm_gaus_values = np.zeros(nwvl)
    fwhm_emp_values = np.zeros(nwvl)
    r_ee80_values = np.zeros(nwvl)
 
    #print(f'{"Wavelength":10s} {"Strehl":>6s} {"FWHM_gaus":>10s} {"FWHM_emp":>10s} {"r_EE80":>6s}')
    #print(f'{"(microns)":10s} {"":>6s} {"(mas)":>10s} {"(mas)":>10s} {"(mas)":>6s}')
    
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


def psf_grid3(wvl_start, wvl_stop, act_start, act_stop, lgs_magnitude, wvl_set=True):
    wvls = np.arange(wvl_start, wvl_stop, 1)
    actuators = np.arange(act_start, act_stop, 1000)

    if wvl_set == True:
        wvl_values = np.array([432, 544, 652, 810, 877, 1020, 1248, 1673, 2200])

    base_psf_color_min, base_min_zoom_px, base_psf_color_max, base_max_zoom_px = setup_colorbar(wvl_start, wvl_stop, act_start, act_stop, lgs_magnitude)
    
    num_rows = len(actuators)
    plots_per_row = len(wvls)
    
    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(12, 10))

    lgs_directory = f"vismcao/A_keck_scao_lgs/actuators_{lgs_magnitude}mag"
    os.chdir(lgs_directory)

    for i_row, actuator in enumerate(actuators):
        act_directory = f"{actuator}a"
        os.chdir(act_directory)

        _, strehl, fwhm, _, _ = return_psf_metrics_x0y0(directory='./', oversamp=3, seed=1, cut_radius=20)
        
        for j_column, wvl in enumerate(wvls):
            psf = psfs.MAOS_PSF_stack(isgrid=True, bandpass=wvl)
            
            n = psf.psfs.shape[0]
            side = int(np.sqrt(n))
            n_side = int(np.sqrt(n))
            pos = np.linspace(0, (n_side - 1), side)
            pos = pos.astype(int)

            #Size of Subplot
            zoom = 0.25
            zoom_px = np.ceil(zoom / psf.pixel_scale).astype("int")
            #zoom_px = round(psf.psfs.shape[1] / 2)
            #print("zoom_px:", zoom_px)
            
            zoom_px_min = (psf.psfs.shape[1] / 2) - zoom_px
            #print("zoom_px_min:", zoom_px_min)
            
            zoom_px_max = (psf.psfs.shape[1] / 2) + zoom_px
            #print("zoom_px_max:", zoom_px_max)

            #Color Range
            psf_color = psf.psfs[:, int(zoom_px_min): int(zoom_px_max), int(zoom_px_min): int(zoom_px_max)]
            cmap_name = 'afmhot'
            psf_color_min = np.min(psf_color)
            psf_color_max = np.max(psf_color)
            #print(psf_color_min, psf_color_max)
            norm = colors.LogNorm(vmin=base_psf_color_min, vmax=base_psf_color_max)
            #print(base_psf_color_min, base_psf_color_max)

            ax = axes[i_row, j_column]
            for ii_row in range(side):
                for ii_col in range(side):
                    idx = np.where((psf.pos[:, 0] == pos[ii_row]) & (psf.pos[:, 1] == pos[ii_col]))[0][0]
                    ax.imshow(psf.psfs[idx, :, :], norm=norm, cmap=plt.get_cmap(cmap_name), aspect='equal', origin='lower')
                    ax.axis("equal")
                    ax.tick_params(axis='x', bottom=False, labelbottom=False)
                    ax.tick_params(axis='y', left=False, labelleft=False)
                    
                    description = "Strehl: {:.2f}\nFWHM: {:.2f}".format(strehl[j_column], fwhm[j_column])
                    print("strehl:", strehl[j_column])
                    print("fwhm:", fwhm[j_column])
                    ax.text(.05, .1, description, fontsize=6, ha='left', va='center', color='white', transform=ax.transAxes)
                    ax.set_xlim([zoom_px_min, zoom_px_max])
                    ax.set_ylim([zoom_px_min, zoom_px_max])


            if i_row == 0:
                ax.set_title(f"{wvl_values[j_column]} nm")
            if j_column == 0:
                ax.set_ylabel(f"{actuator} actuators")
                
        os.chdir("..")

    os.chdir("..")
    os.chdir("..")
    os.chdir("..")
    
    plt.suptitle("Wavelength vs. Actuators for 6 mag LGS")
    #plt.tight_layout()
    plt.savefig("psf_wvl_act_optical.png")
    plt.show()