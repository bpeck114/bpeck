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
from astropy.io import fits
from paarti.psf_metrics import metrics
from bpeck.mcao import utils

def generate_study_names1(prefix, values):
    study_names = []
    for i in values:
        study_names.append(f"{prefix}_{i}")
    print("([", ", ".join(study_names), "])")

def generate_study_names2(prefix, values, index):
    study_names = []
    for i in values:
        study_names.append(f"{prefix}_{i}[0][{index}]")
    print("([", ", ".join(study_names), "])")

def generate_study_names3(prefix, values, index):
    study_names = []
    for i in values:
        study_names.append(f"{prefix}_{i}[{index}]")
    print("([", ", ".join(study_names), "])")