# LOAD MODULES
# Standard library
import os
import sys

# Third party
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

DIR = "..."
os.chdir(DIR)
sys.path.append(DIR)

# Proprietary
from src.data.ihdp_1 import load_data as ihdp_1
from src.data.ihdp_2 import load_data as ihdp_2
from src.data.news_2 import load_data as news_2
from src.data.synth_1 import load_data as synth_1
from src.data.tcga_2 import load_data as tcga_2
from src.data.tcga_3 import load_data as tcga_3

from src.utils.viz import dose_plot

# Settings
num_bins = 20

####################
# Plot per dataset #
####################

name = "ihdp_1"

# Load data
data = ihdp_1(
    bias=1.
)

dose_plot(data.d,w=3,h=2, color="steelblue", labels=True, file_name="dd_"+name+".pdf", num_bins=num_bins)

##########################

name = "ihdp_2"

# Load data
data = ihdp_2(
    bias=4.
)

dose_plot(data.d,w=3,h=2, color="steelblue", labels=True, file_name="dd_"+name+".pdf", num_bins=num_bins)

##########################

name = "news_2"

# Load data
data = news_2(
    bias=4.
)

dose_plot(data.d,w=3,h=2, color="steelblue", labels=True, file_name="dd_"+name+".pdf", num_bins=num_bins)

##########################

name = "synth_1"

# Load data
data = synth_1(
    bias=1.
)

dose_plot(data.d,w=3,h=2, color="steelblue", labels=True, file_name="dd_"+name+".pdf", num_bins=num_bins)

##########################

name = "tcga_2"

# Load data
data = tcga_2(
    num_treatments=3,
    treatment_bias=2,
    dose_bias=2,
)

dose_plot(data.d,w=3,h=2, color="steelblue", labels=True, file_name="dd_"+name+".pdf", num_bins=num_bins)

##########################

name = "tcga_3"

# Load data
data = tcga_3(
    bias_inter=0.5,
    bias_intra=5,
)

dose_plot(data.d,w=3,h=2, color="steelblue", labels=True, file_name="dd_"+name+".pdf", num_bins=num_bins)
