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
from src.data.ihdp_3 import load_data as ihdp_3
from src.data.news_2 import load_data as news_2
from src.data.synth_1 import load_data as synth_1
from src.data.tcga_2 import load_data as tcga_2
from src.data.tcga_3 import load_data as tcga_3

from src.utils.viz import tsne_plot

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

tsne_plot(data.x, data.d, w=3,h=3, file_name="tsne_"+name+".pdf")

##########################

name = "ihdp_3"

# Load data
data = ihdp_3(
    bias=4.
)

tsne_plot(data.x, data.d, w=3,h=3, file_name="tsne_"+name+".pdf")

##########################

name = "news_2"

# Load data
data = news_2(
    bias=4.
)

tsne_plot(data.x, data.d, w=3,h=3, file_name="tsne_"+name+".pdf")

##########################

name = "synth_1"

# Load data
data = synth_1(
    bias=1.
)

tsne_plot(data.x, data.d, w=3,h=3, file_name="tsne_"+name+".pdf")

##########################

name = "tcga_2"

# Load data
data = tcga_2(
    num_treatments=3,
    treatment_bias=2,
    dose_bias=2,
)

tsne_plot(data.x, data.d, w=3,h=3, file_name="tsne_"+name+".pdf")

##########################

name = "tcga_3"

# Load data
data = tcga_3(
    bias_inter=0.5,
    bias_intra=5,
)

tsne_plot(data.x, data.d, w=3,h=3, file_name="tsne_"+name+".pdf")