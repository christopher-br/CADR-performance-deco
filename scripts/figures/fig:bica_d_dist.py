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
from src.data.tcga_2 import load_data as tcga_2

from src.utils.viz import dose_plot

# Plot per dataset
name = "tcga_2"

# Load data
data = tcga_2(
    num_treatments=3,
    treatment_bias=0,
    dose_bias=1,
)

dose_plot(data.d,w=3,h=2, color="steelblue", labels=True, file_name="bias_1.pdf")

data = tcga_2(
    num_treatments=3,
    treatment_bias=0,
    dose_bias=2,
)

dose_plot(data.d,w=3,h=2, color="steelblue", labels=True, file_name="bias_2.pdf")

data = tcga_2(
    num_treatments=3,
    treatment_bias=0,
    dose_bias=8,
)

dose_plot(data.d,w=3,h=2, color="steelblue", labels=True, file_name="bias_8.pdf")
