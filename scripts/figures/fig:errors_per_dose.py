## KEY SETTINGS
#####################################

w = 4
h = 3

DIR = "..."
DATA_NAME = "tcga_2"

# Standard library
from typing import Callable

# Chg os and sys path
import os
import sys
os.chdir(DIR)
sys.path.append(DIR)

# Third party
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import MaxNLocator
plt.style.use('science')

# Number of parameter combinations to consider
RANDOM_SEARCH_N = 5


# LOAD MODULES
#####################################

# Standard library
import warnings

# Proprietary
from src.data.tcga_2 import load_data
from src.methods.neural import MLP
from src.utils.metrics import (
    mean_integrated_prediction_error,
    brier_score,
)
from src.utils.setup import (
    load_config,
)
from src.utils.training import train_val_tuner


## SETUP
#####################################

# Disable device summaries
import logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logger = logging.getLogger("lightning.pytorch.core")
logger.addHandler(logging.FileHandler("core.log"))

# Disable warnings
warnings.filterwarnings("ignore")

# Load config
DATA_SETTINGS = {
    'x_resampling': 0, 
    'treatment_bias': 2, 
    'dose_bias': 2, 
    'rm_confounding_t': 0, 
    'rm_confounding_d': 0,
}
HYPERPARAS = load_config("config/models/config.yaml")


## CUSTOM FUNCTIONS
#####################################

def update_dict(dict, data, model, name):
    mise = mean_integrated_prediction_error(data.x_test, data.t_test, data.ground_truth, model)
    bs = brier_score(data.x_test, data.y_test, data.d_test, data.t_test ,model)
    dict.update({f"MISE {name}": mise, f"Brier score {name}": bs})

def error_per_obs(
    x: np.ndarray,
    t: int,
    d: float,
    response: Callable,
    model: Callable,
):
    num_obs = x.shape[0]
    
    d_rep = np.repeat(d, num_obs)
    t_rep = np.repeat(t, num_obs)

    # Get true outcomes
    y = response(x, d_rep, t_rep)
    # Get predictions
    y_hat = model.predict(x, d_rep, t_rep)
    
    errors = np.sqrt((y - y_hat) ** 2).reshape(-1)
    
    return errors.tolist()


## ITERATE OVER DATA COMBINATIONS
#####################################

data = load_data(**DATA_SETTINGS)

# TRAIN MODELS
# mlp
name = "mlp"
parameters = HYPERPARAS[name]
parameters.update({"input_size": [data.x.shape[1]]})
model, best_params = train_val_tuner(
    data = data,
    model = MLP,
    parameters = parameters,
    name = name,
    num_combinations = RANDOM_SEARCH_N,
)

for treat in [0,1,2]:
    # Get errors per dose
    errors_per_dose = []

    # Calculate histogram
    hist, hist_edges = np.histogram(data.d_train[data.t_train == treat], 20, density=True, range=(0,1))

    fig = plt.figure()
    
    ax0 = fig.add_axes([0, 0, 1, 1])
    ax0.axis('off')
    
    ax1 = fig.add_axes([0.15,0.15,0.7,0.83])

    # Plot histogram
    ax1.bar(
        hist_edges[:-1] + 0.5 * (hist_edges[1]-hist_edges[0]),
        hist,
        width=hist_edges[1]-hist_edges[0], alpha=0.5)
    ax1.set_ylabel('Density', fontsize=20)

    # Create a second y-axis for the boxplot
    ax2 = ax1.twinx()

    for dose in np.linspace(0.05, 0.95, 10):
        errors = error_per_obs(data.x_test, treat, dose, data.ground_truth, model)
        errors_per_dose.append(errors)

    # Plot boxplot on the second y-axis
    box = ax2.boxplot(errors_per_dose, positions=np.linspace(0.05, 0.95, 10), widths=0.025, patch_artist=True, showfliers=False)
    for patch in box['boxes']:
        patch.set_facecolor('black')
    ax2.set_ylabel('Errors', fontsize=20)
    
    # Set x-ticks
    xticklabels = [0,0.2,0.4,0.6,0.8,1]

    ax1.set_xticks(np.linspace(0, 1, 6))
    ax2.set_xticks(np.linspace(0, 1, 6))

    # Set x-tick labels
    ax1.set_xticklabels(xticklabels)
    ax2.set_xticklabels(xticklabels)

    # Limit x-axis
    ax1.set_xlim([0, 1])
    
    # Set x-axis label
    ax1.set_xlabel('Dose', fontsize=20)
    
    # Set y to whole numbers
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Adjust margins
    fig.set_size_inches(w,h)

    # Save figure
    plt.savefig(f"errors_per_dose_t{treat}.pdf")
