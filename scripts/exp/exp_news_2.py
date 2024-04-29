## KEY SETTINGS
#####################################

DIR = "..."
DATA_NAME = "news_2"

# Chg os and sys path
import os
import sys
os.chdir(DIR)
sys.path.append(DIR)

# Num of experiments per data configuration
NUM_ITERATIONS = 10

# Number of parameter combinations to consider
RANDOM_SEARCH_N = 5


# LOAD MODULES
#####################################

# Standard library
import warnings
import silence_tensorflow.auto # To silence tensorflow in compat mode

# Third party
from tqdm import tqdm

# Regressors:
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Proprietary
from src.data.news_2 import load_data
from src.methods.other import SLearner
from src.methods.neural import DRNet, MLP, VCNet, SCIGAN
from src.methods.utils.regressors import LinearRegression, GeneralizedAdditiveModel
from src.utils.metrics import (
    mean_integrated_prediction_error,
    brier_score,
)
from src.utils.setup import (
    load_config,
    check_create_csv,
    get_rows,
    add_row,
    add_dict,
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
DATA_PARAS = load_config("config/data/config.yaml")[DATA_NAME]
HYPERPARAS = load_config("config/models/config.yaml")

# Add iteration indicator to data paras
DATA_PARAS["keys"].append("seed")
DATA_PARAS["keys"] = tuple(DATA_PARAS["keys"])
new_tuples = []
for i in range(NUM_ITERATIONS):
    for tup in DATA_PARAS["tuples"]:
        new_tuples.append(tuple(tup + [i]))
DATA_PARAS["tuples"] = new_tuples

# Generate tracker
check_create_csv(DATA_NAME+"_tracker.csv", DATA_PARAS["keys"])


## CUSTOM FUNCTIONS
#####################################

def update_dict(dict, data, model, name):
    mise = mean_integrated_prediction_error(data.x_test, data.t_test, data.ground_truth, model)
    bs = brier_score(data.x_test, data.y_test, data.d_test, data.t_test ,model)
    dict.update({f"MISE {name}": mise, f"Brier score {name}": bs})


## ITERATE OVER DATA COMBINATIONS
#####################################

for comb in tqdm(DATA_PARAS["tuples"], desc="Iterate over data combinations", leave=False):
    completed = get_rows(DATA_NAME+"_tracker.csv")
    if comb in completed:
        continue
    else:
        add_row(
            row=comb,
            file_path=DATA_NAME+"_tracker.csv",
        )
        
    data_settings = dict(zip(DATA_PARAS["keys"], comb))
    
    results = {}
    results.update(data_settings)
    
    data = load_data(**data_settings)
    
    # TRAIN MODELS

    # Linear regression
    name = "linreg"
    parameters = HYPERPARAS[name]
    parameters.update({"base_model": [LinearRegression]})
    parameters.update({"pca_degree": [50]})
    model, best_params = train_val_tuner(
        data = data,
        model=SLearner,
        parameters=parameters,
        name=name,
        num_combinations=RANDOM_SEARCH_N,
    )
    
    update_dict(results, data, model, name)
    
    # CART
    name = "cart"
    parameters = HYPERPARAS[name]
    parameters.update({"base_model": [DecisionTreeRegressor]})
    model, best_params = train_val_tuner(
        data = data,
        model=SLearner,
        parameters=parameters,
        name=name,
        num_combinations=RANDOM_SEARCH_N,
    )
    
    update_dict(results, data, model, name)
    
    # GAM
    name = "gam"
    parameters = HYPERPARAS[name]
    parameters.update({"base_model": [GeneralizedAdditiveModel]})
    parameters.update({"pca_degree": [50]})
    model, best_params = train_val_tuner(
        data = data,
        model=SLearner,
        parameters=parameters,
        name=name,
        num_combinations=RANDOM_SEARCH_N,
    )
    
    update_dict(results, data, model, name)
    
    # xgboost
    name = "xgboost"
    parameters = HYPERPARAS[name]
    parameters.update({"base_model": [XGBRegressor]})
    model, best_params = train_val_tuner(
        data = data,
        model=SLearner,
        parameters=parameters,
        name=name,
        num_combinations=RANDOM_SEARCH_N,
    )
    
    update_dict(results, data, model, name)
    
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
    
    update_dict(results, data, model, name)

    # SCIGAN
    name = "scigan"
    parameters = HYPERPARAS[name]
    parameters.update({"input_size": [data.x.shape[1]]})
    parameters.update({"num_treatments": [1]})
    parameters.update({"export_dir": ["models/"+DATA_NAME]})
    model, best_params = train_val_tuner(
        data = data,
        model = SCIGAN,
        parameters = parameters,
        name = name,
        num_combinations = RANDOM_SEARCH_N,
    )
    
    update_dict(results, data, model, name)

    # DRNet
    name = "drnet"
    parameters = HYPERPARAS[name]
    parameters.update({"input_size": [data.x.shape[1]]})
    model, best_params = train_val_tuner(
        data = data,
        model = DRNet,
        parameters = parameters,
        name = name,
        num_combinations = RANDOM_SEARCH_N,
    )
    
    update_dict(results, data, model, name)
    
    # VCNet
    name = "vcnet"
    parameters = HYPERPARAS[name]
    parameters.update({"input_size": [data.x.shape[1]]})
    model, best_params = train_val_tuner(
        data = data,
        model = VCNet,
        parameters = parameters,
        name = name,
        num_combinations = RANDOM_SEARCH_N,
    )
    
    update_dict(results, data, model, name)

    # FINALIZE ITERATION
    add_dict("res:exp_"+DATA_NAME+".csv", results)