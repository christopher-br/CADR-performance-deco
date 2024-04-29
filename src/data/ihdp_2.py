# Source: Proprietary to this repository

# LOAD MODULES
# Standard library
from typing import Union, Optional

# Proprietary
from src.data.utils import train_val_test_ids, sample_rows, get_beta, ContinuousData

# Third party
from sklearn.utils import Bunch
from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd

# SUPPORT FUNCTIONS
def response_0(
    alpha_0: float,
    alpha_1: float,
    alpha_2: float,
    alpha_3: float,
    alpha_4: float,
    d: np.ndarray,
) -> np.ndarray:
    result_0 = 10. * (alpha_0 + 12.0 * d * (d - 0.75 * (alpha_1 + alpha_2)) ** 2)
    
    return result_0

def response_1(
    alpha_0: float,
    alpha_1: float,
    alpha_2: float,
    alpha_3: float,
    alpha_4: float,
    d: np.ndarray,
) -> np.ndarray:
    result_1 = 10. * (alpha_1 + np.sin(np.pi * (alpha_2 + alpha_3) * d))
    
    return result_1

def response_2(
    alpha_0: float,
    alpha_1: float,
    alpha_2: float,
    alpha_3: float,
    alpha_4: float,
    d: np.ndarray,
) -> np.ndarray:
    result_2 = 10. * (alpha_2 + 12.0 * (alpha_3 * d - alpha_4 * d**2))
    
    return result_2

def response_3(
    alpha_0: float,
    alpha_1: float,
    alpha_2: float,
    alpha_3: float,
    alpha_4: float,
    d: np.ndarray,
) -> np.ndarray:
    result_3 = alpha_0 * 3 * np.sin(20 * alpha_2 * d) + 20 * alpha_3 * d - 20 * alpha_4 * d ** 2 + 5

    return result_3

# DATA LOADING FUNCTION
def load_data(
    data_path: str = "data/IHDP-1.csv",
    bias: float = 1.,
    sample_size: Optional[Union[int, float]] = None,
    train_share: float = 0.7,
    val_share: float = 0.1,
    seed: int = 5,
    noise_outcome: float = 0.5,
    rm_confounding: bool = False,
    x_resampling: bool = False,
) -> Bunch:
    """
    Loads and preprocesses data from a CSV file.

    The function loads the data, optionally samples a subset of rows, normalizes the data, calculates an outcome variable, adds noise to the outcome, and splits the data into training, validation, and test sets.

    Parameters:
    data_path (str): Path to the CSV file to load. Defaults to "data/datasets/IHDP-1.csv".
    bias (float): The determinism of dose assignment. Defaults to 1 (no bias)
    sample_size (int, float): Number or proportion of rows to sample from the data. If None, all rows are used. Defaults to None.
    train_share (float): Proportion of the data to include in the training set. Defaults to 0.7.
    val_share (float): Proportion of the data to include in the validation set. Defaults to 0.1.
    seed (int): Seed for the random number generator. Defaults to 42.
    noise_outcome (float): Standard deviation of the noise to add to the outcome variable. Defaults to 0.5.
    rm_confounding (bool): Whether to remove confounding from the doses. Defaults to False.

    Returns:
    Bunch: A Bunch object containing the preprocessed data, the outcome variable, the ground truth function, and the indices for the training, validation, and test sets.
    """
    # Set seed
    np.random.seed(seed)
    
    # Load raw data
    matrix = pd.read_csv(data_path, sep=",", header=0)
    
    # To numpy
    matrix = matrix.to_numpy()
    
    # Sample rows if sample_size is specified
    if sample_size is not None:
        matrix = sample_rows(matrix, sample_size, seed=seed)
    
    # Drop columns
    matrix = matrix[:,2:27]
    
    # Save cols for family and treatment assigment
    fam_cols = matrix[:, 8:10]
    
    # Get family id
    fam_ids = np.dot(fam_cols, [2,1])
    
    # Drop fam_cols from matrix
    # matrix = np.hstack((matrix[:, 0:8], matrix[:, 10:]))
    
    # Save info
    num_rows = matrix.shape[0]
    num_cols = matrix.shape[1]
    
    # Normalize
    for col in range(num_cols):
        minval = min(matrix[:, col]) * 1.
        maxval = max(matrix[:, col]) * 1.
        matrix[:, col] = (1. * (matrix[:, col] - minval))/maxval
    
    # Resample x from uniform distribution per variable
    if x_resampling:
        min_vals = matrix.min(axis=0)
        max_vals = matrix.max(axis=0)
        # Continuous variables
        for i in range(0,6):
            matrix[:,i] = np.random.uniform(min_vals[i], max_vals[i], num_rows)
        # Binary variables
        for i in range(6,25):
            matrix[:,i] = np.random.binomial(1, 0.5, num_rows)
    
    # Save continuous variables
    x0 = matrix[:,0]
    x1 = matrix[:,1]
    x2 = matrix[:,2]
    x3 = matrix[:,3]
    x4 = matrix[:,4]
    x5 = matrix[:,5]
    
    # Get doses
    fam_modes = [0.125, 0.375, 0.625, 0.875]
    doses = np.zeros(num_rows)
    for obs in range(num_rows):
        # Get mode
        mode = fam_modes[int(fam_ids[obs])]
        # Get beta
        beta = get_beta(bias, mode)
        
        # Get dose
        dose = np.random.beta(bias, beta)
        
        doses[obs] = dose

    # Remove confounding if necessary
    if rm_confounding:
        np.random.shuffle(doses)

    def get_outcome(
        matrix: np.ndarray, 
        doses: np.ndarray, 
        treatments: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the outcome variable for a given dataset and doses.

        The function calculates the outcome variable based on a complex formula involving the doses and several columns of the dataset. The formula involves trigonometric functions, exponential functions, and hyperbolic functions.

        Parameters:
        matrix (np.ndarray): The dataset, represented as a 2D numpy array. Each row represents an observation, and each column represents a variable.
        doses (np.ndarray): A 1D numpy array representing the doses for each observation.
        **kwargs: Dummy to improve compatibility with other functions.

        Returns:
        np.ndarray: A 1D numpy array representing the calculated outcome for each observation.
        """
        # Make d np array
        doses = np.array(doses).reshape(-1)
        
        # Only take continuous variables
        x0 = matrix[:,0]
        x1 = matrix[:,1]
        x2 = matrix[:,2]
        x3 = matrix[:,3]
        x4 = matrix[:,4]
        x5 = matrix[:,5]
        
        # Save cols for family and treatment assigment
        fam_cols = matrix[:, 8:10]
        
        # Get family id
        fam_ids = np.dot(fam_cols, [2,1])
        
        y = np.zeros(matrix.shape[0])
        
        # Calc outcome
        for obs in range(matrix.shape[0]):
            fam = fam_ids[obs]
            if fam == 0:
                y[obs] = response_0(x0[obs], x1[obs], x2[obs], x3[obs], x4[obs], doses[obs])
            elif fam == 1:
                y[obs] = response_1(x0[obs], x1[obs], x2[obs], x3[obs], x4[obs], doses[obs])
            elif fam == 2:
                y[obs] = response_2(x0[obs], x1[obs], x2[obs], x3[obs], x4[obs], doses[obs])
            else:
                y[obs] = response_3(x0[obs], x1[obs], x2[obs], x3[obs], x4[obs], doses[obs])
        
        return y
    
    # Sample outcomes
    y = get_outcome(matrix, doses, None)
    
    # Add noise
    y = y + np.random.randn(matrix.shape[0]) * noise_outcome
    
    # Generate a dummy array for treatment
    t = np.zeros(num_rows).astype(int)
    
    # Get train/val/test ids
    train_ids, val_ids, test_ids = train_val_test_ids(num_rows,
                                                      train_share=train_share,
                                                      val_share=val_share,)
    
    # Generate bunch
    data = ContinuousData(
        x = matrix,
        t = t,
        d = doses,
        y = y,
        ground_truth = get_outcome,
        train_ids = train_ids,
        val_ids = val_ids,
        test_ids = test_ids,
    )
    
    return data