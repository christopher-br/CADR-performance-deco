# LOAD MODULES
# Standard library
from typing import Union, Optional

# Third party
from sklearn.neighbors import KernelDensity

# Proprietary
from src.data.utils import (
    train_val_test_ids,
    sample_rows,
    sample_columns,
    ContinuousData,
)

# CUSTOM FUNCTIONS
def response_0(x,d,w_response):
    in0 = np.dot(x, w_response[0])
    in1 = np.dot(x, w_response[1])
    in2 = np.dot(x, w_response[2])
    
    y = 10. * (in0 + 12.0 * d * (d - 0.75 * (in1 / in2)) ** 2)
    
    return y

def response_1(x,d,w_response):
    in0 = np.dot(x, w_response[0])
    in1 = np.dot(x, w_response[1])
    in2 = np.dot(x, w_response[2])
    
    y = 10. * (in0 + np.sin(np.pi * (in1 / in2) * d))
    
    return y

def response_2(x,d,w_response):
    in0 = np.dot(x, w_response[0])
    in1 = np.dot(x, w_response[1])
    in2 = np.dot(x, w_response[2])
    
    y = 10. * (in0 + 12.0 * (in1 * d - in2 * d**2))
    
    return y

# Third party
from sklearn.utils import Bunch
from sklearn.cluster import KMeans
import numpy as np

def load_data(
    data_path: str = "data/TCGA-1.csv",
    sample_size: Optional[Union[int, float]] = None,
    num_covariates: Optional[Union[int, float]] = None,
    option: int = 1,
    bias_inter: float = .5,
    bias_intra: float = 5.,
    n_cluster: int = 3,
    train_share: float = 0.7,
    val_share: float = 0.1,
    seed: int = 42,
    noise_outcome: float = 1.,
    rm_confounding: bool = False,
    x_resampling: bool = False,
) -> Bunch:
    """
    Loads and processes the TCGA-S-3 dataset for a machine learning experiment.

    The function loads the dataset from the specified path, applies a bias to the treatments and doses, splits the data into training, validation, and test sets, and adds Gaussian noise to the outcomes.

    Parameters:
    data_path (str): The path to the dataset. Defaults to "continuous/data/datasets/TCGA-S-3.csv".
    sample_size (Optional[Union[int, float]]): The sample size to use for the dataset. If None, the full dataset is used. Defaults to None.
    num_covariates (Optional[Union[int, float]]): The number of covariates in the dataset. If None, all covariates are used. Defaults to None.
    option (int): The option for which to calculate the outcome. Can be 1, 2, or any other integer. Defaults to 1.
    bias_inter (float): The inter-cluster bias. Defaults to 0.5.
    bias_intra (float): The intra-cluster bias. Defaults to 5.0.
    n_cluster (int): The number of clusters in the dataset. Defaults to 3.
    train_share (float): The proportion of the dataset to use for the training set. Defaults to 0.7.
    val_share (float): The proportion of the dataset to use for the validation set. Defaults to 0.1.
    seed (int): The seed for the random number generator. Defaults to 42.
    noise_outcome (float): The standard deviation of the Gaussian noise added to the outcomes. Defaults to 1.

    Returns:
    Bunch: A Bunch object containing the processed data and metadata.
    """
    # Define eps
    eps = np.finfo(float).eps
    
    # Set seed
    np.random.seed(seed)
    
    # Load raw data
    matrix = np.loadtxt(data_path, delimiter=",", skiprows=1)
    
    # Sample rows and columns
    if sample_size is not None:
        matrix = sample_rows(matrix, sample_size)
    if num_covariates is not None:
        matrix = sample_columns(matrix, num_covariates)
        
    # Set max bias inter to 1 - eps
    bias_inter = min(bias_inter, 1 - eps)
    
    # Save info
    num_rows = matrix.shape[0]
    num_cols = matrix.shape[1]
    
    # Standardize
    matrix = (matrix - np.min(matrix, axis=0)) / (eps + (np.max(matrix, axis=0) - np.min(matrix, axis=0)))
    
    # Normalize
    matrix = matrix / max(np.linalg.norm(matrix, axis=1))
    
    # Resample x from uniform distribution per variable
    if x_resampling:
        min_vals = matrix.min(axis=0)
        max_vals = matrix.max(axis=0)
        # Continuous variables
        for i in range(num_cols):
            matrix[:,i] = np.random.uniform(min_vals[i], max_vals[i], num_rows)
    
    # Get cluster
    cluster = KMeans(n_clusters=n_cluster, init='k-means++', n_init=10).fit(matrix).labels_
    
    # Get cluster modes
    lb = 0.5 - 0.5 * bias_inter
    ub = 0.5 + 0.5 * bias_inter
    modal_dose_per_cluster = np.linspace(lb,ub,n_cluster)
    
    # Get weights
    w_response = np.zeros(shape=(5, num_cols))
    for i in range(5):
        w_response[i] = np.random.uniform(0, 10, size=(num_cols))
        w_response[i] = w_response[i] / np.linalg.norm(w_response[i])
    
    # Define get_outcome function
    def get_outcome(
        matrix: np.ndarray, 
        doses: np.ndarray, 
        treatments: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the outcome for a given matrix, doses, and option.

        The function calculates the outcome based on the matrix, doses, and option. The calculation involves calling response functions for each option and selecting the appropriate outcome based on the option.

        Parameters:
        matrix (np.ndarray): The dataset, represented as a 2D numpy array. Each row represents an observation, and each column represents a variable.
        doses (np.ndarray): A 1D numpy array representing the doses for each observation.
        **kwargs: Dummy to improve compatibility with other functions.

        Returns:
        np.ndarray: A 1D numpy array representing the calculated outcome for each observation.
        """
        if option == 1:
            y = response_0(matrix, doses, w_response)
        elif option == 2:
            y = response_1(matrix, doses, w_response)
        else:
            y = response_2(matrix, doses, w_response)
        return y
    
    # Get doses
    doses = []
    y = []
    # Iterate over observations
    for idx, i in enumerate(matrix):
        # Get modal dose per cluster
        modal_dose = modal_dose_per_cluster[cluster[idx]]
        
        # Get beta according to Bica et al. (2020) calculation
        beta = ((bias_intra) / modal_dose) + (2. - (bias_intra + 1))
        
        # Sample dose
        dose = np.random.beta(a=(1+bias_intra), b=beta)
        doses.append(dose)
        
        outcome = get_outcome(i, dose, None) + np.random.normal(0, noise_outcome)
        
        y.append(outcome)
    
    # Convert t and y to np array
    doses = np.array(doses)
    y = np.array(y)
    
    # Remove confounding if necessary
    if rm_confounding:
        np.random.shuffle(doses)
        
        # Get outcomes
        y = get_outcome(matrix, doses, None) + np.random.normal(0, noise_outcome)
    
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
