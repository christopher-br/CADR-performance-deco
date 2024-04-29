# LOAD MODULES
# Standard library
from typing import Callable

# Third party
import numpy as np
from scipy.integrate import romb
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

def mise_t(
    x: np.ndarray,
    t: np.ndarray,
    response: Callable,
    model: Callable,
    num_integration_samples: int = 65,
) -> float:
    """
    Calculates the Mean Integrated Prediction Error (MIPE) for a given model and the factual treatments.

    The MIPE integrates the squared difference between the true response and the model's predicted response over all possible doses
    (1/n) * \sum{\ingral{0}{1}{(y_i(d) - y-hat_i(d))^2}dd}

    Parameters:
        x (np.ndarray): The covariates.
        response (Callable): A function representing the true response.
        model (Callable): The model to be evaluated.
        num_integration_samples (int, optional): The number of samples to be used for the integration. Defaults to 65.

    Returns:
        float: The Mean Integrated Prediction Error of the model.
    """
    # Get step size
    step_size = 1 / num_integration_samples
    num_obs = x.shape[0]

    # Generate data
    x = np.repeat(x, repeats=num_integration_samples, axis=0)
    d = np.linspace(0, 1, num_integration_samples)
    d = np.tile(d, num_obs)
    t = np.repeat(t, repeats=num_integration_samples)

    # Get true outcomes
    y = response(x, d, t)
    # Get predictions
    y_hat = model.predict(x, d, t)

    # Get mise
    mises = []
    y_chunks = y.reshape(-1, num_integration_samples)
    y_hat_chunks = y_hat.reshape(-1, num_integration_samples)

    for y_chunk, y_hat_chunk in zip(y_chunks, y_hat_chunks):
        mise = romb((y_chunk - y_hat_chunk) ** 2, dx=step_size)
        mises.append(mise)

    return np.sqrt(np.mean(mises))

def mean_integrated_prediction_error(
    x: np.ndarray,
    t: np.ndarray,
    response: Callable,
    model: Callable,
    num_integration_samples: int = 65,
) -> float:
    """
    Calculates the Mean Integrated Prediction Error (MIPE) for a given model average over all treatments.

    The MIPE integrates the squared difference between the true response and the model's predicted response over all possible doses
    (1/n) * \sum{\ingral{0}{1}{(y_i(d) - y-hat_i(d))^2}dd}

    Parameters:
        x (np.ndarray): The covariates.
        response (Callable): A function representing the true response.
        model (Callable): The model to be evaluated.
        num_integration_samples (int, optional): The number of samples to be used for the integration. Defaults to 65.

    Returns:
        float: The Mean Integrated Prediction Error of the model.
    """
    # Get treatments
    treatments = np.unique(t)

    # Get mise
    mises = []
    
    for t in treatments:
        t = np.repeat(t, repeats=x.shape[0])
        mise = mise_t(x, t, response, model, num_integration_samples)
        mise = mise ** 2
        mises.append(mise)
    
    return np.sqrt(np.mean(mises))

def mean_dose_error(
    x: np.ndarray,
    t: np.ndarray,
    response: Callable,
    model: Callable,
    num_integration_samples: int = 65,
) -> float:
    """
    Calculates the Mean Dose Error (MDE) for a given model.

    The MDE measures the mean difference between the true best dose and the dose selected by the model.
    (1/n) * \sum{(d^*_i - d-hat^*_i)^2}

    Parameters:
        x (np.ndarray): The covariates.
        t (np.ndarray): The treatments.
        response (Callable): A function representing the true response.
        model (Callable): The model to be evaluated.
        num_integration_samples (int, optional): The number of samples to be used for the integration. Defaults to 65.

    Returns:
        float: The Mean Dose Error of the model.
    """
    num_obs = x.shape[0]

    # Generate data
    x = np.repeat(x, repeats=num_integration_samples, axis=0)
    d = np.linspace(0, 1, num_integration_samples)
    d = np.tile(d, num_obs)
    t = np.repeat(t, repeats=num_integration_samples)

    # Get true outcomes
    y = response(x, d, t)
    # Get predictions
    y_hat = model.predict(x, d, t)

    # Get mean dose error
    squared_dose_errors = []
    y_chunks = y.reshape(-1, num_integration_samples)
    y_hat_chunks = y_hat.reshape(-1, num_integration_samples)
    d_chunks = d.reshape(-1, num_integration_samples)

    # Get errors
    for y_chunk, y_hat_chunk, d_chunk in zip(y_chunks, y_hat_chunks, d_chunks):
        pred_best_id = np.argmax(y_hat_chunk)
        actual_best_id = np.argmax(y_chunk)
        squared_dose_error = (d_chunk[pred_best_id] - d_chunk[actual_best_id]) ** 2
        squared_dose_errors.append(squared_dose_error)

    return np.sqrt(np.mean(squared_dose_errors))

def mean_outcome_defect(
    x: np.ndarray,
    t: np.ndarray,
    response: Callable,
    model: Callable,
    num_integration_samples: int = 65,
) -> float:
    """
    Calculates the Mean Outcome Defect (MOD) for a given model.

    The MOD measures the mean difference between the true best outcome and the true outcome at the best dose as predicted by the model
    (1/n) * \sum{(y^*_i - y_i(d-hat^*_i))^2}

    Parameters:
        x (np.ndarray): The covariates.
        t (np.ndarray): The treatments.
        response (Callable): A function representing the true response.
        model (Callable): The model to be evaluated.
        num_integration_samples (int, optional): The number of samples to be used for the integration. Defaults to 65.

    Returns:
        float: The Mean Outcome Defect of the model.
    """
    num_obs = x.shape[0]

    # Generate data
    x = np.repeat(x, repeats=num_integration_samples, axis=0)
    d = np.linspace(0, 1, num_integration_samples)
    d = np.tile(d, num_obs)
    t = np.repeat(t, repeats=num_integration_samples)

    # Get true outcomes
    y = response(x, d, t)
    # Get predictions
    y_hat = model.predict(x, d, t)

    # Get mean dose error
    squared_outcome_defects = []
    y_chunks = y.reshape(-1, num_integration_samples)
    y_hat_chunks = y_hat.reshape(-1, num_integration_samples)
    d_chunks = d.reshape(-1, num_integration_samples)

    # Get errors
    for y_chunk, y_hat_chunk, d_chunk in zip(y_chunks, y_hat_chunks, d_chunks):
        pred_best_id = np.argmax(y_hat_chunk)
        generated_outcome = y_chunk[pred_best_id]
        actual_best_outcome = np.max(y_chunk)
        squared_defect = (actual_best_outcome - generated_outcome) ** 2
        squared_outcome_defects.append(squared_defect)

    return np.sqrt(np.mean(squared_outcome_defects))

def brier_score(
    x: np.ndarray, 
    y: np.ndarray, 
    d: np.ndarray, 
    t: np.ndarray,
    model: Callable
) -> float:
    """
    Calculates the Brier score for a given model.

    The Brier score compares the estimated response to the actual response for an observed dose.

    Parameters:
        x (np.ndarray): The covariates.
        y (np.ndarray): The actual outcomes.
        d (np.ndarray): The dose.
        t (np.ndarray): The treatments.
        model (Callable): The model to be evaluated.

    Returns:
        float: The Brier score of the model.
    """
    # Get predictions
    y_hat = model.predict(x, d, t)

    # Return Brier score
    return np.sqrt(np.mean((y - y_hat) ** 2))

def auc_roc(
    x: np.ndarray, 
    d: np.ndarray, 
    t: np.ndarray,
    outcomes: np.ndarray, 
    doses: np.ndarray, 
    model: Callable
) -> float:
    """
    Calculates the Area Under the Receiver Operating Characteristic (AUC-ROC) for a given model.

    The AUC-ROC is a measure of the usefulness of a model in those cases where the classes are balanced. 
    It is used in binary classification to study the output of a classifier.

    Parameters:
        x (np.ndarray): The covariates.
        d (np.ndarray): The doses.
        t (np.ndarray): The treatments.
        outcomes (np.ndarray): The actual outcomes.
        model (Callable): The model to be evaluated.

    Returns:
        float: The AUC-ROC of the model.
    """
    # True outcomes
    y = outcomes
    # Get predictions
    y_hat = model.predict(x, d, t)

    # Calc roc curve
    fpr, tpr, _ = roc_curve(y, y_hat)
    area_under = auc(fpr, tpr)

    return area_under

def auc_pr(
    x: np.ndarray, 
    d: np.ndarray,
    t: np.ndarray,
    outcomes: np.ndarray, 
    model: Callable
) -> float:
    """
    Calculates the Area Under the Precision-Recall Curve (AUC-PR) for a given model.

    The AUC-PR is a measure of the usefulness of a model in those cases where the classes are imbalanced. 
    It is used in binary classification to study the output of a classifier.

    Parameters:
        x (np.ndarray): The covariates.
        d (np.ndarray): The doses.
        t (np.ndarray): The treatments.
        outcomes (np.ndarray): The actual outcomes.
        model (Callable): The model to be evaluated.

    Returns:
        float: The AUC-PR of the model.
    """
    # True outcomes
    y = outcomes
    # Get predictions
    y_hat = model.predict(x, d, t)

    # Calc roc curve
    precision, recall, _ = precision_recall_curve(y, y_hat)
    area_under = auc(recall, precision)

    return area_under