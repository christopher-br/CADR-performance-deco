# LOAD MODULES
# Standard library
from typing import Callable

# Third party
import numpy as np
from scipy.integrate import romb
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
plt.style.use('science')


def _chunk_rows_for_grid(
    x: np.ndarray,
    grid_size: int,
    target_elements: int = 20_000_000,
) -> int:
    """Returns a safe observation-chunk size for expanded dose-grid evaluation."""
    x = np.asarray(x)
    num_features = max(1, x.shape[1])
    denom = max(1, int(grid_size) * num_features)
    return max(1, int(target_elements // denom))


def _dose_grid(num_dose_grid: int = 200) -> np.ndarray:
    """Builds a bounded dose grid in [0, 1]."""
    return np.linspace(0.0, 1.0, max(3, int(num_dose_grid)), dtype=float)


def _evaluate_response_grid_batch(
    response: Callable,
    x: np.ndarray,
    doses: np.ndarray,
    treatment: int,
) -> np.ndarray:
    """Response evaluation over a dose grid for all rows at one treatment."""
    x = np.asarray(x)
    num_rows = x.shape[0]
    grid_size = doses.shape[0]
    chunk_rows = _chunk_rows_for_grid(x, grid_size)

    scores = np.empty((num_rows, grid_size), dtype=float)
    for start in range(0, num_rows, chunk_rows):
        end = min(start + chunk_rows, num_rows)
        x_chunk = x[start:end]
        rows = end - start

        x_grid = np.repeat(x_chunk, repeats=grid_size, axis=0)
        d_grid = np.tile(doses, rows)
        t_grid = np.full(rows * grid_size, int(treatment), dtype=int)

        chunk_scores = np.asarray(response(x_grid, d_grid, t_grid), dtype=float).reshape(rows, grid_size)
        scores[start:end] = chunk_scores

    return scores


def _evaluate_model_grid_batch(
    model: Callable,
    x: np.ndarray,
    doses: np.ndarray,
    treatment: int,
) -> np.ndarray:
    """Model evaluation over a dose grid for all rows at one treatment."""
    x = np.asarray(x)
    num_rows = x.shape[0]
    grid_size = doses.shape[0]
    chunk_rows = _chunk_rows_for_grid(x, grid_size)

    scores = np.empty((num_rows, grid_size), dtype=float)
    for start in range(0, num_rows, chunk_rows):
        end = min(start + chunk_rows, num_rows)
        x_chunk = x[start:end]
        rows = end - start

        x_grid = np.repeat(x_chunk, repeats=grid_size, axis=0)
        d_grid = np.tile(doses, rows)
        t_grid = np.full(rows * grid_size, int(treatment), dtype=int)

        chunk_scores = np.asarray(model.predict(x_grid, d_grid, t_grid), dtype=float).reshape(rows, grid_size)
        scores[start:end] = chunk_scores

    return scores


def _policy_regret_mse(
    x: np.ndarray,
    t: np.ndarray,
    response: Callable,
    model: Callable,
    num_dose_grid: int,
) -> float:
    """Computes mean squared regret for treatment-dose policy selection."""
    x = np.asarray(x)
    treatments = np.unique(t.astype(int)).astype(int)
    doses = _dose_grid(num_dose_grid=num_dose_grid)

    num_rows = x.shape[0]
    row_idx = np.arange(num_rows)

    true_best_outcome = np.full(num_rows, -np.inf, dtype=float)
    pred_best_outcome = np.full(num_rows, -np.inf, dtype=float)
    pred_best_treatment_idx = np.zeros(num_rows, dtype=int)
    pred_best_dose_idx = np.zeros(num_rows, dtype=int)
    true_scores_by_treatment = []

    for treatment_idx, treatment in enumerate(treatments):
        true_scores = _evaluate_response_grid_batch(response, x, doses, int(treatment))
        pred_scores = _evaluate_model_grid_batch(model, x, doses, int(treatment))
        true_scores_by_treatment.append(true_scores)

        true_t_best = np.max(true_scores, axis=1)
        pred_t_best_idx = np.argmax(pred_scores, axis=1)
        pred_t_best = pred_scores[row_idx, pred_t_best_idx]

        true_best_outcome = np.maximum(true_best_outcome, true_t_best)

        is_better = pred_t_best > pred_best_outcome
        pred_best_outcome[is_better] = pred_t_best[is_better]
        pred_best_treatment_idx[is_better] = treatment_idx
        pred_best_dose_idx[is_better] = pred_t_best_idx[is_better]

    true_scores_stack = np.stack(true_scores_by_treatment, axis=0)
    true_outcome_at_policy = true_scores_stack[pred_best_treatment_idx, row_idx, pred_best_dose_idx]

    regrets = np.maximum(0.0, true_best_outcome - true_outcome_at_policy)

    return float(np.mean(regrets ** 2))

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
    x = np.asarray(x)
    t = np.asarray(t)

    # Get step size
    step_size = 1 / num_integration_samples
    num_obs = x.shape[0]
    d_grid = np.linspace(0, 1, num_integration_samples)
    chunk_rows = _chunk_rows_for_grid(x, num_integration_samples)

    mises = []
    for start in range(0, num_obs, chunk_rows):
        end = min(start + chunk_rows, num_obs)
        x_chunk = x[start:end]
        t_chunk = t[start:end]
        rows = end - start

        x_expanded = np.repeat(x_chunk, repeats=num_integration_samples, axis=0)
        d_expanded = np.tile(d_grid, rows)
        t_expanded = np.repeat(t_chunk, repeats=num_integration_samples)

        y = np.asarray(response(x_expanded, d_expanded, t_expanded), dtype=float).reshape(rows, num_integration_samples)
        y_hat = np.asarray(model.predict(x_expanded, d_expanded, t_expanded), dtype=float).reshape(rows, num_integration_samples)

        chunk_mises = romb((y - y_hat) ** 2, dx=step_size, axis=1)
        mises.extend(chunk_mises.tolist())

    return float(np.sqrt(np.mean(mises)))

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
    x = np.asarray(x)
    t = np.asarray(t)
    num_obs = x.shape[0]
    d_grid = np.linspace(0, 1, num_integration_samples)
    chunk_rows = _chunk_rows_for_grid(x, num_integration_samples)

    squared_dose_errors = []
    for start in range(0, num_obs, chunk_rows):
        end = min(start + chunk_rows, num_obs)
        x_chunk = x[start:end]
        t_chunk = t[start:end]
        rows = end - start

        x_expanded = np.repeat(x_chunk, repeats=num_integration_samples, axis=0)
        d_expanded = np.tile(d_grid, rows)
        t_expanded = np.repeat(t_chunk, repeats=num_integration_samples)

        y = np.asarray(response(x_expanded, d_expanded, t_expanded), dtype=float).reshape(rows, num_integration_samples)
        y_hat = np.asarray(model.predict(x_expanded, d_expanded, t_expanded), dtype=float).reshape(rows, num_integration_samples)

        pred_best_id = np.argmax(y_hat, axis=1)
        actual_best_id = np.argmax(y, axis=1)
        chunk_errors = (d_grid[pred_best_id] - d_grid[actual_best_id]) ** 2
        squared_dose_errors.extend(chunk_errors.tolist())

    return float(np.sqrt(np.mean(squared_dose_errors)))

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
    treatments = np.unique(t.astype(int)).astype(int)

    # In multi-treatment settings, use treatment-dose policy selection (same objective as policy_error).
    if treatments.size > 1:
        mse = _policy_regret_mse(
            x=x,
            t=t,
            response=response,
            model=model,
            num_dose_grid=num_integration_samples,
        )
        return float(np.sqrt(mse))

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


def dose_policy_error(
    x: np.ndarray,
    t: np.ndarray,
    response: Callable,
    model: Callable,
    num_dose_grid: int = 200,
) -> float:
    """
    Computes dose policy error using dense dose-grid optimization.

    For each observation and treatment category, this metric compares:
    - the true outcome at the true optimal dose
    - the true outcome at the model-selected optimal dose
    and averages the resulting squared regret.
    """
    treatments = np.unique(t.astype(int)).astype(int)

    # In multi-treatment settings, dose policy uses joint treatment-dose optimization.
    if treatments.size > 1:
        return _policy_regret_mse(
            x=x,
            t=t,
            response=response,
            model=model,
            num_dose_grid=num_dose_grid,
        )

    x = np.asarray(x)
    doses = _dose_grid(num_dose_grid=num_dose_grid)
    squared_regrets = []

    for treatment in treatments:
        true_scores = _evaluate_response_grid_batch(response, x, doses, int(treatment))
        pred_scores = _evaluate_model_grid_batch(model, x, doses, int(treatment))

        true_best_outcome = np.max(true_scores, axis=1)
        pred_best_idx = np.argmax(pred_scores, axis=1)
        row_idx = np.arange(x.shape[0])
        true_outcome_at_pred = true_scores[row_idx, pred_best_idx]

        regrets = np.maximum(0.0, true_best_outcome - true_outcome_at_pred)
        squared_regrets.extend((regrets ** 2).tolist())

    return float(np.mean(squared_regrets))


def policy_error(
    x: np.ndarray,
    t: np.ndarray,
    response: Callable,
    model: Callable,
    num_dose_grid: int = 200,
) -> float:
    """
    Computes treatment-and-dose policy error using dense dose-grid optimization.

    For each observation, the metric compares:
    - the true best outcome across treatment categories and doses
    - the true outcome under the model's selected treatment category and dose
    and averages the resulting squared regret.
    """
    treatments = np.unique(t.astype(int)).astype(int)

    if treatments.size <= 1:
        return np.nan

    return _policy_regret_mse(
        x=x,
        t=t,
        response=response,
        model=model,
        num_dose_grid=num_dose_grid,
    )

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