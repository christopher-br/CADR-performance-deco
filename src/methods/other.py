# LOAD MODULES
# Standard library
from typing import Callable, Optional, Type, Dict

# Proprietary
from src.methods.utils.classes import ContinuousCATE
from src.methods.utils.regressors import LinearRegression, LogisticRegression

# Third party
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from scipy.stats import norm

class KernelRidge(ContinuousCATE):
    """
    Kernel Ridge Regression with a tensor-product kernel K = K_T ⊙ K_D ⊙ K_X.

    Each logical argument of the CADR mu(t, d, x) gets its own kernel, combined
    via Hadamard product, following Singh, Xu & Gretton (2024), "Kernel methods
    for causal functions" (Biometrika, asad042), Section 5 / Algorithm 1.

    - K_X: RBF on covariates with bandwidth ``sigma_x * median_pairwise_distance(X_train)``.
    - K_D: RBF on dose with bandwidth ``sigma_d * median_pairwise_distance(D_train)``.
    - K_T: delta kernel 1(t_i = t_j) (default; cf. Singh et al. Corollary 1, the
      natural choice for a categorical intervention) or RBF on T with bandwidth
      ``sigma_t * median_pairwise_distance(T_train)``.

    For single-intervention datasets (t constant), the delta K_T is the all-ones
    matrix and K reduces to K_D ⊙ K_X.

    Parameters:
        lambd1 (float): Ridge penalty applied as n * lambd1 * I.
        sigma_x (float): Multiplier on the median-heuristic bandwidth for X.
        sigma_d (float): Multiplier on the median-heuristic bandwidth for D.
        sigma_t (float): Multiplier on the median-heuristic bandwidth for T
            (only used when t_kernel="rbf").
        t_kernel (str): "delta" (default) or "rbf".
    """
    def __init__(
        self,
        lambd1: float = 1.0,
        sigma_x: float = 1.0,
        sigma_d: float = 1.0,
        sigma_t: float = 1.0,
        t_kernel: str = "delta",
        **kwargs
    ) -> None:
        self.lambd1 = lambd1
        self.sigma_x = sigma_x
        self.sigma_d = sigma_d
        self.sigma_t = sigma_t
        self.t_kernel = t_kernel

    @staticmethod
    def _median_pairwise_distance(A: np.ndarray) -> float:
        n = A.shape[0]
        if n < 2:
            return 1.0
        sq = np.sum(A**2, axis=1, keepdims=True)
        dist = sq + sq.T - 2 * A @ A.T
        np.maximum(dist, 0, out=dist)
        np.sqrt(dist, out=dist)
        iu = np.triu_indices(n, k=1)
        med = float(np.median(dist[iu]))
        return med if med > 0 else 1.0

    @staticmethod
    def _rbf_kernel(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
        sq_A = np.sum(A**2, axis=1, keepdims=True)
        sq_B = np.sum(B**2, axis=1, keepdims=True)
        dist = sq_A + sq_B.T - 2 * A @ B.T
        np.maximum(dist, 0, out=dist)
        return np.exp(-dist / (2 * sigma**2))

    def _build_t_kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.t_kernel == "delta":
            return (A == B.T).astype(float)
        if self.t_kernel == "rbf":
            return self._rbf_kernel(A, B, self._sigma_t_eff)
        raise ValueError(f"Unknown t_kernel: {self.t_kernel!r}")

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        t: np.ndarray,
    ) -> None:
        self.x_train = np.asarray(x)
        self.d_train = np.asarray(d).reshape(-1, 1)
        self.t_train = np.asarray(t).reshape(-1, 1)
        n = self.x_train.shape[0]

        self._sigma_x_eff = self.sigma_x * self._median_pairwise_distance(self.x_train)
        self._sigma_d_eff = self.sigma_d * self._median_pairwise_distance(self.d_train)
        if self.t_kernel == "rbf":
            self._sigma_t_eff = self.sigma_t * self._median_pairwise_distance(self.t_train)

        K_X = self._rbf_kernel(self.x_train, self.x_train, self._sigma_x_eff)
        K_D = self._rbf_kernel(self.d_train, self.d_train, self._sigma_d_eff)
        K_T = self._build_t_kernel(self.t_train, self.t_train)
        K = K_X * K_D * K_T

        self.alpha = np.linalg.solve(K + n * self.lambd1 * np.eye(n), y)

    def predict(
        self,
        x: np.ndarray,
        d: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        x_new = np.asarray(x)
        d_new = np.asarray(d).reshape(-1, 1)
        t_new = np.asarray(t).reshape(-1, 1)

        k_X = self._rbf_kernel(x_new, self.x_train, self._sigma_x_eff)
        k_D = self._rbf_kernel(d_new, self.d_train, self._sigma_d_eff)
        k_T = self._build_t_kernel(t_new, self.t_train)

        return (k_X * k_D * k_T) @ self.alpha


class NLearner(ContinuousCATE):
    """
    N-Learner class that inherits from the ContinuousCATE class.

    This class represents an N-Learner, which is a type of causal model. It inherits 
    from the ContinuousCATE class, which provides base functionality for continuous 
    causal additive treatment effect models.
    
    For settings with multiple continuous-valued treatments, the N-Learner fits a separate 
    model for every treatment. 
    
    Methods:
        fit(X, Y, D, T): Fits the model to the data.
        predict(X): Predicts the treatment effect for the given data.
    """
    def __init__(
        self, 
        model_dict: Dict[int, Callable] = {0: LinearRegression}, 
        **kwargs
    ) -> None:
        """
        Initializes a new instance of the class.

        This method initializes a new instance of the class with the specified base model and additional parameters.

        Parameters:
            model_dict: A dictionary with the treatment as key and the model per treatment as value.
            **kwargs: Additional keyword arguments to be passed to the base model.
        """
        # Save model (sklearn estimator)
        self.model_dict = model_dict
        self.kwargs = kwargs
        self.models = {}
    
    def fit(
        self, 
        x: np.ndarray, 
        y: np.ndarray,
        d: np.ndarray,
        t: np.ndarray,
    ) -> None:
        """
        This is dummy method, as the class only takes pre-trained models.
        """
        print("Not able to fit model. Please use pre-trained models.")
    
    def predict(
        self, 
        x: np.ndarray,
        d: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Predicts the treatment effect for the given data.
        """
        # Generate predictions based on regressor flag:
        y_hat = np.zeros(x.shape[0])
        for unique_t in np.unique(t):
            mask = t == unique_t
            x_subset = x[mask]
            d_subset = d[mask]
            t_subset = t[mask]
            
            # Calculate predictions
            y_hat[mask] = self.model_dict[unique_t].predict(x_subset, d_subset, t_subset)

        return y_hat

class SLearner(ContinuousCATE):
    """
    S-Learner class that inherits from the ContinuousCATE class.

    This class represents an S-Learner, which is a type of causal model. It inherits 
    from the ContinuousCATE class, which provides base functionality for continuous 
    causal additive treatment effect models.
    
    Methods:
        fit(X, Y, D, T): Fits the model to the data.
        predict(X, D, T): Predicts the treatment effect for the given data.
    """
    def __init__(
        self, 
        base_model: BaseEstimator = LinearRegression, 
        pca_degree: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Initializes a new instance of the class.

        This method initializes a new instance of the class with the specified base model and additional parameters.

        Parameters:
            base_model (BaseEstimator, optional): The base model to be used in the class. Defaults to LinearRegression.
            **kwargs: Additional keyword arguments to be passed to the base model.
        """
        # Save model (sklearn estimator)
        self.base_model = base_model(**kwargs)

        # Save settings
        self.pca_degree = pca_degree

    def fit(
        self, 
        x: np.ndarray, 
        y: np.ndarray,
        d: np.ndarray,
        t: np.ndarray,
    ) -> None:
        """
        Fits model to the data.
        """
        # Update pca_degree
        if self.pca_degree is not None:
            if x.shape[1] < self.pca_degree:
                self.pca_degree = x.shape[1]
            
            # Define PCA
            self.pca = PCA(self.pca_degree, random_state=42).fit(x)
        
            # PCA transform data
            x = self.pca.transform(x)

        # Concat data
        xdt = np.column_stack((x, d, t))

        # Fit model
        self.base_model.fit(xdt, y)

    def predict(
        self, 
        x: np.ndarray,
        d: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Predicts the treatment effect for the given data.
        """
        # PCA transform data
        if self.pca_degree is not None:
            x = self.pca.transform(x)

        xdt = np.column_stack((x, d, t))

        # Generate predictions based on regressor flag:
        try:
            y_hat = self.base_model.predict_proba(xdt)[:, 1]
        except:
            y_hat = self.base_model.predict(xdt)

        return y_hat

class HIE(ContinuousCATE):
    """
    HIE class that inherits from the ContinuousCATE class.

    This class represents a Hirano Imbens Estimator (HIE), which is a type of causal model. It inherits 
    from the ContinuousCATE class, which provides base functionality for continuous 
    causal additive treatment effect models.

    Methods:
        fit(X, Y, D, T): Fits the model to the data.
        predict(X): Predicts the treatment effect for the given data.
    """
    def __init__(
        self,
        gps_model: Type[BaseEstimator] = LogisticRegression,
        effect_model: Type[BaseEstimator] = LogisticRegression,
        treatment_interaction_degree: int = 1,
        outcome_interaction_degree: int = 2,
        pca_degree: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Initializes a new instance of the class.

        This method initializes a new instance of the class with the specified parameters.

        Parameters:
            gps_model (Type[BaseEstimator], optional): The model to be used for the Generalized Propensity Score. Defaults to LogisticRegression.
            effect_model (Type[BaseEstimator], optional): The model to be used for the effect estimation. Defaults to LogisticRegression.
            treatment_interaction_degree (int, optional): The degree of interaction for the treatment. Defaults to 1.
            outcome_interaction_degree (int, optional): The degree of interaction for the outcome. Defaults to 2.
            pca_degree (Optional[int], optional): The degree of PCA to be applied. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the base models.
        """
        # Define feature transformer
        self.treatment_feature_transformer = PolynomialFeatures(
            treatment_interaction_degree, include_bias=False
        )
        self.outcome_feature_transformer = PolynomialFeatures(
            outcome_interaction_degree, include_bias=False
        )

        # Initialize treatment and outcome models
        self.treatment_model = gps_model(**kwargs)
        self.outcome_model = effect_model(**kwargs)

        # Save settings
        self.treatment_error_scale = 1
        self.pca_degree = pca_degree
        

    def fit(
        self, 
        x: np.ndarray, 
        y: np.ndarray,
        d: np.ndarray,
        t: np.ndarray,
    ) -> None:
        """
        Fits the model to the given data.

        This method takes as input the features, target, and treatment indicator, and fits the model to the data.

        Parameters:
            x (np.ndarray): The features.
            y (np.ndarray): The target.
            d (np.ndarray): The dose.
            t (np.ndarray): The treatment indicator.
        """
        # Update pca_degree
        if self.pca_degree is not None:
            if x.shape[1] < self.pca_degree:
                self.pca_degree = x.shape[1]
            
            # Define PCA
            self.pca = PCA(self.pca_degree, random_state=42).fit(x)
        
            # PCA transform data
            x = self.pca.transform(x)
        
        # Create x by concatenating x and t
        x = np.column_stack((x, t))
        
        # Transform data
        x = self.treatment_feature_transformer.fit_transform(x)

        # Train treatment model
        self.treatment_model.fit(x, d)

        # Estimate errors
        try:
            errors = self.treatment_model.predict_proba(x)[:, 1] - d
        except:
            errors = self.treatment_model.predict(x) - d

        # Update treatment std error
        _, self.treatment_error_scale = norm.fit(errors)

        # Get propensity score and generate prediction data
        propensity_scores = norm.pdf(errors, loc=0, scale=self.treatment_error_scale)

        # Train data (propensity score, t)
        pd = np.column_stack((propensity_scores, d))

        # Transform data
        pd = self.outcome_feature_transformer.fit_transform(pd)

        # Fit outcome model
        self.outcome_model.fit(pd, y)

    def predict(
        self, 
        x: np.ndarray, 
        d: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """
        Predicts the treatment effect for the given features and treatment indicator.

        This method takes as input the features and treatment indicator, and predicts the treatment effect.

        Parameters:
            x (np.ndarray): The features.
            d (np.ndarray): The dose.
            t (np.ndarray): The treatment indicator.

        Returns:
            np.ndarray: The predicted treatment effect.
        """
        # PCA transform data
        if self.pca_degree is not None:
            x = self.pca.transform(x)
            
        # Create x by concatenating x and t
        x = np.column_stack((x, t))
        
        # Transform data
        x = self.treatment_feature_transformer.fit_transform(x)

        # Errors
        try:
            errors = self.treatment_model.predict_proba(x)[:, 1] - d
        except:
            errors = self.treatment_model.predict(x) - d

        # Get propensity score and generate prediction data
        propensity_scores = norm.pdf(errors, loc=0, scale=self.treatment_error_scale)

        # Train data (propensity score, t)
        pd = np.column_stack((propensity_scores, d))

        # Transform data
        pd = self.outcome_feature_transformer.fit_transform(pd)

        # Predict
        try:
            y_hat = self.outcome_model.predict_proba(pd)[:, 1]
        except:
            y_hat = self.outcome_model.predict(pd)

        return y_hat