# LOAD MODULES
# Standard library
from typing import Optional

# Third party
import numpy as np
import cvxopt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

from pygam import LinearGAM

class Regressor:
    """
    Dummy regressor class. Used to illustrate the structure of the code.
    Must contain a fit and predict method.
    """
    def fit(
        self, 
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Fits the regressor to the data.
        """
        ...
        
    def predict(
        self, 
        x: np.ndarray,
    ) -> None:
        """
        Predicts the outcome for the data.
        """
        ...

class LinearRegression(Regressor):
    """
    Base regressor class to be used in hybrid models.
    """
    def __init__(
        self,
        fit_intercept: bool=True,
        penalty: Optional[str] = None,
    ) -> None:
        """
        Initializes a new instance of the class.

        This method initializes a new instance of the class with the specified parameters.

        Parameters:
            fit_intercept (bool, optional): Whether to calculate the intercept for this model. Defaults to True.
            penalty (str, optional): The type of regularization to be applied. Can be "elastic_net" or "sqrt_lasso". Defaults to None.
        """
        # Save parameters
        self.penalty = penalty
        # Feature transformer
        self.feat_transform = PolynomialFeatures(degree=1,
                                                 include_bias=fit_intercept)
        
    def fit(
        self, 
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Fits the regressor to the data.
        """

        # Transform x
        x = self.feat_transform.fit_transform(x)
        # Train model
        if self.penalty is not None:
            self.model = sm.OLS(y, x).fit_regularized(method=self.penalty)
        else:
            self.model = sm.OLS(y, x).fit()
            
    def predict(
        self, 
        x: np.ndarray,
    ):
        """
        Predicts the outcome for the data.
        """

        # Transform x
        x = self.feat_transform.fit_transform(x)
        
        return self.model.predict(x)

class LogisticRegression(Regressor):
    """
    Base regressor class to be used in hybrid models.
    """
    def __init__(
        self,
        fit_intercept: bool=True,
        penalty: Optional[str] = None,
    ) -> None:
        """
        Initializes a new instance of the class.

        This method initializes a new instance of the class with the specified parameters.

        Parameters:
            fit_intercept (bool, optional): Whether to calculate the intercept for this model. Defaults to True.
            penalty (str, optional): The type of regularization to be applied. Can be "elastic_net" or "sqrt_lasso". Defaults to None.
        """
        # Save parameters
        self.penalty = penalty
        # Feature transformer
        self.feat_transform = PolynomialFeatures(degree=1,
                                                 include_bias=fit_intercept)
        
    def fit(
        self, 
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Fits the regressor to the data.
        """

        # Transform x
        x = self.feat_transform.fit_transform(x)
        # Train model
        if self.penalty is not None:
            self.model = sm.Logit(y, x).fit_regularized(method=self.penalty,
                                                        disp=0)
        else:
            self.model = sm.Logit(y, x).fit(disp=0)
            
    def predict(
        self, 
        x: np.ndarray,
    ):
        """
        Predicts the outcome for the data.
        """

        # Transform x
        x = self.feat_transform.fit_transform(x)
        
        return self.model.predict(x)

class GeneralizedAdditiveModel(Regressor):
    """
    Base regressor class to be used in hybrid models.
    """
    def __init__(
        self,
        num_configurations: int=100,
        verbose: bool=False,
    ) -> None:
        """
        Initializes a new instance of the class.
        
        Parameters:
            num_configurations (int, optional): The number of configurations to be used in the random search. Defaults to 100.
            
        """
        # Save settings
        self.num_configurations = num_configurations
        
        self.model = LinearGAM(verbose=verbose)

    def fit(
        self, 
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Fits the regressor to the data.
        """
        # Lambda combs for the GAM
        lams = np.random.rand(self.num_configurations, x.shape[1])
        lams = lams * 6 - 3
        lams = np.exp(lams)
        
        self.model.gridsearch(x, y, lam=lams, progress=False)
        
    def predict(
        self, 
        x: np.ndarray,
    ) -> None:
        """
        Predicts the outcome for the data.
        """
        return self.model.predict(x)
