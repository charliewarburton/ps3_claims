import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import Union, Optional

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile: float = 0.05, upper_quantile: float = 0.95) -> None:
        """
        Initialize the Winsorizer with lower and upper quantiles.
        
        Parameters
        ----------
        lower_quantile : float, default=0.05
            Lower quantile to clip values at (0 <= lower_quantile <= upper_quantile <= 1).
        upper_quantile : float, default=0.95
            Upper quantile to clip values at.
            
        Raises
        ------
        ValueError
            If quantiles are not between 0 and 1 or lower_quantile > upper_quantile.
        """
        if not 0 <= lower_quantile <= upper_quantile <= 1:
            raise ValueError("Quantiles must satisfy 0 <= lower_quantile <= upper_quantile <= 1.")
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        
    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Validate input array and convert to numpy array if needed."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "Winsorizer":
        """
        Compute the quantiles from the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to compute quantiles on.
            
        Returns
        -------
        self : Winsorizer
            Fitted transformer.
        """
        X = self._validate_input(X)
        self.lower_bound_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_bound_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Clip values according to the computed quantiles.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_clipped : ndarray
            Clipped data.
            
        Raises
        ------
        NotFittedError
            If the transformer has not been fitted yet.
        """
        check_is_fitted(self, ["lower_bound_", "upper_bound_"])
        X = self._validate_input(X)
        return np.clip(X, self.lower_bound_, self.upper_bound_)