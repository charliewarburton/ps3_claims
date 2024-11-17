import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

        # Bounds for the winsorization based on data
        self.lower_bound = None
        self.upper_bound = None
        pass

    def fit(self, X, y=None):
        """
        Fit is used to compute the lower and upper bounds for each feature in the dataset.
        """
        # Ensure X is a numpy array or pandas DataFrame. If not convert it to a numpy array.
        X = np.array(X) if not isinstance(X, (np.ndarray, pd.DataFrame)) else X

        # Compute the lower and upper bounds for each feature
        self.lower_bound_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_bound_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        """
        Apply winsorization to the input data.
        """
        # Ensure X is a numpy array or pandas DataFrame
        X = np.array(X) if not isinstance(X, (np.ndarray, pd.DataFrame)) else X

        # Apply winsorization: clip values to the computed bounds
        X_winsorized = np.clip(X, self.lower_bound_, self.upper_bound_)

        return X_winsorized
