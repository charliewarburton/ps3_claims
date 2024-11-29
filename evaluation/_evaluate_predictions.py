import numpy as np
import pandas as pd
from typing import Optional, Union
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, auc


def evaluate_predictions(
    predictions: ArrayLike,
    actuals: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
) -> pd.DataFrame:
    """
    Takes a list of predictions and actual values and computes a set of evaluation metrics.

    Parameters:
        predictions: Array of predicted values
        actuals: Array of actual values
        sample_weight: Array of sample weights. Default is None. Defined as Exposure in script.py

    Returns:
        metrics_df: DataFrame of evaluation metrics and their values
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    if sample_weight is not None:
        # Bias (deviation from actual exposure adjusted mean)
        bias = np.average(predictions - actuals, weights=sample_weight)

        # Deviance
        deviance = np.average((predictions - actuals) ** 2, weights=sample_weight)

        # Mean Absolute Error (MAE)
        MAE = mean_absolute_error(actuals, predictions, sample_weight=sample_weight)

        # Mean Squared Error (MSE)
        MSE = mean_squared_error(actuals, predictions, sample_weight=sample_weight)

        # Root Mean Squared Error (RMSE)
        RMSE = np.sqrt(MSE)

        # Gini coefficient calculation
        # Sort by predictions
        order = np.argsort(predictions)
        ordered_actuals = actuals[order]
        ordered_weights = sample_weight[order]

        # Calculate cumulative proportions
        cum_actuals = np.cumsum(ordered_actuals * ordered_weights)
        cum_actuals = cum_actuals / cum_actuals[-1]
        cum_population = np.linspace(0, 1, len(cum_actuals))

        # Calculate Gini coefficient
        gini = 1 - 2 * auc(cum_population, cum_actuals)

    else:
        # If no weights are provided, compute the simple mean
        bias = np.mean(predictions - actuals)
        deviance = np.mean((predictions - actuals) ** 2)
        MAE = mean_absolute_error(actuals, predictions)
        MSE = mean_squared_error(actuals, predictions)
        RMSE = np.sqrt(MSE)

        # Gini coefficient calculation without weights
        order = np.argsort(predictions)
        ordered_actuals = actuals[order]
        cum_actuals = np.cumsum(ordered_actuals)
        cum_actuals = cum_actuals / cum_actuals[-1]
        cum_population = np.linspace(0, 1, len(cum_actuals))
        gini = 1 - 2 * auc(cum_population, cum_actuals)

    metrics = {
        "Bias": bias,
        "Deviance": deviance,
        "MAE": MAE,
        "MSE": MSE,
        "RMSE": RMSE,
        "Gini": gini,
    }

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    metrics_df["Value"] = metrics_df["Value"].round(decimals=2)

    return metrics_df
