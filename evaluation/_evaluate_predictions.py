import numpy as np
import pandas as pd

def evaluate_predictions(predictions, actuals, sample_weight=None):
    """
    Takes a list of predictions and actual values and computes a set of evaluation metrics.

    Parameters: 
        predictions (np.array): Array of predicted values
        actuals (np.array): Array of actual values
        sample_weight (np.array): Array of sample weights. Default is None. Defined as Exposure in script.py

    Returns:
        metrics_df (pd.DataFrame): DataFrame of evaluation metrics and their values
    """
    if sample_weight is not None:
        # Bias (deviation from actual exposure adjusted mean)
        bias = np.average(predictions - actuals, weights=sample_weight)

        # Deviance
        deviance = np.average(predictions - actuals, weights=sample_weight)

        # Mean Absolute Error (MAE)
        MAE = np.average(np.abs(predictions - actuals), weights=sample_weight)

        # Mean Squared Error (MSE)
        MSE = np.average((predictions - actuals) ** 2, weights=sample_weight)

        # Root Mean Squared Error (RMSE)
        RMSE = np.sqrt(MSE)
    else:
        # If no weights are provided, compute the simple mean
        bias = np.mean(predictions - actuals)
        deviance = np.mean(predictions - actuals)
        MAE = np.mean(np.abs(predictions - actuals))
        MSE = np.mean((predictions - actuals) ** 2)
        RMSE = np.sqrt(MSE)

    metrics = {
        "Bias": bias,
        "Deviance": deviance,
        "MAE": MAE,
        "MSE": MSE,
        "RMSE": RMSE,
    }

    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])

    metrics_df["Value"] = metrics_df["Value"].round(decimals=2)

    return metrics_df 