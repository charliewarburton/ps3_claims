import numpy as np
import pandas as pd
from typing import Optional, Union
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, auc

# TODO: Write a function evaluate_predictions within the evaluation module,
# which computes various metrics given the true outcome values and the model's predictions.
# Steps:
# 1. Create a module folder 'evaluation' and an empty '__init__.py' which we will
#    use to register the function at the module level.
# 2. Create a new file '_evaluate_predictions.py' in which you create the respective
#    function which takes the predictions and actuals as input, as well as some
#    sample weight (in our case exposure).
# 3. Compute the bias of your estimates as deviation from the actual exposure
#    adjusted mean.
# 4. Compute the deviance.
# 5. Compute the MAE and RMSE.
# 6. Bonus: Compute the Gini coefficient as defined in the plot of the Lorenz
#    curve at the bottom of ps3_script.
# 7. Return a dataframe with the names of the metrics as index.
# 8. Use the function and compare the constrained and unconstrained LGBM models.
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

    else:
        # If no weights are provided, compute the simple mean
        bias = np.mean(predictions - actuals)
        deviance = np.mean((predictions - actuals) ** 2)
        MAE = mean_absolute_error(actuals, predictions)
        MSE = mean_squared_error(actuals, predictions)
        RMSE = np.sqrt(MSE)

    metrics = {
        "Bias": bias,
        "Deviance": deviance,
        "MAE": MAE,
        "MSE": MSE,
        "RMSE": RMSE,
    }

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])

    metrics_df["Value"] = metrics_df["Value"].round(decimals=2)

    return metrics_df
