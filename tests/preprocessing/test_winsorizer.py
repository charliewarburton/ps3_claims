import numpy as np
import pytest

from ps3.preprocessing._winsorizer import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)

    winsorizer = Winsorizer(
        lower_quantile=lower_quantile, upper_quantile=upper_quantile
    )
    # Fit calculates the bounds based on the quartiles
    winsorizer.fit(X)

    # Transform the data - clip values to the computed bounds
    X_winsorized = winsorizer.transform(X)

    # Compute expected bounds
    lower_bound = np.quantile(X, lower_quantile)
    upper_bound = np.quantile(X, upper_quantile)

    # Assert that all values are within the expected bounds
    assert np.all(X_winsorized >= lower_bound), "Some values are below the lower bound"
    assert np.all(X_winsorized <= upper_bound), "Some values are above the upper bound"
