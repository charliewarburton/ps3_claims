import hashlib
import pandas as pd
import numpy as np
from typing import Union

def create_sample_split(
    df: pd.DataFrame, id_column: Union[str, int, float], training_frac: float = 0.8
) -> pd.DataFrame:
    """
    Create deterministic train/test splits based on a single ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    id_column : Union[str, int, float]
        Name of the column to base the split on. Can be string or numeric.
    training_frac : float, optional
        Fraction of data to assign to the training set, by default 0.8.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'sample' column indicating 'train' or 'test'.

    Raises
    ------
    ValueError
        If the training fraction is not between 0 and 1.
        If the ID column is not unique.
        If the specified ID column does not exist.
    """
    if not 0 < training_frac < 1:
        raise ValueError("training_frac must be between 0 and 1")

    # Ensure the ID column exists
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in the DataFrame")

    # Check if IDs are unique
    if not df[id_column].is_unique:
        raise ValueError(f"Values in column '{id_column}' are not unique")

    # Check column type and convert if necessary
    if pd.api.types.is_numeric_dtype(df[id_column]):
        # Convert floats to string (leave integers as is)
        id_values = df[id_column].apply(
            lambda x: str(x) if isinstance(x, float) else x
        )
    else:
        id_values = df[id_column].astype(str)

    # Create a stable hash using hashlib
    hash_values = id_values.apply(
        lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16)
    )

    # Sort based on hash values for reproducible splits
    sorted_indices = hash_values.argsort()
    df = df.iloc[sorted_indices].reset_index(drop=True)

    # Calculate split sizes
    n_train = int(len(df) * training_frac)

    # Assign splits directly
    df["sample"] = ["train"] * n_train + ["test"] * (len(df) - n_train)

    return df
