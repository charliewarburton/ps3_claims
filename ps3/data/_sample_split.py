import hashlib

import numpy as np

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """

    # After some initial investigation, the column IDpol is a good candidate for the ID column.
    # This is of type int64

    # List of ID's for the training set
    all_ids = df[id_column].unique()
    np.random.seed(42)
    training_ids = np.random.choice(
        all_ids, size=int(len(all_ids) * training_frac), replace=False
    )

    # Create a new column in the dataframe to indicate if the row is in the training set
    # Set equal to 'train' if in training set, 'test' otherwise
    df["sample"] = np.where(df[id_column].isin(training_ids) == True, "train", "test")
    return df
