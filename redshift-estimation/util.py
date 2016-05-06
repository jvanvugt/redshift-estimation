import numpy as np

def remove_missing(features, labels):
    """
    Remove all rows from the data that have missing data.
    """
    not_missing_X = ~np.isnan(features).any(axis=1)
    not_missing_y = ~np.isnan(labels).any()

    not_missing = np.logical_and(not_missing_X, not_missing_y)
    return features[not_missing, :], labels[not_missing]
