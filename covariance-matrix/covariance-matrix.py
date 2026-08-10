import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.asarray(X, dtype=float)

    # Must be a 2D dataset with at least 2 samples
    if X.ndim != 2 or X.shape[0] < 2:
        return None

    mean = np.mean(X, axis=0)
    X_centered = X - mean

    return (X_centered.T @ X_centered) / (X.shape[0] - 1)
    pass