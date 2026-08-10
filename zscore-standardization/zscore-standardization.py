import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    X = np.array(X, dtype=float)

    X_mean = np.mean(X, axis=axis, keepdims=True)
    X_std = np.std(X, axis=axis, keepdims=True)

    return (X - X_mean)/(X_std + eps)
    pass