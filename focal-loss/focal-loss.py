import numpy as np
import math

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    p = np.array(p, dtype=float)
    y = np.array(y, dtype=float)

    p = np.clip(p, 1e-15, 1 - 1e-15)

    term1 = (1 - p) ** gamma * y * np.log(p)
    term2 = p ** gamma * (1 - y) * np.log(1 - p)

    loss = -(term1 + term2)

    return np.mean(loss)
    pass