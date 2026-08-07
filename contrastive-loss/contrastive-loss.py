import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    a = np.array(a)
    b = np.array(b)
    y = np.array(y)

    if a.ndim == 1:
        d = np.linalg.norm(a - b)
    else:
        d = np.linalg.norm(a - b, axis=1)
    loss = y*(d**2) + (1-y)*np.maximum(0, margin-d)**2

    if reduction == "sum":
        return float(np.sum(loss))

    return np.mean(loss)
    pass