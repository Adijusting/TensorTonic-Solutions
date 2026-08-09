import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    p = p.flatten()
    y = y.flatten()

    dice = (2*np.sum(p*y) + eps) / (np.sum(p) + np.sum(y) + eps)

    d_loss = 1-dice

    return d_loss
    pass