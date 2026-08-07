import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    correct_prob = y_pred[np.arange(n), y_true]

    loss = -np.mean(np.log(correct_prob))
    return loss
    pass