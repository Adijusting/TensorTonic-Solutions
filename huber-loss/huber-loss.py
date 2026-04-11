import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    losses = []
    
    for yt, yp in zip(y_true, y_pred):
        e = yt - yp
        abs_e = abs(e)
        
        if abs_e <= delta:
            loss = 0.5 * e * e
        else:
            loss = delta * (abs_e - 0.5 * delta)
        
        losses.append(loss)

    return np.mean(losses)
    pass