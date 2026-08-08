def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    
    """
    total_loss = 0.0

    for p,y in zip(predictions, targets):
        if y==1:
            p_t = p
        else:
            p_t = 1 - p

        import math
        f_loss = -alpha*(1-p_t)**gamma*math.log(p_t)
        total_loss += f_loss

    return total_loss / len(targets)
    
    