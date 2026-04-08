import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    """
    Perform one AdaDelta update step.
    """
    w = np.asarray(w, dtype=float)
    grad = np.asarray(grad, dtype=float)
    E_grad_sq = np.asarray(E_grad_sq, dtype=float)
    E_update_sq = np.asarray(E_update_sq, dtype=float)
    
    E_grad_sq_t = rho * E_grad_sq + (1 - rho) * (grad ** 2)

    delta_w = - (np.sqrt(E_update_sq + eps) / np.sqrt(E_grad_sq_t + eps)) * grad

    E_update_sq_t = rho * E_update_sq + (1 - rho) * (delta_w ** 2)

    w_t = w + delta_w

    return w_t, E_grad_sq_t, E_update_sq_t
    pass