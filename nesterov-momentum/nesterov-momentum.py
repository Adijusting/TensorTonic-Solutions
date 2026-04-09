import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    # 1. Convert to numpy for vectorized math
    w_arr = np.asarray(w)
    v_arr = np.asarray(v)
    grad_arr = np.asarray(grad)

    # 2. Step 2: Update Velocity
    # v_new = mu * v + lr * g
    v_new = (momentum * v_arr) + (lr * grad_arr)

    # 3. Step 3: Update Weights
    # w_new = w - v_new
    w_new = w_arr - v_new

    # 4. CRITICAL: Return (Weights, Velocity)
    # The error message indicates: expected w [0.95, ...], v [0.05, ...]
    return w_new.tolist(), v_new.tolist()
    pass