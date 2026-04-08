import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    w = np.asarray(w, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    # Update moments (no bias correction)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    # Decoupled weight decay (applied first)
    w = w - lr * weight_decay * w

    # Adam update
    w = w - lr * (m / (np.sqrt(v) + eps))

    return w, m, v