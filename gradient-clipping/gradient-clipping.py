import numpy as np

def clip_gradients(gradients, max_norm):
    grads = [np.asarray(g, dtype=float) for g in gradients]

    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))

    if max_norm > 0 and total_norm > max_norm:
        scale = max_norm / total_norm
        grads = [g * scale for g in grads]

    return np.array(grads, dtype=object)