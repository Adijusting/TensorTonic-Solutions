import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p = np.asarray(p, dtype = float)
    q = np.asarray(q, dtype = float)

    p = p.flatten()
    q = q.flatten()

    d_pq = np.sum(p*np.log(p/q))

    return d_pq
    pass