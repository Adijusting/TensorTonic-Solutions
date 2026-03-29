import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w, g, s = np.asarray(w), np.asarray(g), np.asarray(s)

    s_t = beta*s+(1-beta)*g*g

    w_t = w-lr/(np.sqrt(s_t+eps))*g

    return w_t.tolist(), s_t.tolist()
    pass