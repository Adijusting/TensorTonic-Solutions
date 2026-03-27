import numpy as np
def elu(x, alpha=1.0):
    """
    Apply ELU activation to each element.
    """
    x=np.asarray(x)
    return_arr=np.where(x>0,x,alpha*(np.exp(x)-1))
    return return_arr.tolist()