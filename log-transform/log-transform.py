import math
def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    result = [ math.log(1+x) for x in values]

    return result