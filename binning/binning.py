import numpy as np
def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return [0] * len(values)

    width = (max_val - min_val) / num_bins

    result = [
        min(int((x - min_val) / width), num_bins - 1)
        for x in values
    ]

    return result

    