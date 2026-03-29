import numpy as np
import math

def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """
    base_lr, min_lr, total_steps, current_step = np.array(base_lr), np.array(min_lr), np.array(total_steps), np.array(current_step)

    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * current_step/total_steps))

    return lr.tolist()
    pass