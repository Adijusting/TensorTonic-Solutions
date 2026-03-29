def warmup_decay_schedule(base_lr, warmup_steps, total_steps, current_step):
    """
    Compute the learning rate at a given step using warmup + linear decay.
    """
    if current_step < warmup_steps:
        if warmup_steps==0:
            return float(base_lr)
        return float(base_lr*current_step/warmup_steps)

    else:
        steps_left = total_steps - current_step
        decay_period = total_steps - warmup_steps

        if steps_left <=0:
            return 0.0

        return float(base_lr * (steps_left/decay_period))

    