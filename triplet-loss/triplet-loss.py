import math

def triplet_loss(anchor, positive, negative, margin=1.0):
    # Handle a single triplet
    if len(anchor) > 0 and not isinstance(anchor[0], (list, tuple)):
        d_ap = sum((a - p) ** 2 for a, p in zip(anchor, positive))
        d_an = sum((a - n) ** 2 for a, n in zip(anchor, negative))

        return max(0, d_ap - d_an + margin)

    # Handle multiple triplets
    total_loss = 0.0

    for a, p, n in zip(anchor, positive, negative):
        d_ap = sum((x - y) ** 2 for x, y in zip(a, p))
        d_an = sum((x - y) ** 2 for x, y in zip(a, n))

        total_loss += max(0, d_ap - d_an + margin)

    return total_loss / len(anchor)