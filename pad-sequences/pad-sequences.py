import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    result = []

    for seq in seqs:
        new_seq=[]

        for i in range(max_len):
            if i<len(seq):
                new_seq.append(seq[i])
            else:
                new_seq.append(pad_value)

        result.append(new_seq)

    return result
    pass