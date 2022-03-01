#!/usr/bin/env python3
"""
Calculates the positional encoding for a transformer
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Returns: a numpy.ndarray of shape (max_seq_len, dm)
    containing the positional encoding vectors
    """
    PE = np.zeros([max_seq_len, dm])

    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            # sin to even indices
            PE[pos, i] = np.sin(pos / (10000 ** (i / dm)))
            # cos to odd indices
            PE[pos, i + 1] = np.cos(pos / (10000 ** (i / dm)))

    return PE
