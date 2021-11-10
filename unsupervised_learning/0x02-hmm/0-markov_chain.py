#!/usr/bin/env python3
"""
Determines the probability of a markov chain being in a particular state
after a specified number of iterations
"""


import numpy as np


def markov_chain(P, s, t=1):
    """
    Returns: a numpy.ndarray of shape (1, n) representing
    the probability of being in a specific state
    after t iterations, or None on failure
    """
    if len(P.shape) != 2 or P.shape[0] != P.shape[1] or P.shape[0] < 1:
        return None

    if len(s.shape) != 2 or s.shape[1] != P.shape[0]:
        return None

    if type(t) is not int or t < 1:
        return None

    final = np.matmul(s, np.linalg.matrix_power(P, t))

    return final
