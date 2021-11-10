#!/usr/bin/env python3
"""
Determines the steady state probabilities of a regular markov chain
"""


import numpy as np


def regular(P):
    """
    Returns: a numpy.ndarray of shape (1, n) containing
    the steady state probabilities, or None on failure
    """
    if len(P.shape) != 2 or P.shape[0] != P.shape[1] or P.shape[0] < 1:
        return None

    P = np.linalg.matrix_power(P, 100)

    if np.any(P <= 0):
        return None

    steady_state = np.array([P[0]])
    return steady_state
