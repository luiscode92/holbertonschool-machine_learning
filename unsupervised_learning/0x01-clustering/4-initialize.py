#!/usr/bin/env python3
"""
Initializes variables for a Gaussian Mixture Model
"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Returns: pi, m, S, or None, None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None

    if type(k) is not int or k < 1:
        return None, None, None

    d = X.shape[1]

    pi = np.full((k,), 1 / k)
    m, _ = kmeans(X, k)
    S = np.full((k, d, d), np.identity(d))
    return pi, m, S
