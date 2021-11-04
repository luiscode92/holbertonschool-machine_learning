#!/usr/bin/env python3
"""
Initializes cluster centroids for K-means
"""


import numpy as np


def initialize(X, k):
    """
    Returns: a numpy.ndarray of shape (k, d)
    containing the initialized centroids for each cluster, or None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    if type(k) is not int or k <= 0 or k >= X.shape[0]:
        return None

    d = X.shape[1]
    mini = np.amin(X, axis=0)
    maxi = np.amax(X, axis=0)

    # Initialize cluster centroids
    init = np.random.uniform(mini, maxi, size=(k, d))

    return init
