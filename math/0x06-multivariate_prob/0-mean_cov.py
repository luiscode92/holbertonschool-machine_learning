#!/usr/bin/env python3
"""
Calculates the mean and covariance of a data set
"""


import numpy as np


def mean_cov(X):
    """
    Returns: mean, cov
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n = X.shape[0]

    if n < 2:
        raise ValueError("X must contain multiple data points")

    d = X.shape[1]

    mean = np.mean(X, axis=0).reshape(1, d)

    deviation = X - mean

    cov = np.matmul(deviation.T, deviation) / (n - 1)

    return mean, cov
