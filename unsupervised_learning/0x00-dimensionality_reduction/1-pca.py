#!/usr/bin/env python3
"""
Performs PCA on a dataset
"""


import numpy as np


def pca(X, ndim):
    """
    Returns: T, a numpy.ndarray of shape (n, ndim)
    containing the transformed version of X
    """
    X_m = X - np.mean(X, axis=0)
    _, _, vh = np.linalg.svd(X_m)
    W = vh[:ndim].T
    T = np.matmul(X_m, W)

    return T
