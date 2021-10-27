#!/usr/bin/env python3
"""
Performs PCA on a dataset
"""


import numpy as np


def pca(X, var=0.95):
    """
    Returns: the weights matrix, W,
    that maintains var fraction of X‘s original variance
    """
    _, s, vh = np.linalg.svd(X)
    r = np.argmax(np.cumsum(s) > np.sum(s) * var)
    W = vh[:r + 1].T
    return W
