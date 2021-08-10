#!/usr/bin/env python3
"""
Converts a numeric label vector into a one-hot matrix
"""


import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix
    - Y is a numpy.ndarray with shape (m,) containing numeric class labels
    * m is the number of examples
    - classes is the maximum number of classes found in Y
    Returns: a one-hot encoding of Y with shape (classes, m),
    or None on failure
    """
    if type(Y) is not np.ndarray or len(Y) < 1:
        return None
    if type(classes) is not int or classes <= np.amax(Y):
        return None
    one_hot = np.zeros((classes, Y.shape[0]))
    col = np.arange(Y.size)
    one_hot[Y, col] = 1
    return one_hot
