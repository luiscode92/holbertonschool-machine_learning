#!/usr/bin/env python3
"""
Calculates the sensitivity for each class in a confusion matrix
"""


import numpy as np


def sensitivity(confusion):
    """
    Returns: a numpy.ndarray of shape (classes,)
    containing the sensitivity of each class
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=1)
