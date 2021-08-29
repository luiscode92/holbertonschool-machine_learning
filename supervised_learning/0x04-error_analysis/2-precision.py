#!/usr/bin/env python3
"""
Calculates the precision for each class in a confusion matrix
"""


import numpy as np


def precision(confusion):
    """
    Returns: a numpy.ndarray of shape (classes,)
    containing the precision of each class
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=0)
