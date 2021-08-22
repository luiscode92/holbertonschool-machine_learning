#!/usr/bin/env python3
"""
Calculates the normalization (standardization) constants of a matrix
"""


import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix
    """
    mean = np.mean(X, axis=0)
    stddev = np.std(X, axis=0)
    return mean, stddev
