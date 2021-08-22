#!/usr/bin/env python3
"""
Normalizes an unactivated output of a neural network using batch normalization
"""


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    - Z is a numpy.ndarray of shape (m, n) that should be normalized
    * m is the number of data points
    * n is the number of features in Z
    - gamma is a numpy.ndarray of shape (1, n) containing
      the scales used for batch normalization
    - beta is a numpy.ndarray of shape (1, n) containing
      the offsets used for batch normalization
    - epsilon is a small number used to avoid division by zero
    - Returns: the normalized Z matrix
    """
    # mean of each feature
    mu = np.mean(Z, axis=0)
    # variance of each future
    var = np.var(Z, axis=0)
    # Z normalized
    Z_norm = (Z - mu) / ((var + epsilon) ** 0.5)
    # learnable parameters of model gamma and beta
    """Do not force to mean 0 and variance 1
    in order to take advantage of the non-linearity of, for instance,
    the sigmoid activation function"""
    Z_tilde = gamma * Z_norm + beta
    return Z_tilde
