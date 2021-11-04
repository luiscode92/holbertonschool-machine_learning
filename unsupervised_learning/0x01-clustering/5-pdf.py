#!/usr/bin/env python3
"""
Calculates the probability density function of a Gaussian distribution
"""


import numpy as np


def pdf(X, m, S):
    """
    Returns: P, or None on failure
    - P is a numpy.ndarray of shape (n,)
      containing the PDF values for each data point
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None

    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None

    d = X.shape[1]

    if d != m.shape[0] or d != S.shape[0]:
        return None

    if S.shape[0] != S.shape[1] or d != S.shape[1]:
        return None

    cov_det = np.linalg.det(S)
    cov_inv = np.linalg.inv(S)

    # denominator
    den = np.sqrt(((2 * np.pi) ** d) * cov_det)

    # exponential term
    expo = (-0.5 * np.sum(np.matmul(cov_inv,
            (X.T - m[:, np.newaxis])) *
            (X.T - m[:, np.newaxis]), axis=0))

    P = np.exp(expo) / den
    P = np.where(P < 1e-300, 1e-300, P)
    return P
