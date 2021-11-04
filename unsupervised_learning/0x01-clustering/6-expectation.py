#!/usr/bin/env python3
"""
Calculates the expectation step in the EM algorithm for a GMM
"""


import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Returns: g, l, or None, None on failure
    - g is a numpy.ndarray of shape (k, n) containing
      the posterior probabilities for each data point in each cluster
    - l is the total log likelihood
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None

    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None

    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None

    n, d = X.shape

    if d != S.shape[1] or S.shape[1] != S.shape[2]:
        return (None, None)

    if d != m.shape[1] or m.shape[0] != S.shape[0]:
        return (None, None)

    if pi.shape[0] != m.shape[0]:
        return (None, None)

    # Priors must sum to 1
    if not np.isclose(np.sum(pi), 1):
        return None, None

    k = S.shape[0]
    tmp = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])
        tmp[i] = pi[i] * P
    g = tmp / np.sum(tmp, axis=0)
    total_log_like = np.sum(np.log(np.sum(tmp, axis=0)))

    return g, total_log_like
