#!/usr/bin/env python3
"""
Calculates the maximization step in the EM algorithm for a GMM
"""


import numpy as np


def maximization(X, g):
    """
    Returns: pi, m, S, or None, None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None

    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape

    if n != g.shape[1]:
        return None, None, None

    k = g.shape[0]

    # sum of gi equal to 1
    probs = np.sum(g, axis=0)
    validation = np.ones((n,))
    if not np.isclose(probs, validation).all():
        return None, None, None

    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        pi[i] = np.sum(g[i]) / n
        m[i] = np.matmul(g[i], X) / np.sum(g[i])
        S[i] = np.matmul(g[i] * (X - m[i]).T, X - m[i]) / np.sum(g[i])
    return pi, m, S
