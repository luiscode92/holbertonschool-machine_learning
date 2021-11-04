#!/usr/bin/env python3
"""
Finds the best number of clusters for a GMM
using the Bayesian Information Criterion
"""


import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Returns: best_k, best_result, l, b, or None, None, None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None

    n, d = X.shape

    if type(kmin) is not int or kmin <= 0 or kmin >= n:
        return None, None, None, None

    if type(kmax) is not int or kmax <= 0 or kmax >= n:
        return None, None, None, None

    if kmin >= kmax:
        return None, None, None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None

    if type(tol) is not float or tol <= 0:
        return None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None

    best_k = []

    best_result = []

    log_like = []

    b = []

    for k in range(kmin, kmax + 1):
        best_k.append(k)
        pi, m, S, _, total_log_like = expectation_maximization(X,
                                                               k,
                                                               iterations,
                                                               tol,
                                                               verbose)

        best_result.append((pi, m, S))
        log_like.append(total_log_like)
        p = (k * d * (d + 1) / 2) + (d * k) + k - 1
        BIC = p * np.log(n) - 2 * total_log_like
        b.append(BIC)

    log_like = np.array(log_like)
    b = np.array(b)
    index = np.argmin(b)

    return best_k[index], best_result[index], log_like, b
