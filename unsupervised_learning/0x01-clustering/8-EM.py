#!/usr/bin/env python3
"""
Performs the expectation maximization for a GMM
"""


import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Returns: pi, m, S, g, l, or None, None, None, None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None

    if type(k) is not int or k <= 0 or X.shape[0] <= k:
        return None, None, None, None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None

    if type(tol) is not float or tol <= 0:
        return None, None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)

    g, total_log_like = expectation(X, pi, m, S)

    prev_like = 0

    text = "Log Likelihood after {} iterations: {}"

    for i in range(iterations):
        if verbose and i % 10 == 0:
            print(text.format(i, total_log_like.round(5)))

        pi, m, S = maximization(X, g)
        g, total_log_like = expectation(X, pi, m, S)

        if abs(prev_like - total_log_like) <= tol:
            break

        prev_like = total_log_like

    if verbose:
        print(text.format(i + 1, total_log_like.round(5)))

    return pi, m, S, g, total_log_like
