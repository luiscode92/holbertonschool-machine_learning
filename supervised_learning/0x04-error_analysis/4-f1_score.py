#!/usr/bin/env python3
"""
Calculates the F1 score of a confusion matrix
"""


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Returns: a numpy.ndarray of shape (classes,)
    containing the F1 score of each class.
    The F1 score is the harmonic mean of the
    precision and recall.
    """
    F1_score = 2 * precision(confusion) * sensitivity(confusion) / \
        (precision(confusion) + sensitivity(confusion))

    return F1_score
