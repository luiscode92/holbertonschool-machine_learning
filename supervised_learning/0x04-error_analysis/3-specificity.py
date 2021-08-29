#!/usr/bin/env python3
"""
Calculates the specificity for each class in a confusion matrix
"""


import numpy as np


def specificity(confusion):
    """
    Returns: a numpy.ndarray of shape (classes,)
    containing the specificity of each class.
    Specificity measures the proportion of negatives that are
    correctly identified
    """
    TP = np.diag(confusion)
    # True condition positive = TP + FN
    FN = np.sum(confusion, axis=1) - TP
    # Predicted condition positive = TP + FP
    FP = np.sum(confusion, axis=0) - TP
    TN = np.sum(confusion) - TP - FP - FN
    # specificity = TN / (True condition negative)
    Specificity = TN / (FP + TN)

    return Specificity
