#!/usr/bin/env python3
"""
Creates a confusion matrix
"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Returns a confusion matrix
    """
    confusion = np.matmul(labels.T, logits)
    return confusion
