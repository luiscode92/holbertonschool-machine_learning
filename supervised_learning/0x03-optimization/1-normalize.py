#!/usr/bin/env python3
"""
Normalizes (standardizes) a matrix
"""


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix
    """
    Z = (X - m)/s
    return Z
