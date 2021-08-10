#!/usr/bin/env python3
"""
Slices a matrix along a specific axes
"""


def np_slice(matrix, axes={}):
    """Slices a matrix along a specific axes"""
    new_mat = [slice(None)] * len(matrix.shape)
    for key, value in axes.items():
        new_mat[key] = slice(*value)
    return matrix[tuple(new_mat)]
