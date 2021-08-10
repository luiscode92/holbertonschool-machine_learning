#!/usr/bin/env python3
"""
Calculates the shape or size of a matrix:
"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix"""
    shape = []
    if type(matrix) is list:
        shape.append(len(matrix))
        shape += matrix_shape(matrix[0])

    return shape
