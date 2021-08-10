#!/usr/bin/env python3
"""
Returns the transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix"""
    ext = []
    for i in range(len(matrix[0])):
        interna = []
        for j in range(len(matrix)):
            interna.append(matrix[j][i])
        ext.append(interna)
    return ext
