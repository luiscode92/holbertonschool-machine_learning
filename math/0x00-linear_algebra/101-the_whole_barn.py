#!/usr/bin/env python3
"""
Adds two matrices
"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix"""
    shape = []
    if type(matrix) is list:
        shape.append(len(matrix))
        shape += matrix_shape(matrix[0])

    return shape


def recursion(mat1, mat2):
    """Accesses the primitives"""
    if (type(mat1) and type(mat2)) == list:
        result = []
        for x in zip(mat1, mat2):
            result.append(recursion(x[0], x[1]))
        return result
    else:
        return (mat1+mat2)


def add_matrices(mat1, mat2):
    """Adds two matrices"""
    checks = [
        cond[0] == cond[1]
        for cond in zip(matrix_shape(mat1), matrix_shape(mat2))
        ]

    if all(checks):
        return (recursion(mat1, mat2))
