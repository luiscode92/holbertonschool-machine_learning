#!/usr/bin/env python3
"""
Concatenates two matrices along a specific axis
"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix"""
    shape = []
    if type(matrix) is list:
        shape.append(len(matrix))
        shape += matrix_shape(matrix[0])

    return shape


def recursion(mat1, mat2, iter_axis, axis):
    """Creates concatenation"""
    if iter_axis == axis:
        concat = mat1[:] + mat2[:]
        return concat
    new_dim = []
    for x in zip(mat1, mat2):
        new_dim.append(recursion(x[0], x[1], iter_axis + 1, axis))
    return new_dim


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis"""
    checks = [
        cond[0] == cond[1] if ax != axis else True
        for ax, cond in enumerate(zip(matrix_shape(mat1), matrix_shape(mat2)))
        ]

    if all(checks):
        return (recursion(mat1, mat2, 0, axis))
