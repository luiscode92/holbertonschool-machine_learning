#!/usr/bin/env python3
"""
Calculates the determinant of a matrix
"""


def determinant(matrix):
    """
    Returns: the determinant of matrix
    """

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    # Base case
    if len(matrix) == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return (a * d) - (b * c)

    # recursion
    det = 0
    for i, k in enumerate(matrix[0]):
        rows = [row for row in matrix[1:]]
        new_m = \
            [[row[n] for n in range(len(matrix)) if n != i] for row in rows]
        det += k * (-1) ** i * determinant(new_m)

    return det
