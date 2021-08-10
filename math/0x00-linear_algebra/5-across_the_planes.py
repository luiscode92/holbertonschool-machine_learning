#!/usr/bin/env python3
"""
Adds two matrices in 2D element-wise
"""


def add_matrices2D(mat1, mat2):
    """Adds two matrices in 2D element-wise"""
    externa = []
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        for i in range(len(mat1)):
            interna = []
            for j in range(len(mat1[0])):
                interna.append(mat1[i][j] + mat2[i][j])
            externa.append(interna)
        return externa
    else:
        return None
