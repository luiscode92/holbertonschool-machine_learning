#!/usr/bin/env python3
"""
Concatenates two matrices along a specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis"""
    copy_mat1 = [row[:] for row in mat1]
    copy_mat2 = [row[:] for row in mat2]

    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        copy_mat1.extend(copy_mat2)
        return copy_mat1
    if axis == 1 and len(mat1) == len(mat2):
        new_mat = []
        for i in range(len(mat1)):
            new_mat.append(copy_mat1[i] + copy_mat2[i])
        return new_mat
    else:
        return None
