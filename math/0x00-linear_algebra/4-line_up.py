#!/usr/bin/env python3
"""
Adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise"""
    arr_sum = []
    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            arr_sum.append(arr1[i] + arr2[i])
        return arr_sum
    else:
        return None
