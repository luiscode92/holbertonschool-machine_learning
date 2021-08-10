#!/usr/bin/env python3
"""
Concatenates two arrays
"""


def cat_arrays(arr1, arr2):
    """Concatenates two arrays"""
    new_arr = []
    new_arr.extend(arr1)
    new_arr.extend(arr2)
    return new_arr
