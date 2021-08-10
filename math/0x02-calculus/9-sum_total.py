#!/usr/bin/env python3
"""
Calculates the sum of squares
"""


def summation_i_squared(n):
    """Calculates the sum of squares"""
    if type(n) != int or n <= 0:
        return None
    return int(n*(n+1)*(2*n+1)/6)
