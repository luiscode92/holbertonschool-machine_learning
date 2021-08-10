#!/usr/bin/env python3
"""
Calculates the derivative of a polynomial
"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if type(poly) is not list:
        return None

    if len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    deri = []
    for coef in range(len(poly)):
        deri.append(coef * poly[coef])
    deri.pop(0)
    return deri
