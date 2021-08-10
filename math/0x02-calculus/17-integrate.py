#!/usr/bin/env python3
"""
Calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if type(poly) is not list:
        return None

    if type(C) is not int:
        return None

    if len(poly) == 0:
        return None

    if poly == [0]:
        return [C]

    inte = [C]
    for coef in range(len(poly)):
        new_coef = poly[coef]/(coef + 1)
        if (new_coef).is_integer():
            inte.append(round(new_coef))
        else:
            inte.append(new_coef)
    return inte
