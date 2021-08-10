#!/usr/bin/env python3
"""
Performs element-wise addition, subtraction, multiplication, and division
"""


def np_elementwise(mat1, mat2):
    """Performs element-wise addition,
    subtraction, multiplication, and division"""
    add = mat1 + mat2
    subtract = mat1 - mat2
    multiply = mat1 * mat2
    divide = mat1 / mat2
    return (add, subtract, multiply, divide)
