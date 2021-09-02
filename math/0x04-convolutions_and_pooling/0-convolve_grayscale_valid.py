#!/usr/bin/env python3
"""
Performs a valid convolution on grayscale images
"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Returns: a numpy.ndarray containing the convolved images
    """
    input_w, input_h, m = images.shape[2], images.shape[1], images.shape[0]
    filter_w, filter_h = kernel.shape[1], kernel.shape[0]

    output_h = input_h - filter_h + 1
    output_w = input_w - filter_w + 1

    output = np.zeros((m, output_h, output_w))

    for x in range(output_w):
        for y in range(output_h):
            output[:, y, x] = (kernel * images[:, y: y + filter_h,
                               x: x + filter_w]).sum(axis=(1, 2))

    return output
