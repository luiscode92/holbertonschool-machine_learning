#!/usr/bin/env python3
"""
Performs a convolution on grayscale images with custom padding
"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Returns: a numpy.ndarray containing the convolved images
    """
    input_w, input_h, m = images.shape[2], images.shape[1], images.shape[0]
    filter_w, filter_h = kernel.shape[1], kernel.shape[0]
    pw, ph = padding[1], padding[0]

    output_h = input_h + 2 * ph - filter_h + 1
    output_w = input_w + 2 * pw - filter_w + 1

    pad_size = ((0, 0), (ph, ph), (pw, pw))
    images_padded = np.pad(images,
                           pad_width=pad_size,
                           mode='constant',
                           constant_values=0)

    output = np.zeros((m, output_h, output_w))

    for x in range(output_w):
        for y in range(output_h):
            output[:, y, x] = (kernel * images_padded[:, y: y + filter_h,
                               x: x + filter_w]).sum(axis=(1, 2))

    return output
