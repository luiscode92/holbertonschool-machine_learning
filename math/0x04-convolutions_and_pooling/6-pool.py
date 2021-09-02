#!/usr/bin/env python3
"""
Performs pooling on images
"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Returns: a numpy.ndarray containing the pooled images
    """
    input_w, input_h, m = images.shape[2], images.shape[1], images.shape[0]
    c = images.shape[3]
    kw, kh = kernel_shape[1], kernel_shape[0]
    sw, sh = stride[1], stride[0]

    output_h = int(((input_h - kh) / sh) + 1)
    output_w = int(((input_w - kw) / sw) + 1)

    output = np.zeros((m, output_h, output_w, c))

    for x in range(output_w):
        for y in range(output_h):
            if mode == 'max':
                output[:, y, x, :] = \
                    np.max(images[:,
                                  y * sh: y * sh + kh,
                                  x * sw: x * sw + kw, :], axis=(1, 2))
            if mode == 'avg':
                output[:, y, x, :] = \
                    np.mean(images[:,
                                   y * sh: y * sh + kh,
                                   x * sw: x * sw + kw, :], axis=(1, 2))

    return output
