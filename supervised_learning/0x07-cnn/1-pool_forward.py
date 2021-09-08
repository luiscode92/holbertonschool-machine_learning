#!/usr/bin/env python3
"""
Performs forward propagation over a pooling layer of a neural network
"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Returns: the output of the pooling layer
    """
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    h_new = int(((h_prev - kh) / sh) + 1)
    w_new = int(((w_prev - kw) / sw) + 1)

    A_new = np.zeros((m, h_new, w_new, c_prev))

    for x in range(w_new):
        for y in range(h_new):
            if mode == 'max':
                A_new[:, y, x, :] = \
                    np.max(A_prev[:,
                                  y * sh: y * sh + kh,
                                  x * sw: x * sw + kw, :], axis=(1, 2))
            if mode == 'avg':
                A_new[:, y, x, :] = \
                    np.mean(A_prev[:,
                                   y * sh: y * sh + kh,
                                   x * sw: x * sw + kw, :], axis=(1, 2))

    return A_new
