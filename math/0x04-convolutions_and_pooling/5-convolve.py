#!/usr/bin/env python3
"""
Performs a convolution on images using multiple kernels
"""


import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Returns: a numpy.ndarray containing the convolved images
    """
    input_w, input_h, m = images.shape[2], images.shape[1], images.shape[0]
    nc, kw, kh = kernels.shape[3], kernels.shape[1], kernels.shape[0]
    sw, sh = stride[1], stride[0]

    if padding == 'same':
        ph = int(((input_h - 1) * sh + kh - input_h) / 2) + 1
        pw = int(((input_w - 1) * sw + kw - input_w) / 2) + 1
    if padding == 'valid':
        ph = 0
        pw = 0
    if type(padding) is tuple:
        pw, ph = padding[1], padding[0]

    pad_size = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    images_padded = np.pad(images,
                           pad_width=pad_size,
                           mode='constant',
                           constant_values=0)

    output_h = int(((input_h + 2 * ph - kh) / sh) + 1)
    output_w = int(((input_w + 2 * pw - kw) / sw) + 1)

    output = np.zeros((m, output_h, output_w, nc))

    for x in range(output_w):
        for y in range(output_h):
            for z in range(nc):
                output[:, y, x, z] = \
                    (kernels[:, :, :, z] * images_padded[:,
                     y * sh: y * sh + kh,
                     x * sw: x * sw + kw,
                     :]).sum(axis=(1, 2, 3))

    return output
