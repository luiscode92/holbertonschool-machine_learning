#!/usr/bin/env python3
"""
Performs back propagation over a convolutional layer of a neural network
"""


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Returns: the partial derivatives with respect
    to the previous layer (dA_prev), the kernels (dW), and the biases (db),
    respectively
    """
    m = dZ.shape[0]
    h_new = dZ.shape[1]
    w_new = dZ.shape[2]
    c_new = dZ.shape[3]

    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]

    kh = W.shape[0]
    kw = W.shape[1]

    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1
    if padding == 'valid':
        ph = 0
        pw = 0

    pad_size = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    X_pad = np.pad(A_prev,
                   pad_width=pad_size,
                   mode='constant',
                   constant_values=0)

    dA_prev = np.zeros(X_pad.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        for c in range(c_new):
            for h in range(h_new):
                for w in range(w_new):
                    dA_prev[i, h * sh: h * sh + kh, w * sw: w * sw + kw, :]\
                        += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += X_pad[i,
                                            h * sh: h * sh + kh,
                                            w * sw: w * sw + kw,
                                            :] * dZ[i, h, w, c]

    # subtract padding
    dA_prev = dA_prev[:, ph:dA_prev.shape[1] - ph, pw:dA_prev.shape[2] - pw, :]

    return dA_prev, dW, db
