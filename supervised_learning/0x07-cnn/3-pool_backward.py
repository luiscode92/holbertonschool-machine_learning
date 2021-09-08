#!/usr/bin/env python3
"""
Performs back propagation over a pooling layer of a neural network
"""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Returns: the partial derivatives with respect to the previous layer
    (dA_prev)
    """
    m = dA.shape[0]
    h_new = dA.shape[1]
    w_new = dA.shape[2]
    c = dA.shape[3]

    # h_prev = A_prev.shape[1]
    # w_prev = A_prev.shape[2]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for cn in range(c):
            for h in range(h_new):
                for w in range(w_new):
                    if mode == 'max':
                        aux = A_prev[i,
                                     h * sh: h * sh + kh,
                                     w * sw: w * sw + kw,
                                     cn]
                        mask = aux == np.max(aux)
                        dA_prev[i,
                                h * sh: h * sh + kh,
                                w * sw: w * sw + kw,
                                cn] += dA[i, h, w, cn] * mask
                    if mode == 'avg':
                        dA_prev[i,
                                h * sh: h * sh + kh,
                                w * sw: w * sw + kw,
                                cn] += dA[i, h, w, cn] / (kh * kw)

    return dA_prev
