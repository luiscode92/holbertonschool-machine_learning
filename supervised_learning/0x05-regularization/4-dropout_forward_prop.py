#!/usr/bin/env python3
"""
Conducts forward propagation using Dropout
"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout
    """
    cache = {}
    cache['A0'] = X

    for i in range(L):
        W_key = "W{}".format(i + 1)
        b_key = "b{}".format(i + 1)
        A_key_prev = "A{}".format(i)
        A_key_forw = "A{}".format(i + 1)
        D_key = "D{}".format(i + 1)

        Z = np.matmul(weights[W_key], cache[A_key_prev]) \
            + weights[b_key]
        dropout = np.random.binomial(1, keep_prob, size=Z.shape)
        # if it is not the last layer
        if i != L - 1:
            # tanh activation function
            cache[A_key_forw] = np.tanh(Z)
            # dropout mask in layer
            cache[D_key] = dropout
            cache[A_key_forw] *= cache[D_key]
            cache[A_key_forw] /= keep_prob
        # if it is the last layer
        else:
            # softmax activation function for multi-class classification
            # t is a temporary variable
            t = np.exp(Z)
            # normalize
            cache[A_key_forw] = (t / np.sum(t, axis=0, keepdims=True))
    return cache
