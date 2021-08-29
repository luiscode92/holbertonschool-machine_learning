#!/usr/bin/env python3
"""
Updates the weights of a neural network with Dropout regularization
using gradient descent
"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent
    """
    weights_c = weights.copy()
    m = Y.shape[1]

    for i in reversed(range(L)):
        # if it is the last layer
        if i == L - 1:
            dZ = cache['A{}'.format(i + 1)] - Y

        # hidden layers
        else:
            dZa = np.matmul(weights_c['W{}'.format(i + 2)].T, dZ)
            # derivative of tanh function
            dZb = 1 - cache['A{}'.format(i + 1)] ** 2
            dZ = dZa * dZb
            # dropout mask
            dZ *= cache["D{}".format(i + 1)] / keep_prob

        dW = ((np.matmul(dZ, cache['A{}'.format(i)].T)) / m)
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights['W{}'.format(i + 1)] = \
            weights_c['W{}'.format(i + 1)] \
            - (alpha * dW)
        weights['b{}'.format(i + 1)] = \
            weights_c['b{}'.format(i + 1)] \
            - (alpha * db)
