#!/usr/bin/env python3
"""
Calculates the cost of a neural network with L2 regularization
"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    - cost is the cost of the network without L2 regularization
    - lambtha is the regularization parameter
    - weights is a dictionary of the weights and biases (numpy.ndarrays)
      of the neural network
    - L is the number of layers in the neural network
    - m is the number of data points used
    - Returns: the cost of the network accounting for L2 regularization
    """
    # Frobenius norm
    sum_frob = 0
    for i in range(L):
        sum_frob = sum_frob + np.linalg. \
            norm(weights['W{}'.format(i+1)], ord='fro')
    L2_cost = cost + (lambtha / (2 * m)) * sum_frob
    return L2_cost
