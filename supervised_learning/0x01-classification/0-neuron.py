#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification
"""


import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """
        Class constructor
        nx is the number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # weights vector for the neuron
        # default mean is 0
        # default stddev is 1
        self.W = np.random.normal(size=(1, nx))
        # bias for the neuron
        self.b = 0
        # activated output of the neuron (prediction)
        self.A = 0
