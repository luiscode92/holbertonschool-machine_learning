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
        self.__W = np.random.normal(size=(1, nx))
        # bias for the neuron
        self.__b = 0
        # activated output of the neuron (prediction)
        self.__A = 0

    # getter functions
    @property
    def W(self):
        """Retrieves the weights vector"""
        return self.__W

    @property
    def b(self):
        """Retrieves the bias"""
        return self.__b

    @property
    def A(self):
        """Retrieves the activated output"""
        return self.__A
