#!/usr/bin/env python3
"""
Defines a deep neural network performing binary classification
"""


import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""
    def __init__(self, nx, layers):
        """
        Class constructor
        - nx is the number of input features
        - layers is a list representing the number of nodes
          in each layer of the network
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        # nx is size of layer 0
        self.nx = nx
        # number of layers in the neural network
        self.__L = len(layers)
        # dictionary to hold all intermediary values of the network
        self.__cache = {}
        # dictionary to hold all weights and biases of the network
        # * weights initialized using the He et al. method
        # * biases initialized to 0's
        self.__weights = {}
        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.__weights["W{}".format(i+1)] = (np.random.randn(layers[i],
                                                     self.nx) *
                                                     np.sqrt(2/self.nx))
            else:
                self.__weights["W{}".format(i+1)] = (np.random.randn(layers[i],
                                                     layers[i-1]) *
                                                     np.sqrt(2/layers[i-1]))
            self.__weights["b{}".format(i+1)] = np.zeros((layers[i], 1))

    # getter functions
    @property
    def L(self):
        """Retrieves L"""
        return self.__L

    @property
    def cache(self):
        """Retrieves cache"""
        return self.__cache

    @property
    def weights(self):
        """Retrieves weights"""
        return self.__weights
