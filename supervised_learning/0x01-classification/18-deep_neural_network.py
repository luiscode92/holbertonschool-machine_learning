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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        - nx is the number of input features to the neuron
        - m is the number of examples
        Updates the private attribute __cache
        - The activated outputs of each layer should be saved
          in the __cache dictionary using the key A{l} where {l}
          is the hidden layer the activated output belongs to
        - X should be saved to the cache dictionary using the key A0
        All neurons should use a sigmoid activation function
        You are allowed to use one loop
        Returns the output of the neural network and the cache, respectively
        """
        self.__cache["A0"] = X
        for i in range(self.__L):
            Z = (np.matmul(self.__weights["W{}".format(i+1)],
                 self.__cache["A{}".format(i)]) +
                 self.__weights["b{}".format(i+1)])
            # sigmoid activation function
            self.__cache["A{}".format(i+1)] = (np.exp(Z)/(np.exp(Z)+1))
        return (self.__cache["A{}".format(i+1)], self.__cache)
