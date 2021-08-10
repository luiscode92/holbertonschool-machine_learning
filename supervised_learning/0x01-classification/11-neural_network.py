#!/usr/bin/env python3
"""
Defines a neural network with one hidden layer
performing binary classification
"""


import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer
    performing binary classification
    """
    def __init__(self, nx, nodes):
        """
        Class constructor
        nx is the number of input features
        nodes is the number of nodes found in the hidden layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        # weights vector for the hidden layer
        # default mean is 0
        # default stddev is 1
        self.__W1 = np.random.normal(size=(nodes, nx))
        # The bias for the hidden layer. Upon instantiation,
        # it should be initialized with 0’s.
        self.__b1 = np.zeros((nodes, 1))
        # The activated output for the hidden layer. Upon instantiation,
        # it should be initialized to 0
        self.__A1 = 0
        # weights vector for the output neuron
        # default mean is 0
        # default stddev is 1
        self.__W2 = np.random.normal(size=(1, nodes))
        # bias for the output neuron
        self.__b2 = 0
        # activated output for the output neuron (prediction)
        self.__A2 = 0

    # getter functions
    @property
    def W1(self):
        """Retrieves W1"""
        return self.__W1

    @property
    def b1(self):
        """Retrieves b1"""
        return self.__b1

    @property
    def A1(self):
        """Retrieves A1"""
        return self.__A1

    @property
    def W2(self):
        """Retrieves W2"""
        return self.__W2

    @property
    def b2(self):
        """Retrieves b2"""
        return self.__b2

    @property
    def A2(self):
        """Retrieves A2"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        - nx is the number of input features to the neuron
        - m is the number of examples
        Updates the private attributes __A1 and __A2
        The neurons should use a sigmoid activation function
        Returns the private attributes __A1 and __A2, respectively
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        # sigmoid activation function
        self.__A1 = np.exp(Z1)/(np.exp(Z1) + 1)
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        # sigmoid activation function
        self.__A2 = np.exp(Z2)/(np.exp(Z2) + 1)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        - Y is a numpy.ndarray with shape (1, m) that contains
          the correct labels for the input data
        - A is a numpy.ndarray with shape (1, m) containing
          the activated output of the neuron for each example
        For avoiding division by zero, use 1.0000001 - A instead of 1 - A
        Returns the cost
        """
        m = Y.shape[1]
        cost = (-1/m)*np.sum(np.multiply(Y, np.log(A)) +
                             np.multiply((1-Y), np.log(1.0000001 - A)))
        return cost
