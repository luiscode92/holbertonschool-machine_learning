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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        - nx is the number of input features to the neuron
        - m is the number of examples
        """
        Z = np.matmul(self.__W, X) + self.__b
        # sigmoid activation function
        self.__A = np.exp(Z)/(np.exp(Z) + 1)
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        - Y is a numpy.ndarray with shape (1, m) that contains
          the correct labels for the input data
        - A is a numpy.ndarray with shape (1, m) containing
          the activated output of the neuron (prediction) for each example
        """
        m = Y.shape[1]
        cost = (-1/m)*np.sum(np.multiply(Y, np.log(A)) +
                             np.multiply((1-Y), np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        - X is a numpy.ndarray with shape (nx, m) that contains
          the input data
          * nx is the number of input features to the neuron
          * m is the number of examples
        - Y is a numpy.ndarray with shape (1, m) that contains
          the correct labels for the input data
        Returns the neuron’s prediction and the cost of
        the network, respectively
        """
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        # cost with A for avoiding division by zero
        cost = self.cost(Y, A)
        evaluate = prediction, cost
        return evaluate

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        - X is a numpy.ndarray with shape (nx, m) that contains
          the input data
          * nx is the number of input features to the neuron
          * m is the number of examples
        - Y is a numpy.ndarray with shape (1, m) that contains
          the correct labels for the input data
        - A is a numpy.ndarray with shape (1, m) containing
          the activated output of the neuron for each example
        - alpha is the learning rate
        - Updates the private attributes __W and __b
        """
        dZ = A - Y
        m = Y.shape[1]
        dW = (1/m)*np.matmul(dZ, X.T)
        db = (1/m)*np.sum(dZ)
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron
        - Updates the private attributes __W, __b, and __A
        - You are allowed to use one loop
        - Returns the evaluation of the training data after
          iterations of training have occurred
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        # The following unused i is not a problem
        for i in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
