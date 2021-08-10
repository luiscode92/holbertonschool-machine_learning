#!/usr/bin/env python3
"""
Performs multiclass classification
Allows different activation functions
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Performs multiclass classification
    Allows different activation functions
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Class constructor
        - nx is the number of input features
        - layers is a list representing the number of nodes
          in each layer of the network
        - activation: type of activation function used
            * sig a sigmoid activation
            * tanh a tanh activation
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.nx = nx
        self.layers = layers
        self.__L = len(layers)
        self.__activation = activation
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            W_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)

            self.weights[b_key] = np.zeros((layers[i], 1))

            if i == 0:
                f = np.sqrt(2 / nx)
                self.__weights['W1'] = np.random.randn(layers[i], nx) * f
            else:
                f = np.sqrt(2 / layers[i - 1])
                h = np.random.randn(layers[i], layers[i - 1]) * f
                self.__weights[W_key] = h

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

    @property
    def activation(self):
        """Retrieves activation function"""
        return self.__activation

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
        self.__cache['A0'] = X

        for i in range(self.__L):
            W_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)
            A_key_prev = "A{}".format(i)
            A_key_forw = "A{}".format(i + 1)

            Z = np.matmul(self.__weights[W_key], self.__cache[A_key_prev]) \
                + self.__weights[b_key]
            # if it is not the last layer
            if i != self.__L - 1:
                # sigmoid activation function
                if self.__activation == 'sig':
                    self.__cache[A_key_forw] = 1 / (1 + np.exp(-Z))
                # tanh activation function
                else:
                    self.__cache[A_key_forw] = np.tanh(Z)
            # if it is the last layer
            else:
                # softmax activation function for multi-class classification
                # t is a temporary variable
                t = np.exp(Z)
                # normalize
                self.__cache[A_key_forw] = (t / np.sum(t,
                                                       axis=0, keepdims=True))
        return self.__cache[A_key_forw], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        - Y is now a one-hot numpy.ndarray of shape (classes, m)
        - A is a numpy.ndarray with shape (classes, m) containing
          the activated output of each neuron in the output layer
          for each example
        Returns the cost
        """
        # cost function for softmax activation function
        return -(1/Y.shape[1]) * np.sum(Y * np.log(A))

    def evaluate(self, X, Y):
        """
        - Evaluates the neural network’s predictions
        - Returns the neuron’s prediction and the cost of the network,
          respectively
          * The prediction should be a numpy.ndarray with shape (classes, m)
            containing the predicted labels for each example
          * The label values should be 1 for the maximum value and zero for
            the other ones in each example
        """
        self.forward_prop(X)[0]
        key = "A" + str(self.__L)
        tmp = np.amax(self.__cache[key], axis=0)
        return (np.where(self.__cache[key] == tmp, 1, 0),
                self.cost(Y, self.__cache[key]))

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        - cache is a dictionary containing all the intermediary values
          of the network
        - alpha is the learning rate
        - Updates the private attribute __weights
        - You are allowed to use one loop
        """
        weights = self.__weights.copy()
        m = Y.shape[1]

        for i in reversed(range(self.__L)):
            # if it is the last layer
            if i == self.__L - 1:
                dZ = cache['A{}'.format(i + 1)] - Y

            # hidden layers
            else:
                dZa = np.matmul(weights['W{}'.format(i + 2)].T, dZ)
                if self.__activation == 'sig':
                    # derivative of sigmoid function
                    dZb = (cache['A{}'.format(i + 1)]
                           * (1 - cache['A{}'.format(i + 1)]))
                else:
                    # derivative of tanh function
                    dZb = 1 - cache['A{}'.format(i + 1)] ** 2
                dZ = dZa * dZb

            dW = (np.matmul(dZ, cache['A{}'.format(i)].T)) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.__weights['W{}'.format(i + 1)] = \
                weights['W{}'.format(i + 1)] \
                - (alpha * dW)
            self.__weights['b{}'.format(i + 1)] = \
                weights['b{}'.format(i + 1)] \
                - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network by updating __weights and __cache
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
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        cost_list = []
        steps_list = []
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, self.__cache['A{}'.format(self.L)])
                cost_list.append(cost)
                steps_list.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
        if graph is True:
            plt.plot(steps_list, cost_list, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format
        filename is the file to which the object should be saved
        If filename does not have the extension .pkl, add it
        """
        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object
        filename is the file from which the object should be loaded
        Returns: the loaded object, or None if filename doesn’t exist
        """
        try:
            # read in binary
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except FileNotFoundError:
            return None
