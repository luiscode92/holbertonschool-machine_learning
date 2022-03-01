#!/usr/bin/env python3
"""
Represents an LSTM unit
"""
import numpy as np


def sigmoid(x):
    """
    Sigmoid function
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    Softmax function
    """
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class LSTMCell:
    """
    Represents an LSTM unit
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        """
        # weight for the forget gate
        self.Wf = np.random.normal(size=(i + h, h))
        # weight for the update gate
        self.Wu = np.random.normal(size=(i + h, h))
        # weight for the intermediate cell state
        self.Wc = np.random.normal(size=(i + h, h))
        # weight for the output gate
        self.Wo = np.random.normal(size=(i + h, h))
        # weight for the outputs
        self.Wy = np.random.normal(size=(h, o))

        # bias for the forget gate
        self.bf = np.zeros((1, h))
        # bias for the update gate
        self.bu = np.zeros((1, h))
        # bias for the intermediate cell state
        self.bc = np.zeros((1, h))
        # bias for the output gate
        self.bo = np.zeros((1, h))
        # bias for the outputs
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.
        Returns: h_next, c_next, y
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f_t = sigmoid(concat @ self.Wf + self.bf)

        # Update gate
        u_t = sigmoid(concat @ self.Wu + self.bu)

        # Output gate
        o_t = sigmoid(concat @ self.Wo + self.bo)

        # Intermediate cell state
        c_t = np.tanh(concat @ self.Wc + self.bc)

        # next cell state
        c_next = f_t * c_prev + u_t * c_t

        # next hidden state
        h_next = o_t * np.tanh(c_next)

        # output of the cell
        y = softmax(h_next @ self.Wy + self.by)

        return h_next, c_next, y
