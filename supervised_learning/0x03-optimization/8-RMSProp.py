#!/usr/bin/env python3
"""
Creates the training operation for a neural network in tensorflow using
the RMSProp optimization algorithm
"""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using
    the RMSProp optimization algorithm
    - loss is the loss of the network
    - alpha is the learning rate
    - beta2 is the RMSProp weight
    - epsilon is a small number to avoid division by zero
    - Returns: the RMSProp optimization operation
    """
    train_op = tf.train.RMSPropOptimizer(alpha,
                                         decay=beta2,
                                         epsilon=epsilon).minimize(loss)
    return train_op
