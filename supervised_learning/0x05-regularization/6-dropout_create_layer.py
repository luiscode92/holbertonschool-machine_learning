#!/usr/bin/env python3
"""
Creates a layer of a neural network using dropout
"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    - prev is a tensor containing the output of the previous layer
    - n is the number of nodes the new layer should contain
    - activation is the activation function that should be used on the layer
    - keep_prob is the probability that a node will be kept
    - Returns: the output of the new layer
    """
    # He et al. initialization for the layer weights
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    # Function with signature l2(weights) that applies L2 regularization
    dropout = tf.layers.Dropout(1 - keep_prob)
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=w,
                            kernel_regularizer=dropout,
                            name='layer')
    return layer(prev)
