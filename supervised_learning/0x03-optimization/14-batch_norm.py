#!/usr/bin/env python3
"""
Creates a batch normalization layer for a neural network in tensorflow
"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    - prev is the activated output of the previous layer
    - n is the number of nodes in the layer to be created
    - activation is the activation function that should be used
      on the output of the layer
    - you should use the tf.layers.Dense layer as the base layer with
      kernal initializer
      tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    - your layer should incorporate two trainable parameters, gamma and beta,
      initialized as vectors of 1 and 0 respectively
    - you should use an epsilon of 1e-8
    - Returns: a tensor of the activated output for the layer
    """
    # He et al. initialization for the layer weights
    kernal_init = tf.contrib.layers.\
        variance_scaling_initializer(mode="FAN_AVG")

    base_layer = tf.layers.Dense(units=n, kernel_initializer=kernal_init,
                                 name='base_layer')
    X = base_layer(prev)
    # Calculate the mean and variance of X
    mu, var = tf.nn.moments(X, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=(1, n)), trainable=True,
                        name='gamma')
    beta = tf.Variable(tf.constant(0.0, shape=(1, n)), trainable=True,
                       name='beta')
    # returns the normalized, scaled, offset tensor
    Z = tf.nn.batch_normalization(x=X, mean=mu, variance=var,
                                  offset=beta, scale=gamma,
                                  variance_epsilon=1e-8,
                                  name='Z')
    # activation function
    A = activation(Z)
    return A
