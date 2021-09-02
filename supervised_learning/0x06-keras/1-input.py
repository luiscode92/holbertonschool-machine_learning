#!/usr/bin/env python3
"""
Builds a neural network with the Keras library
"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    - nx is the number of input features to the network
    - layers is a list containing the number of nodes
      in each layer of the network
    - activations is a list containing the activation functions used
      for each layer of the network
    - lambtha is the L2 regularization parameter
    - keep_prob is the probability that a node will be kept for dropout
    - You are not allowed to use the Sequential class
    - Returns: the keras model
    """
    # Instantiate a Keras tensor
    inputs = K.Input(shape=(nx,))

    # regularization scheme
    l2_reg = K.regularizers.l2(lambtha)

    # first layer
    x = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=l2_reg)(inputs)

    for i in range(1, len(layers)):
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=l2_reg)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
