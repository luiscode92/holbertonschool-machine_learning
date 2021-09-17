#!/usr/bin/env python3
"""
Builds a transition layer as described in
Densely Connected Convolutional Networks
"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Returns: The output of the transition layer and
    the number of filters within the output, respectively
    """
    nb_filters = int(nb_filters * compression)

    BN1 = K.layers.BatchNormalization()(X)

    ReLU1 = K.layers.Activation('relu')(BN1)

    conv1 = K.layers.Conv2D(filters=nb_filters, kernel_size=1,
                            padding='same',
                            kernel_initializer='he_normal',
                            strides=1)(ReLU1)

    pool1 = K.layers.AveragePooling2D(pool_size=2,
                                      strides=2)(conv1)

    return pool1, nb_filters
