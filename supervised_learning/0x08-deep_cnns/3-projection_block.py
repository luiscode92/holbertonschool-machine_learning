#!/usr/bin/env python3
"""
Builds a projection block as described in
Deep Residual Learning for Image Recognition (2015)
"""


import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Returns: the activated output of the projection block
    """
    F11, F3, F12 = filters
    conv1 = K.layers.Conv2D(filters=F11,
                            kernel_size=1,
                            padding='same',
                            kernel_initializer='he_normal',
                            strides=s)(A_prev)

    BN1 = K.layers.BatchNormalization(axis=3)(conv1)
    ReLU1 = K.layers.Activation('relu')(BN1)

    conv2 = K.layers.Conv2D(filters=F3,
                            kernel_size=3,
                            padding='same',
                            kernel_initializer='he_normal',
                            strides=1)(ReLU1)

    BN2 = K.layers.BatchNormalization(axis=3)(conv2)
    ReLU2 = K.layers.Activation('relu')(BN2)

    conv3 = K.layers.Conv2D(filters=F12,
                            kernel_size=1,
                            padding='same',
                            kernel_initializer='he_normal',
                            strides=1)(ReLU2)

    BN3 = K.layers.BatchNormalization(axis=3)(conv3)

    shortcut = K.layers.Conv2D(filters=F12,
                               kernel_size=1,
                               padding='same',
                               kernel_initializer='he_normal',
                               strides=s)(A_prev)

    BN_shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    addition = K.layers.Add()([BN3, BN_shortcut])

    output = K.layers.Activation('relu')(addition)

    return output
