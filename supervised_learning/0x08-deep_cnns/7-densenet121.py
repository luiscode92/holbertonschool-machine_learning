#!/usr/bin/env python3
"""
Builds the DenseNet-121 architecture as described in
Densely Connected Convolutional Networks
"""


import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))

    BN1 = K.layers.BatchNormalization(axis=3)(X)

    ReLU1 = K.layers.Activation('relu')(BN1)

    conv1 = K.layers.Conv2D(filters=64, kernel_size=7, padding='same',
                            kernel_initializer='he_normal',
                            strides=2)(ReLU1)

    pool1 = K.layers.MaxPool2D(pool_size=3, strides=2,
                               padding='same')(conv1)

    d1, f_d1 = dense_block(pool1, 64, growth_rate, 6)

    t1, f_t1 = transition_layer(d1, f_d1, compression)

    d2, f_d2 = dense_block(t1, f_t1, growth_rate, 12)

    t2, f_t2 = transition_layer(d2, f_d2, compression)

    d3, f_d3 = dense_block(t2, f_t2, growth_rate, 24)

    t3, f_t3 = transition_layer(d3, f_d3, compression)

    d4, _ = dense_block(t3, f_t3, growth_rate, 16)

    pool2 = K.layers.AveragePooling2D(pool_size=7,
                                      padding='same')(d4)

    linear = K.layers.Dense(units=1000,
                            activation='softmax',
                            kernel_initializer='he_normal')(pool2)

    keras_model = K.models.Model(inputs=X, outputs=linear)

    return keras_model
