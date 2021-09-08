#!/usr/bin/env python3
"""
Builds a modified version of the LeNet-5 architecture using keras
"""


import tensorflow.keras as K


def lenet5(X):
    """
    Returns: a K.Model compiled to use Adam optimization
    (with default hyperparameters) and accuracy metrics
    """
    # All layers requiring initialization should initialize their kernels
    # with the he_normal initialization method
    initializer = K.initializers.he_normal(seed=None)

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = K.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                            activation='relu',
                            kernel_initializer=initializer)(X)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = K.layers.MaxPool2D(pool_size=2, strides=(2, 2))(conv1)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = K.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                            activation='relu',
                            kernel_initializer=initializer)(pool1)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = K.layers.MaxPool2D(pool_size=2, strides=(2, 2))(conv2)

    # Flatten
    pool2 = K.layers.Flatten()(pool2)

    # Fully connected layer (dense) with 120 nodes
    fc3 = K.layers.Dense(units=120, activation='relu',
                         kernel_initializer=initializer)(pool2)

    # Fully connected layer (dense) with 84 nodes
    fc4 = K.layers.Dense(units=84, activation='relu',
                         kernel_initializer=initializer)(fc3)

    # Fully connected softmax output layer with 10 nodes
    output_layer = K.layers.Dense(units=10, activation='softmax',
                                  kernel_initializer=initializer)(fc4)

    model = K.Model(inputs=X, outputs=output_layer)

    adam_param = K.optimizers.Adam()

    model.compile(optimizer=adam_param, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
