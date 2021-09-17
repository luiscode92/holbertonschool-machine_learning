#!/usr/bin/env python3
"""
Builds the ResNet-50 architecture as described in
Deep Residual Learning for Image Recognition (2015)
"""


import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(filters=64, kernel_size=7, padding='same',
                            kernel_initializer='he_normal',
                            strides=2)(X)

    BN1 = K.layers.BatchNormalization(axis=3)(conv1)
    ReLU1 = K.layers.Activation('relu')(BN1)

    pool1 = K.layers.MaxPool2D(pool_size=3, strides=2,
                               padding='same')(ReLU1)

    PB1 = projection_block(pool1, [64, 64, 256], s=1)

    IB1 = identity_block(PB1, [64, 64, 256])

    IB2 = identity_block(IB1, [64, 64, 256])

    PB2 = projection_block(IB2, [128, 128, 512])

    IB3 = identity_block(PB2, [128, 128, 512])

    IB4 = identity_block(IB3, [128, 128, 512])

    IB5 = identity_block(IB4, [128, 128, 512])

    PB3 = projection_block(IB5, [256, 256, 1024])

    IB6 = identity_block(PB3, [256, 256, 1024])

    IB7 = identity_block(IB6, [256, 256, 1024])

    IB8 = identity_block(IB7, [256, 256, 1024])

    IB9 = identity_block(IB8, [256, 256, 1024])

    IB10 = identity_block(IB9, [256, 256, 1024])

    PB4 = projection_block(IB10, [512, 512, 2048])

    IB11 = identity_block(PB4, [512, 512, 2048])

    IB12 = identity_block(IB11, [512, 512, 2048])

    pool2 = K.layers.AveragePooling2D(pool_size=7, strides=1)(IB12)

    linear = K.layers.Dense(units=1000,
                            activation='softmax',
                            kernel_initializer='he_normal')(pool2)

    keras_model = K.models.Model(inputs=X, outputs=linear)

    return keras_model
