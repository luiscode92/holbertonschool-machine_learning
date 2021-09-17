#!/usr/bin/env python3
"""
Builds the inception network as described in
Going Deeper with Convolutions (2014)
"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))

    # Conv 7x7+2(S)
    conv1 = K.layers.Conv2D(filters=64, kernel_size=7, activation='relu',
                            padding="same", kernel_initializer='he_normal',
                            strides=2)(X)

    # MaxPool 3x3+2(S)
    pool1 = K.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(conv1)

    # Conv 1x1+1(V)
    conv2 = K.layers.Conv2D(filters=64, kernel_size=1, activation='relu',
                            padding="same", kernel_initializer='he_normal',
                            strides=1)(pool1)

    # Conv 3x3+1(S)
    conv3 = K.layers.Conv2D(filters=192, kernel_size=3, activation='relu',
                            padding="same", kernel_initializer='he_normal',
                            strides=1)(conv2)

    # MaxPool 3x3+2(S)
    pool2 = K.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(conv3)

    inception_3a = inception_block(A_prev=pool2,
                                   filters=(64, 96, 128, 16, 32, 32))

    inception_3b = inception_block(A_prev=inception_3a,
                                   filters=(128, 128, 192, 32, 96, 64))

    pool3 = K.layers.MaxPool2D(pool_size=3,
                               strides=2, padding='same')(inception_3b)

    inception_4a = inception_block(A_prev=pool3,
                                   filters=(192, 96, 208, 16, 48, 64))

    inception_4b = inception_block(A_prev=inception_4a,
                                   filters=(160, 112, 224, 24, 64, 64))

    inception_4c = inception_block(A_prev=inception_4b,
                                   filters=(128, 128, 256, 24, 64, 64))

    inception_4d = inception_block(A_prev=inception_4c,
                                   filters=(112, 144, 288, 32, 64, 64))

    inception_4e = inception_block(A_prev=inception_4d,
                                   filters=(256, 160, 320, 32, 128, 128))

    pool4 = K.layers.MaxPool2D(pool_size=3,
                               strides=2, padding='same')(inception_4e)

    inception_5a = inception_block(A_prev=pool4,
                                   filters=(256, 160, 320, 32, 128, 128))

    inception_5b = inception_block(A_prev=inception_5a,
                                   filters=(384, 192, 384, 48, 128, 128))

    avg_pool = K.layers.AveragePooling2D(pool_size=7,
                                         padding='same')(inception_5b)
    dropout = K.layers.Dropout(rate=0.4)(avg_pool)

    linear = K.layers.Dense(units=1000,
                            activation='softmax',
                            kernel_initializer='he_normal')(dropout)

    keras_model = K.models.Model(inputs=X, outputs=linear)

    return keras_model
