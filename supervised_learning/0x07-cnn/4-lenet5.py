#!/usr/bin/env python3
"""
Builds a modified version of the LeNet-5 architecture using tensorflow
"""


import tensorflow as tf


def lenet5(x, y):
    """
    Returns:
    - a tensor for the softmax activated output
    - a training operation that utilizes Adam optimization
      (with default hyperparameters)
    - a tensor for the loss of the netowrk
    - a tensor for the accuracy of the network
    """
    # All layers requiring initialization should initialize their kernels
    # with the he_normal initialization method
    initializer = tf.contrib.layers.variance_scaling_initializer()

    # All hidden layers requiring activation should use the relu
    # activation function
    relu = tf.nn.relu

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5),
                             padding='same', activation=relu,
                             kernel_initializer=initializer)(x)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv1)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5),
                             padding='valid', activation=relu,
                             kernel_initializer=initializer)(pool1)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)

    # Flatten
    pool2 = tf.layers.Flatten()(pool2)

    # Fully connected (dense) layer with 120 nodes
    fc3 = tf.layers.Dense(units=120, activation=relu,
                          kernel_initializer=initializer)(pool2)

    # Fully connected layer with 84 nodes
    fc4 = tf.layers.Dense(units=84, activation=relu,
                          kernel_initializer=initializer)(fc3)

    # Fully connected softmax output layer with 10 nodes
    output_layer =\
        tf.layers.Dense(units=10, kernel_initializer=initializer)(fc4)

    # tensor for the softmax activated output
    y_pred = tf.nn.softmax(output_layer)

    # tensor for the loss of the network
    loss = tf.losses.softmax_cross_entropy(y, logits=output_layer)

    # training operation that utilizes Adam optimization
    # (with default hyperparameters)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # tensor for the accuracy of the network
    # comparison between prediction and true label (Boolean)
    equal = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(equal, tf.float32))

    return y_pred, train_op, loss, acc
