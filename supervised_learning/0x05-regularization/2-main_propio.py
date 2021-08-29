#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
regularization_loss = __import__('2-l2_reg_cost').l2_reg_cost

def create_placeholders(nx, classes):
    """
    Returns two placeholders, x and y, for the neural network
    nx: the number of feature columns in our data
    classes: the number of classes in our classifier
    x is the placeholder for the input data to the neural network
    y is the placeholder for the one-hot labels for the input data
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y

def create_layer(prev, n, activation):
    """
    - prev is the tensor output of the previous layer
    - n is the number of nodes in the layer to create
    - activation is the activation function that the layer should use
    - use tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
      to implement He et. al initialization for the layer weights
    - each layer should be given the name layer
    - Returns: the tensor output of the layer
    """
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=w,
                            name='layer')
    return layer(prev)

def forward_prop(x, layer_sizes=[], activations=[]):
    """
    - x is the placeholder for the input data
    - layer_sizes is a list containing the number of nodes
      in each layer of the network
    - activations is a list containing the activation functions
      for each layer of the network
    - Returns: the prediction of the network in tensor form
    """
    A = x
    for i in range(len(layer_sizes)):
        A = create_layer(A, layer_sizes[i], activations[i])
    return A

def calculate_loss(y, y_pred):
    """
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the loss of the prediction
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss


x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
loss = calculate_loss(y, y_pred)
regu_loss = regularization_loss(loss)
print(loss)
print(regu_loss)
