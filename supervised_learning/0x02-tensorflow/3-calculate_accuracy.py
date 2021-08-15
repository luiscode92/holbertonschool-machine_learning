#!/usr/bin/env python3
"""
Calculates the accuracy of a prediction
"""


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction
    """
    # decode y and y_pred
    y_decoded = tf.argmax(y, axis=1)
    y_pred_decoded = tf.argmax(y_pred, axis=1)
    # Boolean truth value of (y==y_pred) element-wise
    equal = tf.equal(y_decoded, y_pred_decoded)
    # cast to float for decimal accuracy
    equal = tf.cast(equal, tf.float32)
    # Mean of elements across dimensions of a tensor
    accuracy = tf.reduce_mean(equal)
    return accuracy
