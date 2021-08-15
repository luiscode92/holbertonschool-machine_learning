#!/usr/bin/env python3
"""
Evaluates the output of a neural network
"""


import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network
    - X is a numpy.ndarray containing the input data to evaluate
    - Y is a numpy.ndarray containing the one-hot labels for X
    - save_path is the location to load the model from
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        # Restore variables from disk
        saver.restore(sess, save_path)
        # get x and y placeholders
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        # get y_pred
        y_pred = tf.get_collection('y_pred')[0]
        # get accuracy
        accuracy = tf.get_collection('accuracy')[0]
        # get loss
        loss = tf.get_collection('loss')[0]

        # Calculate prediction, accuracy, and loss, with X and Y as input
        # forward_prop operation
        test_prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        # calculate_accuracy operation
        test_accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        # calculate_loss operation
        test_cost = sess.run(loss, feed_dict={x: X, y: Y})
    return test_prediction, test_accuracy, test_cost
