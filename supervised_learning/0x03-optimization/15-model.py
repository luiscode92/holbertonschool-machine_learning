#!/usr/bin/env python3
"""Contains the model function"""

import tensorflow as tf
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    return X[shuffle], Y[shuffle]


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using
    the Adam optimization algorithm
    - loss is the loss of the network
    - alpha is the learning rate
    - beta1 is the weight used for the first moment
    - beta2 is the weight used for the second moment
    - epsilon is a small number to avoid division by zero
    - Returns: the Adam optimization operation
    """
    train_op = tf.train.AdamOptimizer(alpha,
                                      beta1,
                                      beta2,
                                      epsilon).minimize(loss)
    return train_op


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


def create_batch_norm_layer(prev, n, activation):
    """
    - prev is the activated output of the previous layer
    - n is the number of nodes in the layer to be created
    - activation is the activation function that should be used
      on the output of the layer
    - you should use the tf.layers.Dense layer as the base layer with
      kernal initializer
      tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    - your layer should incorporate two trainable parameters, gamma and beta,
      initialized as vectors of 1 and 0 respectively
    - you should use an epsilon of 1e-8
    - Returns: a tensor of the activated output for the layer
    """
    if activation is None:
        A = create_layer(prev, n, activation)
        return A
    # He et al. initialization for the layer weights
    kernal_init = tf.contrib.layers.\
        variance_scaling_initializer(mode="FAN_AVG")

    base_layer = tf.layers.Dense(units=n, kernel_initializer=kernal_init,
                                 name='base_layer')
    X = base_layer(prev)
    # Calculate the mean and variance of X
    mu, var = tf.nn.moments(X, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=(1, n)), trainable=True,
                        name='gamma')
    beta = tf.Variable(tf.constant(0.0, shape=(1, n)), trainable=True,
                       name='beta')
    # returns the normalized, scaled, offset tensor
    Z = tf.nn.batch_normalization(x=X, mean=mu, variance=var,
                                  offset=beta, scale=gamma,
                                  variance_epsilon=1e-8,
                                  name='Z')
    # activation function
    A = activation(Z)
    return A


def forward_prop(x, layer, activations):
    """
    Creates the forward propagation graph for the neural network
    - x is the placeholder for the input data
    - layer is a list containing the number of nodes in
      each layer of the network
    - activations is a list containing the activation functions
      for each layer of the network
    Returns: prediction of the network in tensor form
    """
    # first layer activation with x features as input
    y_pred = create_batch_norm_layer(x, layer[0], activations[0])

    # next layers activations with y_pred from the prev layer as input
    for i in range(1, len(layer)):
        y_pred = create_batch_norm_layer(y_pred, layer[i],
                                         activations[i])

    return y_pred


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


def calculate_loss(y, y_pred):
    """
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the loss of the prediction
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    - alpha is the original learning rate
    - decay_rate is the weight used to determine the rate
      at which alpha will decay
    - global_step is the number of passes of gradient descent
      that have elapsed
    - decay_step is the number of passes of gradient descent
      that should occur before alpha is decayed further
    - the learning rate decay should occur in a stepwise fashion
    - Returns: the learning rate decay operation
    """
    learning_rate = tf.train.inverse_time_decay(alpha,
                                                global_step,
                                                decay_step,
                                                decay_rate,
                                                staircase=True)
    return learning_rate


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow using
    Adam optimization,
    mini-batch gradient descent,
    learning rate decay,
    and batch normalization
    """
    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]

    steps = X_train.shape[0] / batch_size
    if (steps).is_integer() is True:
        steps = int(steps)
    else:
        steps = int(steps) + 1

    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]],
                       name='x')
    tf.add_to_collection('x', x)

    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]],
                       name='y')
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha,
                                decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    # initialize all variables
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs + 1):
            # execute cost and accuracy operations for training set
            train_cost, train_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train})

            # execute cost and accuracy operations for validation set
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))

            # train_cost is the cost of the model
            # on the entire training set
            print("\tTraining Cost: {}".format(train_cost))

            # train_accuracy is the accuracy of the model
            # on the entire training set
            print("\tTraining Accuracy: {}".format(train_accuracy))

            # valid_cost is the cost of the model
            # on the entire validation set
            print("\tValidation Cost: {}".format(valid_cost))

            # valid_accuracy is the accuracy of the model
            # on the entire validation set
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                # learning rate decay
                sess.run(global_step.assign(epoch))
                # update learning rate
                sess.run(alpha)

                # shuffle data, both training set and labels
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                # mini-batch within epoch
                for step_number in range(steps):

                    # data selection mini batch from training set and labels
                    start = step_number * batch_size

                    end = (step_number + 1) * batch_size
                    if end > X_train.shape[0]:
                        end = X_train.shape[0]

                    X = X_shuffled[start:end]
                    Y = Y_shuffled[start:end]

                    # execute training for step
                    sess.run(train_op, feed_dict={x: X, y: Y})

                    if step_number != 0 and (step_number + 1) % 100 == 0:
                        # step_number is the number of times gradient
                        # descent has been run in the current epoch
                        print("\tStep {}:".format(step_number + 1))

                        # calculate cost and accuracy for step
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X, y: Y})

                        # step_cost is the cost of the model
                        # on the current mini-batch
                        print("\t\tCost: {}".format(step_cost))

                        # step_accuracy is the accuracy of the model
                        # on the current mini-batch
                        print("\t\tAccuracy: {}".format(step_accuracy))

        return saver.save(sess, save_path)
