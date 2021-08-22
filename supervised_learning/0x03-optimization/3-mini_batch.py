#!/usr/bin/env python3
"""
Trains a loaded neural network model using mini-batch gradient descent
"""


import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        # placeholder for the input data
        x = tf.get_collection("x")[0]

        # placeholder for the labels
        y = tf.get_collection("y")[0]

        # op to calculate the accuracy of the model
        accuracy = tf.get_collection("accuracy")[0]

        # op to calculate the cost of the model
        loss = tf.get_collection("loss")[0]

        # op to perform one pass of gradient descent on the model
        train_op = tf.get_collection("train_op")[0]

        # find number of steps (iterations)
        steps = X_train.shape[0] // batch_size
        if steps % batch_size != 0:
            steps = steps + 1
            extra = True
        else:
            extra = False

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
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:

                # shuffle data, both training set and labels
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                # step within epoch
                for step_number in range(steps):
                    # data selection mini batch from training set and labels
                    start = step_number * batch_size

                    if step_number == steps - 1 and extra:
                        end = X_train.shape[0]
                    else:
                        end = step_number * batch_size + batch_size

                    X = X_shuffled[start:end]
                    Y = Y_shuffled[start:end]

                    # execute training for step
                    sess.run(train_op, feed_dict={x: X, y: Y})

                    if step_number != 0 and (step_number + 1) % 100 == 0:
                        print("\tStep {}:".format(step_number + 1))

                        # calculate cost and accuracy for step
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X, y: Y})

                        # where {step_cost} is the cost of the model
                        # on the current mini-batch
                        print("\t\tCost: {}".format(step_cost))

                        # where {step_accuracy} is the accuracy of the model
                        # on the current mini-batch
                        print("\t\tAccuracy: {}".format(step_accuracy))

        return saver.save(sess, save_path)
