#!/usr/bin/env python3
"""
Trains a model using mini-batch gradient descent.
Also analyzes validation data
"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Returns: the History object generated after training the model
    """
    History = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose,
                          validation_data=validation_data, shuffle=shuffle)
    return History
