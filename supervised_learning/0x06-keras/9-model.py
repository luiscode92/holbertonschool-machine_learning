#!/usr/bin/env python3
"""
Saves an entire model.
Loads an entire model.
"""


import tensorflow.keras as K


def save_model(network, filename):
    """
    - network is the model to save
    - filename is the path of the file that the model should be saved to
    - Returns: None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    - filename is the path of the file that the model should be loaded from
    - Returns: the loaded model
    """
    network = K.models.load_model(filename)
    return network
