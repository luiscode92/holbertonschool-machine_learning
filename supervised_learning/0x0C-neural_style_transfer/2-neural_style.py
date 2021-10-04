#!/usr/bin/env python3
"""
Create a class NST that performs tasks for neural style transfer,
loads the model for neural style transfer
and calculates gram matrices
"""


import numpy as np
import tensorflow as tf


class NST:
    """
    Performs tasks for neural style transfer
    """
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor
        """
        return None

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels
        """
        return None

    def load_model(self):
        """
        Loads the model for neural style transfer
        """
        return None

    @staticmethod
    def gram_matrix(input_layer):
        """
        Returns: a tf.Tensor of shape (1, c, c)
        containing the gram matrix of input_layer
        """
        return None
