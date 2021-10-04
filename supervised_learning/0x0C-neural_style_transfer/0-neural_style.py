# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()
class NST:
    """
    Create a class NST that performs tasks for neural style transfer:
    Public class attributes:
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                        'block4_conv1', 'block5_conv1']
        content_layer = 'block5_conv2'
    Class constructor: def __init__(self, style_image, content_image,
                                    alpha=1e4, beta=1):
        style_image - the image used as a style reference, stored as a
                        numpy.ndarray
        content_image - the image used as a content reference, stored as a
                        numpy.ndarray
        alpha - the weight for content cost
        beta - the weight for style cost
        -if style_image is not a np.ndarray with the shape (h, w, 3), raise a
        TypeError with the message style_image must be a numpy.ndarray with
        shape (h, w, 3)
        -if content_image is not a np.ndarray with the shape (h, w, 3), raise
        a TypeError with the message content_image must be a numpy.ndarray with
        shape (h, w, 3)
        -if alpha is not a non-negative number, raise a TypeError with the
        message alpha must be a non-negative number
        -if beta is not a non-negative number, raise a TypeError with the
        message beta must be a non-negative number
        Sets Tensorflow to execute eagerly
        Sets the instance attributes:
            style_image - the preprocessed style image
            content_image - the preprocessed content image
            alpha - the weight for content cost
            beta - the weight for style cost
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv1'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Init method for the Class
