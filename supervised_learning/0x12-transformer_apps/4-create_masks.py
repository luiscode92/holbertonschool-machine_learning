#!/usr/bin/env python3
"""
Creates all masks for training/validation.
https://www.tensorflow.org/text/tutorials/transformer
"""
import tensorflow.compat.v2 as tf


def create_padding_mask(seq):
    """
    It ensures that the model does not treat padding as the input.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    # (batch_size, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Mask used to mask the future tokens in a sequence.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    # (seq_len, seq_len)
    return mask


def create_masks(inputs, target):
    """
    Creates all masks for training/validation.
    Returns: encoder_mask, combined_mask, decoder_mask
    """
    # Encoder padding mask
    encoder_mask = create_padding_mask(inputs)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    decoder_mask = create_padding_mask(inputs)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    dec_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
