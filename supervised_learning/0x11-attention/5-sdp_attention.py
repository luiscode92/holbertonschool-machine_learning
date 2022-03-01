#!/usr/bin/env python3
"""
Calculates the scaled dot product attention.
https://www.tensorflow.org/tutorials/text/transformer
#scaled_dot_product_attention
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Returns: output, weights
    """
    # (..., seq_len_q, seq_len_k)
    matmul_QK = tf.matmul(Q, K, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_QK / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    # (..., seq_len_q, seq_len_k)
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(weights, V)  # (..., seq_len_q, depth_v)

    return output, weights
