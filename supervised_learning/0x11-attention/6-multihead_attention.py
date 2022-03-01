#!/usr/bin/env python3
"""
Class that inherits from tensorflow.keras.layers.Layer
to perform multi head attention.
https://www.tensorflow.org/tutorials/text/transformer
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Performs multi head attention
    """
    def __init__(self, dm, h):
        """
        Class constructor
        """
        super().__init__()
        # Number of heads
        self.h = h
        # Dimensionality of the model
        self.dm = dm
        # Depth of each attention head
        self.depth = dm // h
        # Dense layer used to generate the query matrix
        self.Wq = tf.keras.layers.Dense(units=dm)
        # Dense layer used to generate the key matrix
        self.Wk = tf.keras.layers.Dense(units=dm)
        # Dense layer used to generate the value matrix
        self.Wv = tf.keras.layers.Dense(units=dm)
        # Dense layer used to generate the attention output
        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, x, batch):
        """
        Split the last dimension into (h, depth).
        Transpose the result such that the shape is
        (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Returns: output, weights
        """
        batch = tf.shape(Q)[0]
        # Helping Kelsie
        # batch = Q.get_shape().as_list()[0]

        # (batch, seq_len_q, dk)
        Q = self.Wq(Q)
        # (batch, seq_len_v, dk)
        K = self.Wk(K)
        # (batch, seq_len_v, dv)
        V = self.Wv(V)

        # (batch, h, seq_len_q, depth)
        Q = self.split_heads(Q, batch)
        # (batch, h, seq_len_k, depth)
        K = self.split_heads(K, batch)
        # (batch, h, seq_len_v, depth)
        V = self.split_heads(V, batch)

        # scaled_attention.shape == (batch, h, seq_len_q, depth)
        # weights.shape == (batch, h, seq_len_q, seq_len_k)
        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        # (batch, seq_len_q, h, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch, seq_len_q, dm)
        concat_attention = \
            tf.reshape(scaled_attention, (batch, -1, self.dm))

        # (batch, seq_len_q, dm)
        output = self.linear(concat_attention)

        return output, weights
