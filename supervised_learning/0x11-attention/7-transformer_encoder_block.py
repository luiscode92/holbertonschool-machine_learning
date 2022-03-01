#!/usr/bin/env python3
"""
Class that inherits from tensorflow.keras.layers.Layer
to create an encoder block for a transformer.
https://www.tensorflow.org/tutorials/text/transformer
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Creates an encoder block for a transformer
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor
        """
        super().__init__()
        # MultiHeadAttention layer
        self.mha = MultiHeadAttention(dm, h)
        # the hidden dense layer with hidden units and relu activation
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        # the output dense layer with dm units
        self.dense_output = tf.keras.layers.Dense(units=dm)
        # the first layer norm layer, with epsilon=1e-6
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # the second layer norm layer, with epsilon=1e-6
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # the first dropout layer
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        # the second dropout layer
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask=None):
        """
        Returns: a tensor of shape (batch, input_seq_len, dm)
        containing the block’s output
        """
        # (batch, input_seq_len, dm)
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch, input_seq_len, dm)
        out1 = self.layernorm1(x + attn_output)

        # (batch, input_seq_len, dm)
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch, input_seq_len, dm)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
