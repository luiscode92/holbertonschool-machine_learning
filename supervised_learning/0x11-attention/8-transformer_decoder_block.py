#!/usr/bin/env python3
"""
Class that inherits from tensorflow.keras.layers.Layer
to create a decoder block for a transformer
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Creates a decoder block for a transformer
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor
        """
        super().__init__()
        # the first MultiHeadAttention layer
        self.mha1 = MultiHeadAttention(dm, h)
        # the second MultiHeadAttention layer
        self.mha2 = MultiHeadAttention(dm, h)
        # the hidden dense layer with hidden units and relu activation
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        # the output dense layer with dm units
        self.dense_output = tf.keras.layers.Dense(units=dm)
        # the first layer norm layer, with epsilon=1e-6
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # the second layer norm layer, with epsilon=1e-6
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # the third layer norm layer, with epsilon=1e-6
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # the first dropout layer
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        # the second dropout layer
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)
        # the third dropout layer
        self.dropout3 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Returns: a tensor of shape (batch, target_seq_len, dm)
        containing the block’s output
        """
        # encoder_output.shape == (batch, input_seq_len, dm)

        # (batch, target_seq_len, dm)
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # (batch, target_seq_len, dm)
        attn2, _ = self.mha2(out1, encoder_output, encoder_output,
                             padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        # (batch, target_seq_len, dm)
        out2 = self.layernorm2(attn2 + out1)

        # (batch, target_seq_len, dm)
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch, target_seq_len, dm)
        out3 = self.layernorm3(ffn_output + out2)

        return out3
