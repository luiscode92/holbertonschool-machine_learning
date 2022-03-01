#!/usr/bin/env python3
"""
Class that inherits from tensorflow.keras.layers.Layer
to calculate the attention for machine translation.
https://towardsdatascience.com/implementing-neural-machine-translation-
with-attention-using-tensorflow-fc9c6f26155f
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Inherits from tensorflow.keras.layers.Layer
    to calculate the attention for machine translation
    """
    def __init__(self, units):
        """
        Class constructor
        """
        super().__init__()
        # Dense layer to be applied to the previous decoder hidden state
        self.W = tf.keras.layers.Dense(units=units)
        # Dense layer to be applied to the encoder hidden states
        self.U = tf.keras.layers.Dense(units=units)
        # Dense layer to be applied to
        # the tanh of the sum of the outputs of W and U
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
        * s_prev (batch, units) contains the previous decoder hidden state.
        * hidden_states (batch, input_seq_len, units)
          contains the outputs of the encoder
        Returns: context, weights
        """
        # Expansion for broadcasting
        # s_prev (batch, units)
        # exp_s_prev (batch, 1, units)
        exp_s_prev = tf.expand_dims(s_prev, axis=1)

        # Calculate the attention score
        score = self.V(tf.nn.tanh(self.W(exp_s_prev) + self.U(hidden_states)))
        # Attention Weights
        weights = tf.nn.softmax(score, axis=1)

        # Context as the weighted sum of the hidden_states
        # Context vector for the decoder
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
