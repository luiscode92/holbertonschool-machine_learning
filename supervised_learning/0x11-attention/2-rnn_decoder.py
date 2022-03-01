#!/usr/bin/env python3
"""
Class that inherits from tensorflow.keras.layers.Layer
to decode for machine translation
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Inherits from tensorflow.keras.layers.Layer
    to decode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        """
        super().__init__()
        # Embedding layer that converts
        # words from the vocabulary into an embedding vector.
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        # GRU layer
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # Dense layer
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """
        * x (batch, 1) contains the previous word in the target sequence
          as an index of the target vocabulary.
        * s_prev (batch, units) contains the previous decoder hidden state.
        * hidden_states (batch, input_seq_len, units)
          contains the outputs of the encoder.
        Returns: y(batch, vocab), s(batch, units)
        """
        _, units = s_prev.shape
        attention = SelfAttention(units)
        context, _ = attention(s_prev, hidden_states)
        embeddings = self.embedding(x)

        # Expansion for broadcasting
        # context (batch, units)
        # exp_context (batch, 1, units)
        exp_context = tf.expand_dims(context, axis=1)

        concat_input = tf.concat([exp_context, embeddings], axis=-1)

        # outputs and the last hidden state
        # s contains the new decoder hidden state
        outputs, s = self.gru(concat_input)
        # outputs (batch, 1, units)
        # After reshape
        # outputs (batch, units)
        outputs = tf.reshape(outputs, (outputs.shape[0], outputs.shape[2]))

        # Output word as a one hot vector in the target vocabulary
        y = self.F(outputs)

        return y, s
