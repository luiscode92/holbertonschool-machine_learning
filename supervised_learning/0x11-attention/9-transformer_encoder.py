#!/usr/bin/env python3
"""
Class that inherits from tensorflow.keras.layers.Layer
to create the encoder for a transformer.
https://www.tensorflow.org/tutorials/text/transformer#encoder
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Creates the encoder for a transformer
    """
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor
        """
        super().__init__()
        # number of blocks in the encoder
        self.N = N
        # dimensionality of the model
        self.dm = dm
        # the embedding layer for the inputs
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab,
                                                   output_dim=dm)
        # numpy.ndarray (max_seq_len, dm) containing the positional encodings
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        # the dropout layer, to be applied to the positional encodings
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask):
        """
        Returns: a tensor of shape (batch, input_seq_len, dm)
        containing the encoder output
        """
        # input_seq_len = tf.shape(x)[1]
        # TypeError: slice indices must be integers
        # or None or have an __index__ method
        input_seq_len = x.shape[1]

        # Compute the embeddings
        # (batch, input_seq_len, dm)
        embeddings = self.embedding(x)
        # Scale the embeddings
        embeddings *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        # Sum the positional encodings with the embeddings
        embeddings += self.positional_encoding[:input_seq_len]

        output = self.dropout(embeddings, training=training)

        for i in range(self.N):
            output = self.blocks[i](output, training, mask)

        return output
