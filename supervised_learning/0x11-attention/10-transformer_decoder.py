#!/usr/bin/env python3
"""
Class that inherits from tensorflow.keras.layers.Layer
to create the decoder for a transformer.
https://www.tensorflow.org/tutorials/text/transformer#decoder
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Creates the decoder for a transformer
    """
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor
        """
        super().__init__()
        # number of blocks in the decoder
        self.N = N
        # dimensionality of the model
        self.dm = dm
        # the embedding layer for the targets
        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab,
                                                   output_dim=dm)
        # numpy.ndarray (max_seq_len, dm) containing the positional encodings
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        # a list of length N containing all of the DecoderBlock‘s
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        # the dropout layer, to be applied to the positional encodings
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """
        Returns: a tensor of shape (batch, target_seq_len, dm)
        containing the decoder output
        """
        target_seq_len = x.shape[1]

        # Compute the embeddings
        # (batch, target_seq_len, dm)
        embeddings = self.embedding(x)
        # Scale the embeddings
        embeddings *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        # Sum the positional encodings with the embeddings
        embeddings += self.positional_encoding[:target_seq_len]

        output = self.dropout(embeddings, training=training)

        for i in range(self.N):
            output = self.blocks[i](output, encoder_output, training,
                                    look_ahead_mask, padding_mask)

        return output
