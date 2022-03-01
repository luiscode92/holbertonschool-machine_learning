#!/usr/bin/env python3
"""
Class that inherits from tensorflow.keras.Model
to create a transformer network.
https://www.tensorflow.org/tutorials/text/transformer#create_the_transformer
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Creates a transformer network
    """
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Class constructor
        """
        super().__init__()
        # the encoder layer
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input,
                               drop_rate)
        # the decoder layer
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target,
                               drop_rate)
        # a final Dense layer with target_vocab units
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, inputs, target, training,
             encoder_mask, look_ahead_mask, decoder_mask):
        """
        Returns: a tensor of shape (batch, target_seq_len, target_vocab)
        containing the transformer output
        """
        # (batch, input_seq_len, dm)
        enc_output = self.encoder(inputs, training, encoder_mask)

        # dec_output.shape == (batch, target_seq_len, dm)
        # Error: dec_output, _ = self.decoder(target...)
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)

        final_output = self.linear(dec_output)

        return final_output
