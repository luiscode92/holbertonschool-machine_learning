#!/usr/bin/env python3
"""
Transformer from project 0x11. Attention.
You may need to make slight adjustments to this model
to get it to functionally train.
https://www.tensorflow.org/text/tutorials/transformer
"""
import tensorflow.compat.v2 as tf
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer.
    Returns: a numpy.ndarray of shape (max_seq_len, dm)
    containing the positional encoding vectors
    """
    PE = np.zeros([max_seq_len, dm])

    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            # sin to even indices
            PE[pos, i] = np.sin(pos / (10000 ** (i / dm)))
            # cos to odd indices
            PE[pos, i + 1] = np.cos(pos / (10000 ** (i / dm)))

    return PE


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.
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
