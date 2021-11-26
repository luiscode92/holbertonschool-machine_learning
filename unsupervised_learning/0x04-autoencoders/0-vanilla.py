#!/usr/bin/env python3
"""
Creates an autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Returns: encoder, decoder, auto
    """
    # Encoder
    inputs_enc = keras.Input(shape=(input_dims,))

    layer_enc = keras.layers.Dense(units=hidden_layers[0],
                                   activation='relu')(inputs_enc)

    for i in range(1, len(hidden_layers)):
        layer_enc = keras.layers.Dense(units=hidden_layers[i],
                                       activation='relu')(layer_enc)

    layer_enc = keras.layers.Dense(units=latent_dims,
                                   activation='relu')(layer_enc)

    encoder = keras.Model(inputs=inputs_enc, outputs=layer_enc)

    # Decoder
    inputs_dec = keras.Input(shape=(latent_dims,))

    layer_dec = keras.layers.Dense(units=hidden_layers[-1],
                                   activation='relu')(inputs_dec)

    for i in range(len(hidden_layers) - 2, -1, -1):
        layer_dec = keras.layers.Dense(units=hidden_layers[i],
                                       activation='relu')(layer_dec)

    layer_dec = keras.layers.Dense(units=input_dims,
                                   activation='sigmoid')(layer_dec)

    decoder = keras.Model(inputs=inputs_dec, outputs=layer_dec)

    # Autoencoder
    auto_bottleneck = encoder(inputs_enc)
    auto_output = decoder(auto_bottleneck)

    auto = keras.Model(inputs=inputs_enc, outputs=auto_output)

    auto.compile(optimizer=keras.optimizers.Adam(),
                 loss='binary_crossentropy')

    return encoder, decoder, auto
