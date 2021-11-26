#!/usr/bin/env python3
"""
Creates a convolutional autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Returns: encoder, decoder, auto
    """
    # Encoder
    inputs_enc = keras.Input(shape=input_dims)

    layer_enc = keras.layers.Conv2D(filters=filters[0], kernel_size=3,
                                    padding='same',
                                    activation='relu')(inputs_enc)

    layer_enc = keras.layers.MaxPool2D(pool_size=2,
                                       padding='same')(layer_enc)

    for i in range(1, len(filters)):
        layer_enc = keras.layers.Conv2D(filters=filters[i], kernel_size=3,
                                        padding='same',
                                        activation='relu')(layer_enc)

        layer_enc = keras.layers.MaxPool2D(pool_size=2,
                                           padding='same')(layer_enc)

    encoder = keras.Model(inputs=inputs_enc, outputs=layer_enc)

    # Decoder
    inputs_dec = keras.Input(shape=latent_dims)

    layer_dec = keras.layers.Conv2D(filters=filters[-1], kernel_size=3,
                                    padding='same',
                                    activation='relu')(inputs_dec)

    layer_dec = keras.layers.UpSampling2D(size=2)(layer_dec)

    for i in range(len(filters) - 1, 1, -1):
        layer_dec = keras.layers.Conv2D(filters=filters[i], kernel_size=3,
                                        padding='same',
                                        activation='relu')(layer_dec)

        layer_dec = keras.layers.UpSampling2D(size=2)(layer_dec)

    # second to last convolution
    layer_dec = keras.layers.Conv2D(filters=filters[0], kernel_size=3,
                                    padding='valid',
                                    activation='relu')(layer_dec)

    layer_dec = keras.layers.UpSampling2D(size=2)(layer_dec)

    # last convolution layer in the decoder
    layer_dec = keras.layers.Conv2D(filters=input_dims[-1], kernel_size=3,
                                    padding='same',
                                    activation='sigmoid')(layer_dec)

    decoder = keras.Model(inputs=inputs_dec, outputs=layer_dec)

    # Autoencoder
    auto_bottleneck = encoder(inputs_enc)
    auto_output = decoder(auto_bottleneck)

    auto = keras.Model(inputs=inputs_enc, outputs=auto_output)

    auto.compile(optimizer=keras.optimizers.Adam(),
                 loss='binary_crossentropy')

    return encoder, decoder, auto
