#!/usr/bin/env python3
"""
Creates a variational autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Returns: encoder, decoder, auto
    """
    # Encoder
    inputs_enc = keras.Input(shape=(input_dims, ))

    layer_enc = keras.layers.Dense(units=hidden_layers[0],
                                   activation='relu')(inputs_enc)

    for layer in hidden_layers[1:]:
        layer_enc = keras.layers.Dense(units=layer,
                                       activation='relu')(layer_enc)

    z_mean = keras.layers.Dense(latent_dims)(layer_enc)

    z_log_sigma = keras.layers.Dense(latent_dims)(layer_enc)

    def sampling(inputs_enc):
        """
        Reparametrization trick: Sample from a normal standard distribution
        Uses (z_mean, z_log_sigma) to sample z, the vector encoding a digit
        https://keras.io/examples/generative/vae/
        """
        z_mean, z_log_sigma = inputs_enc

        batch = keras.backend.shape(z_mean)[0]

        dims = keras.backend.int_shape(z_mean)[1]

        epsilon = keras.backend.random_normal(shape=(batch, dims))

        z = z_mean + keras.backend.exp(0.5 * z_log_sigma) * epsilon

        return z

    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims, ))([z_mean,
                                                           z_log_sigma])

    encoder = keras.models.Model(inputs=inputs_enc,
                                 outputs=[z, z_mean, z_log_sigma])

    # Decoder
    inputs_dec = keras.Input(shape=(latent_dims, ))

    layer_dec = keras.layers.Dense(units=hidden_layers[-1],
                                   activation='relu')(inputs_dec)

    for layer in reversed(hidden_layers[:-1]):
        layer_dec = keras.layers.Dense(units=layer,
                                       activation='relu')(layer_dec)

    layer_dec = keras.layers.Dense(units=input_dims,
                                   activation='sigmoid')(layer_dec)

    decoder = keras.models.Model(inputs=inputs_dec, outputs=layer_dec)

    # Autoencoder
    auto_bottleneck = encoder(inputs_enc)[0]
    auto_output = decoder(auto_bottleneck)

    auto = keras.Model(inputs=inputs_enc, outputs=auto_output)

    def variational_autoencoder_loss(inputs, outputs):
        """
        https://blog.keras.io/building-autoencoders-in-keras.html
        """
        reconstruction_loss = keras.losses.binary_crossentropy(inputs_enc,
                                                               auto_output)

        reconstruction_loss *= input_dims

        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean)\
            - keras.backend.exp(z_log_sigma)

        kl_loss = keras.backend.sum(kl_loss, axis=-1)

        kl_loss *= -0.5

        vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)

        return vae_loss

    auto.compile(optimizer='Adam', loss=variational_autoencoder_loss)

    return encoder, decoder, auto
