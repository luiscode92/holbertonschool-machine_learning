#!/usr/bin/env python3
"""
Trains a model using mini-batch gradient descent.
Also analyzes validation data.
Also trains the model using early stopping
Also trains the model with learning rate decay
Also saves the best iteration of the model
"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Returns: the History object generated after training the model
    """
    def learning_rate(epoch):
        """ updates the learning rate using inverse time decay """
        return alpha / (1 + decay_rate * epoch)

    callback_list = []

    # models save callback
    if save_best:
        the_best = K.callbacks.ModelCheckpoint(filepath,
                                               save_best_only=True,
                                               monitor='val_loss',
                                               mode='min')
        callback_list.append(the_best)
    # learning rate decay callback
    if validation_data and learning_rate_decay:
        lrd = K.callbacks.LearningRateScheduler(learning_rate,
                                                verbose=1)
        callback_list.append(lrd)

    # early stopping callback
    if validation_data and early_stopping:
        es = K.callbacks.EarlyStopping(monitor='val_loss',
                                       mode='min',
                                       patience=patience)
        callback_list.append(es)

    # training
    History = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle,
                          callbacks=callback_list)

    return History
