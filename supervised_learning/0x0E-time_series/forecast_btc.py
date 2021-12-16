#!/usr/bin/env python3
"""
Creates, trains, and validates a keras model for the forecasting of BTC
Tensorflow 2.4.1
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class WindowGenerator:
    """
    Data windowing
    """
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        """
        Class constructor
        shift = offset
        """
        # Store the raw data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices
        self.label_columns = label_columns

        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}

        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = \
            np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = \
            np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        """
        Returns: inputs, labels
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                 for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='Close', max_subplots=3):
        """
        Returns: nothing
        """
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        # labels = labels*3723.88566 + 5774.6824
        # inputs = inputs*3723.88566 + 5774.6824

        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col,
                                                                 None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)

            if model is not None:
                predictions = model(inputs)
                # tf.expand_dims(predictions, -1)
                # checking shapes
                # print("check_1", labels.shape)
                # print("check_2", predictions.shape)
                plt.scatter(self.label_indices,
                            predictions[n, 0],
                            marker='X', edgecolors='k',
                            label='Predictions',
                            c='#ff7f0e', s=64)
                # labels[n, :, label_col_index]
                # predictions[n, 0]*3723.88566 + 5774.6824
                # below it works
                # self.label_indices
                # predictions[n, 0],

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data):
        """
        Creates tf.data.Datasets
        Returns: ds
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=24,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


def compile_and_fit(model, window, patience=2, epochs=20):
    """
    Returns: history
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])

    return history


def forecasting(train, validation, test):
    """
    Returns: nothing
    """
    window = WindowGenerator(input_width=24, label_width=1, shift=1,
                             train_df=train, val_df=validation, test_df=test,
                             label_columns=['Close'])

    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(24, return_sequences=False),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)])

    # Train the model and evaluate its performance
    history = compile_and_fit(lstm_model, window)

    val_performance = {}
    performance = {}
    val_performance['LSTM'] = lstm_model.evaluate(window.val)
    performance['LSTM'] = lstm_model.evaluate(window.test, verbose=0)

    window.plot(lstm_model)


# here goes the main file

if __name__ == "__main__":
    preprocess = __import__('preprocess_data').preprocess_raw_data

    train, validation, test = preprocess()

    forecasting(train, validation, test)
