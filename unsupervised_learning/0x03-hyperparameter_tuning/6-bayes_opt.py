#!/usr/bin/env python3
"""
Trains a CNN
to classify the CIFAR 10 dataset.

Optimizes an ML model of your choice using GPyOpt.
"""
import tensorflow.keras as K
from GPyOpt.methods import BayesianOptimization


def preprocess_data(X, Y):
    """
    Returns: X_p, Y_p
    """
    # Preprocessing needed in each Keras Application
    X_p = K.applications.densenet.preprocess_input(X)
    # Converts a label vector into a one-hot matrix
    Y_p = K.utils.to_categorical(Y, num_classes=10)

    return X_p, Y_p


# Dataset of 50,000 32x32 color training images and 10,000 test images,
# labeled over 10 categories
(x_train, y_train), (X, Y) = K.datasets.cifar10.load_data()

# preprocessing
x_train_p, y_train_p = preprocess_data(x_train, y_train)
x_test_p, y_test_p = preprocess_data(X, Y)

# DenseNet121
#  loads weights pre-trained on ImageNet
dense_121 = K.applications.DenseNet121(weights='imagenet',
                                       include_top=False,
                                       input_shape=(224, 224, 3))

dense_121.trainable = False

input = K.Input(shape=(32, 32, 3))
# lambtha layer scales up the data to the correct size
lambtha = K.layers.Lambda(lambda X: K.backend.resize_images(X, 7, 7,
                          data_format="channels_last",
                          interpolation='bilinear'))(input)
output = dense_121(lambtha, training=False)
output = K.layers.Flatten()(output)
output = K.layers.Dense(512, activation='relu')(output)
output = K.layers.Dropout(0.2)(output)
output = K.layers.Dense(256, activation='relu')(output)
output = K.layers.Dropout(0.2)(output)
output = K.layers.Dense(128, activation='relu')(output)
output = K.layers.Dropout(0.2)(output)
output = K.layers.Dense(10, activation='softmax')(output)

model = K.Model(input, output)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['acc'])

# training
history = model.fit(x=x_train_p, y=y_train_p,
                    batch_size=128, epochs=5,
                    validation_data=(x_test_p, y_test_p),
                    verbose=1)

model.save('cifar10.h5')
