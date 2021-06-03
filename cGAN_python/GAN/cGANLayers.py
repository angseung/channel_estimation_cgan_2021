import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import datetime
import h5py
import hdf5storage
import skfuzzy as fuzz

PRINT_DEBUG_OPT = True
num_tx = 64
num_user = 32
len_pilot = 8
num_chan = 2

def make_generator_model():
    initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=None)

    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(num_tx, len_pilot, num_chan)))
    assert model.output_shape == (None, num_tx, len_pilot, num_chan) # (64, 8, 2)
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
    model.add(layers.UpSampling2D(size=(2, 8), interpolation='nearest')) # (128, 64, 2)
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
    assert model.output_shape == (None, num_tx * 2, len_pilot * 8, num_chan)
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx, len_pilot * 4, num_chan),
                            kernel_initializer=initializer, use_bias=False))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
    assert model.output_shape == (None, num_tx, len_pilot * 4, 64)
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, num_tx, len_pilot * 4, 64)
    residual_relu = model.output

    # ENC 1
    model.add(layers.Conv2D(64 * 2, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx, len_pilot * 4, 64),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    model.add((layers.BatchNormalization()))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)

    # ENC 2
    model.add(layers.Conv2D(64 * 4, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx / 2, len_pilot * 4 / 2, 64 * 2),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    model.add((layers.BatchNormalization()))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)

    # ENC 3
    model.add(layers.Conv2D(64 * 8, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx / 2, len_pilot * 4 / 2, 64 * 4),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    model.add((layers.BatchNormalization()))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
    # (8, 4, 512)

    # DEC 1
    model.add(layers.Conv2DTranspose(64 * 4, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx / 8, len_pilot * 4 / 8, 64 * 8),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    model.add((layers.BatchNormalization()))
    model.add(layers.Dropout(0.5))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
    # (16, 8, 256)

    # DEC 2
    model.add(layers.Conv2DTranspose(64 * 2, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx / 4, len_pilot * 4 / 4, 64 * 4),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    model.add((layers.BatchNormalization()))
    model.add(layers.Dropout(0.5))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
    # (32, 16, 128)

    # DEC 3
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx / 2, len_pilot * 4 / 2, 64 * 2),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    model.add((layers.BatchNormalization()))
    model.add(layers.Dropout(0.5))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
    # (64, 32, 64)

    # DEC 4
    model.add(layers.Conv2DTranspose(64 / 2, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx, len_pilot * 4, 64),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    model.add((layers.BatchNormalization()))
    model.add(layers.Dropout(0.5))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
    # (128, 64, 32)

    model.add(layers.Conv2D(2, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx * 2, len_pilot * 8, 64 / 2),
                            kernel_initializer=initializer, use_bias=False))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
    # (64, 32, 2)

    return model


def make_discriminator_model():
    initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=None)

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (4, 4), strides=(1, 1), padding='same',
                            input_shape=(num_tx, num_user, num_chan),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
        # (64, 32, 64)

    # ENC 1
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx, num_user, 64),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
        # (32, 16, 64)

    # ENC 2
    model.add(layers.Conv2D(64 * 2, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx / 2, num_user / 2, 64),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
        # (16, 8, 128)

    # ENC 3
    model.add(layers.Conv2D(64 * 2, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx / 4, num_user / 4, 64),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
        # (8, 4, 128)

    # ENC 4
    model.add(layers.Conv2D(64 * 8, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx / 8, num_user / 8, 64 * 2),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


a = np.zeros((num_tx, len_pilot, num_chan))
m = make_discriminator_model()
print(m.output_shape)

