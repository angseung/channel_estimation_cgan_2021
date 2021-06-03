import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import datetime
import h5py
import hdf5storage
import skfuzzy as fuzz

PRINT_DEBUG_OPT = False
num_tx = 64
num_rx = 32
len_pilot = 8
num_chan = 2

def make_generator_model():
    # initializer = tf.keras.initializers.Orthogonal(gain = 1.0, seed = None)
    initializer = tf.random_normal_initializer(mean=0., stddev=0.02)

    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(num_rx, len_pilot, num_chan)))
    assert model.output_shape == (None, num_rx, len_pilot, num_chan)
    if (PRINT_DEBUG_OPT):
        print(model.output_shape) # (32, 8, 2)
    model.add(layers.UpSampling2D(size=(2, 16), interpolation='nearest'))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape) # (64, 128, 2)
    assert model.output_shape == (None, num_rx * 2, len_pilot * 16, num_chan)
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_rx * 2, len_pilot * 16, num_chan),
                            kernel_initializer=initializer, use_bias=False))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape) # (32, 64, 64)
    assert model.output_shape == (None, num_rx, len_pilot * 8, 64)
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, num_rx, len_pilot * 8, 64)

    # ENC 1
    model.add(layers.Conv2D(64 * 2, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_rx, len_pilot * 8, 64),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    # model.add((layers.BatchNormalization()))
    model.add(tfa.layers.InstanceNormalization())
    if (PRINT_DEBUG_OPT):
        print(model.output_shape) # (16, 32, 128)

    # ENC 2
    model.add(layers.Conv2D(64 * 4, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_rx / 2, len_pilot * 4, 64 * 2),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    # model.add((layers.BatchNormalization()))
    model.add(tfa.layers.InstanceNormalization())
    if (PRINT_DEBUG_OPT):
        print(model.output_shape) # (8, 16, 256)

    # ENC 3
    model.add(layers.Conv2D(64 * 8, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_rx / 4, len_pilot * 2, 64 * 4),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    # model.add((layers.BatchNormalization()))
    model.add(tfa.layers.InstanceNormalization())
    if (PRINT_DEBUG_OPT):
        print(model.output_shape) # (4, 8, 512)

    # DEC 1
    model.add(layers.UpSampling2D(size=(2, 2)))
    # model.add(layers.Conv2DTranspose(64 * 4, (4, 4), strides=(2, 2), padding='same',
    #                         input_shape=(num_rx / 8, len_pilot, 64 * 8),
    #                         kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    # model.add((layers.BatchNormalization()))
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.Dropout(0.5))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape) # (8, 16, 256)

    # DEC 2
    model.add(layers.UpSampling2D(size=(2, 2)))
    # model.add(layers.Conv2DTranspose(64 * 2, (4, 4), strides=(2, 2), padding='same',
    #                         input_shape=(num_rx / 4, len_pilot * 2, 64 * 4),
    #                         kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    # model.add((layers.BatchNormalization()))
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.Dropout(0.5))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape) # (16, 32, 128)

    # DEC 3
    model.add(layers.UpSampling2D(size=(2, 2)))
    # model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same',
    #                         input_shape=(num_rx / 2, len_pilot * 4, 64 * 2),
    #                         kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    # model.add((layers.BatchNormalization()))
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.Dropout(0.5))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape) # (32, 64, 64)

    # DEC 4
    model.add(layers.UpSampling2D(size=(2, 2)))
    # model.add(layers.Conv2DTranspose(64 / 2, (4, 4), strides=(2, 2), padding='same',
    #                         input_shape=(num_rx, len_pilot * 8, 64),
    #                         kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    # model.add((layers.BatchNormalization()))
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.Dropout(0.5))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape) # (64, 128, 32)

    model.add(layers.Conv2D(2, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_rx * 2, len_pilot * 16, 64 / 2),
                            kernel_initializer=initializer, use_bias=False))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape) # (32, 64, 2)

    return model


def make_discriminator_model():
    initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=None)

    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(num_rx, num_tx, num_chan)))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
        # (32, 64, 2)
    model.add(layers.Conv2D(64, (4, 4), strides=(1, 1), padding='same',
                            input_shape=(num_rx, num_tx, num_chan),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
        # (32, 64, 64)

    # ENC 1
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx, num_rx, 64),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    # model.add(layers.BatchNormalization())
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.Dropout(0.5))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
        # (16, 32, 64)

    # ENC 2
    model.add(layers.Conv2D(64 * 2, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx / 2, num_rx / 2, 64),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    # model.add(layers.BatchNormalization())
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.Dropout(0.5))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
        # (8, 16, 128)

    # ENC 3
    model.add(layers.Conv2D(64 * 2, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx / 4, num_rx / 4, 64),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    # model.add(layers.BatchNormalization())
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.Dropout(0.5))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
        # (4, 8, 128)

    # ENC 4
    model.add(layers.Conv2D(64 * 4, (4, 4), strides=(2, 2), padding='same',
                            input_shape=(num_tx / 8, num_rx / 8, 64 * 2),
                            kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    # model.add(layers.BatchNormalization())
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.Dropout(0.5))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)
        # (2, 4, 512)

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    if (PRINT_DEBUG_OPT):
        print(model.output_shape)

    return model

if (__name__ == "__main__"):
    a = np.zeros((num_tx, len_pilot, num_chan))
    m = make_generator_model()
    print("*****")
    m = make_discriminator_model()
    # print(m.output_shape)

