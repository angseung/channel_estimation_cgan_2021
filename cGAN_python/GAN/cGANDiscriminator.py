import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from GAN.cGANGenerator import EncoderLayer
import os

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
layers = tf.keras.layers

"""
The Discriminator is a PatchGAN.
"""

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        # downsample
        self.encoder_layer_1 = EncoderLayer(filters=64, kernel_size=4, apply_batchnorm=False)
        self.encoder_layer_2 = EncoderLayer(filters=128, kernel_size=4)
        self.encoder_layer_3 = EncoderLayer(filters=128, kernel_size=4)

        # conv block1
        self.zero_pad1 = layers.ZeroPadding2D()                                
        self.conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        # self.bn1 = tfa.layers.InstanceNormalization()
        self.ac = layers.LeakyReLU()

        # block2
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()                       
        self.last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer) 

    def call(self, y):
        """inputs can be generated image. """
        target = y
        x = target     
        x = self.encoder_layer_1(x)
        x = self.encoder_layer_2(x)
        x = self.encoder_layer_3(x)

        x = self.zero_pad1(x)
        # print(x.shape)
        x = self.conv(x)
        x = self.bn1(x)
        x = self.ac(x)

        x = self.zero_pad2(x)
        x = self.last(x)
        return x

class DiscriminatorOri(tf.keras.Model):
    def __init__(self):
        super(DiscriminatorOri, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        # downsample
        self.encoder_layer_1 = EncoderLayer(filters=64, kernel_size=4, apply_batchnorm=False)
        self.encoder_layer_2 = EncoderLayer(filters=128, kernel_size=4)
        self.encoder_layer_3 = EncoderLayer(filters=128, kernel_size=4)

        # conv block1
        self.zero_pad1 = layers.ZeroPadding2D()
        self.conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        # self.bn1 = tfa.layers.InstanceNormalization()
        self.ac = layers.LeakyReLU()

        # block2
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        self.last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)

    def call(self, y):
        """inputs can be generated image. """
        target = y
        x = target
        x = self.encoder_layer_1(x)
        x = self.encoder_layer_2(x)
        x = self.encoder_layer_3(x)

        x = self.zero_pad1(x)
        # print(x.shape)
        x = self.conv(x)
        x = self.bn1(x)
        x = self.ac(x)

        x = self.zero_pad2(x)
        x = self.last(x)
        return x

class DiscriminatorRev(tf.keras.Model):
    def __init__(self, dropout_rate = 0.3):
        super(DiscriminatorRev, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        # downsample
        # input : (None, 64, 32, 6)
        self.encoder_layer_1 = EncoderLayer(filters=64, kernel_size=4, apply_dropout=True)
        # output : (None, 32, 16, 64)
        self.encoder_layer_2 = EncoderLayer(filters=128, kernel_size=4, apply_dropout=True)
        # output : (None, 16, 8, 128)
        self.encoder_layer_3 = EncoderLayer(filters=128, kernel_size=4, apply_dropout=True)
        # output : (None, 8, 4, 128)
        self.encoder_layer_4 = EncoderLayer(filters=256, kernel_size=4, strides_s=1, apply_dropout=True)
        # output : (None, 8, 4, 256)
        # self.encoder_layer_5 = EncoderLayer(filters=256, kernel_size=4, strides_s=1)
        # output : (None, 8, 4, 256)

        # conv block1
        self.zero_pad1 = layers.ZeroPadding2D()
        # output : (None, 10, 6, 256)
        self.conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)
        # output : (None, 4, 2, 512)
        self.bn1 = layers.BatchNormalization()
        # self.bn1 = tfa.layers.InstanceNormalization()
        self.ac = layers.LeakyReLU()

        # block2
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        # self.conv2 = tf.keras.layers.Conv2D(256, 4, strides=1, kernel_initializer=initializer, use_bias=False)
        # self.conv3 = tf.keras.layers.Conv2D(128, 4, strides=1, kernel_initializer=initializer, use_bias=False)
        # self.conv4 = tf.keras.layers.Conv2D(64, 4, strides=1, kernel_initializer=initializer, use_bias=False)

        self.cv2 = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer,
                                          activation=tf.keras.layers.LeakyReLU(alpha=0.2))

        ## output
        # self.relu = layers.LeakyReLU()
        self.DL = layers.Dropout(rate=dropout_rate)
        # self.FL = layers.Flatten()
        # self.DS = layers.Dense(1)

    def call(self, y):
        """inputs can be generated image. """
        # target = y
        # x = target
        x = self.encoder_layer_1(y)
        x = self.encoder_layer_2(x)
        x = self.encoder_layer_3(x)
        x = self.encoder_layer_4(x)
        # x = self.encoder_layer_5(x)
        # print(x.shape)

        x = self.zero_pad1(x) # (2, 2)
        # print(x.shape)
        x = self.conv(x)
        x = self.bn1(x)
        x = self.ac(x)
        # print(x.shape)

        x = self.zero_pad2(x) # (2, 2)
        # print(x.shape)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        x = self.cv2(x)

        # x = self.relu(x)
        x = self.DL(x)
        # x = self.FL(x)
        # x = self.DS(x)

        return x

class DiscriminatorRev2(tf.keras.Model):
    def __init__(self, dropout_rate=0.5):
        super(DiscriminatorRev2, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        # downsample
        # input : (None, 64, 32, 6)
        self.encoder_layer_1 = EncoderLayer(filters=64, kernel_size=4, apply_dropout=True)
        # output : (None, 32, 16, 64)
        self.encoder_layer_2 = EncoderLayer(filters=128, kernel_size=4, apply_dropout=True)
        # output : (None, 16, 8, 128)
        self.encoder_layer_3 = EncoderLayer(filters=128, kernel_size=4, apply_dropout=True)
        # output : (None, 8, 4, 128)
        self.encoder_layer_4 = EncoderLayer(filters=256, kernel_size=4, strides_s=1, apply_dropout=True)
        # output : (None, 8, 4, 256)
        # self.encoder_layer_5 = EncoderLayer(filters=256, kernel_size=4, strides_s=1)
        # output : (None, 8, 4, 256)

        # conv block1
        self.zero_pad1 = layers.ZeroPadding2D()
        # output : (None, 10, 6, 256)
        self.conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)
        # output : (None, 4, 2, 512)
        self.bn1 = layers.BatchNormalization()
        # self.bn1 = tfa.layers.InstanceNormalization()
        self.ac = layers.LeakyReLU()

        # block2
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        # self.conv2 = tf.keras.layers.Conv2D(256, 4, strides=1, kernel_initializer=initializer, use_bias=False)
        # self.conv3 = tf.keras.layers.Conv2D(128, 4, strides=1, kernel_initializer=initializer, use_bias=False)
        # self.conv4 = tf.keras.layers.Conv2D(64, 4, strides=1, kernel_initializer=initializer, use_bias=False)

        self.cv2 = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)

        ## output
        self.relu = layers.LeakyReLU()
        self.DL = layers.Dropout(rate=dropout_rate)
        # self.FL = layers.Flatten()
        # self.DS = layers.Dense(1, activation="sigmoid")

    def call(self, y):
        """inputs can be generated image. """
        # target = y
        # x = target
        x = self.encoder_layer_1(y)
        x = self.encoder_layer_2(x)
        x = self.encoder_layer_3(x)
        x = self.encoder_layer_4(x)
        # x = self.encoder_layer_5(x)
        # print(x.shape)

        x = self.zero_pad1(x)  # (2, 2)
        # print(x.shape)
        x = self.conv(x)
        x = self.bn1(x)
        x = self.ac(x)
        # print(x.shape)

        x = self.zero_pad2(x)  # (2, 2)
        # print(x.shape)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        x = self.cv2(x)

        x = self.relu(x)
        x = self.DL(x)
        # x = self.FL(x)
        # x = self.DS(x)

        return x













