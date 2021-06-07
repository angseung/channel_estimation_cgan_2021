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
        # self.bn1 = layers.BatchNormalization()
        self.bn1 = tfa.layers.InstanceNormalization()
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
        x = self.conv(x)
        x = self.bn1(x)
        x = self.ac(x)

        x = self.zero_pad2(x)
        x = self.last(x)
        return x


class DiscriminatorRev(tf.keras.Model):
    def __init__(self, dropout_prob = 0.3):
        super(DiscriminatorRev, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        # downsample
        self.encoder_layer_1 = EncoderLayer(filters=64, kernel_size=4, apply_batchnorm=False)
        self.encoder_layer_2 = EncoderLayer(filters=128, kernel_size=4)
        self.encoder_layer_3 = EncoderLayer(filters=128, kernel_size=4)
        self.encoder_layer_4 = EncoderLayer(filters=256, kernel_size=4, strides_s=1)
        self.encoder_layer_5 = EncoderLayer(filters=256, kernel_size=4, strides_s=1)

        # conv block1
        self.zero_pad1 = layers.ZeroPadding2D()
        self.conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)
        # self.bn1 = layers.BatchNormalization()
        self.bn1 = tfa.layers.InstanceNormalization()
        self.ac = layers.LeakyReLU()

        # block2
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        self.cv2 = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)

        ## output
        self.relu = layers.LeakyReLU()
        self.DL = layers.Dropout(dropout_prob)
        self.DS = layers.Dense(1)

    def call(self, y):
        """inputs can be generated image. """
        # target = y
        # x = target
        x = self.encoder_layer_1(y)
        x = self.encoder_layer_2(x)
        x = self.encoder_layer_3(x)
        x = self.encoder_layer_4(x)
        x = self.encoder_layer_5(x)

        x = self.zero_pad1(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = self.ac(x)

        x = self.zero_pad2(x)
        x = self.cv2(x)

        x = self.relu(x)
        x = self.DL(x)
        x = self.DS(x)

        return x













