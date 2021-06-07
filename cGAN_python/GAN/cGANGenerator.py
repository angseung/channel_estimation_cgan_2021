import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import matplotlib.pyplot as plt

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config = config)
layers = tf.keras.layers

"""
The architecture of generator is a modified U-Net.
There are skip connections between the encoder and decoder (as in U-Net).
"""

class EncoderLayer(tf.keras.Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides_s = 2,
                 apply_batchnorm = True,
                 apply_dropout = False,
                 add = False,
                 padding_s = 'same'):

        super(EncoderLayer, self).__init__()

        # initializer = tf.keras.initializers.Orthogonal(gain = 1.0, seed = None)
        initializer = tf.random_normal_initializer(mean=0., stddev=0.02)

        conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides_s,
                             padding=padding_s, kernel_initializer=initializer, use_bias=False)
        ac = layers.LeakyReLU()
        self.encoder_layer = None

        if add:
            self.encoder_layer = tf.keras.Sequential([conv])

        elif apply_batchnorm:
            # bn = layers.BatchNormalization()
            bn = tfa.layers.InstanceNormalization()
            self.encoder_layer = tf.keras.Sequential([conv, bn, ac])

        elif apply_dropout:
            # bn = layers.BatchNormalization()
            bn = tfa.layers.InstanceNormalization()
            drop = layers.Dropout(rate=0.5)
            self.encoder_layer = tf.keras.Sequential([conv, bn, drop, ac])

        else:
            self.encoder_layer = tf.keras.Sequential([conv, ac])

    def call(self, x):
        return self.encoder_layer(x)


class DecoderLayer(tf.keras.Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides_s = 2,
                 apply_dropout = False,
                 add = False):

        super(DecoderLayer, self).__init__()

        # initializer = tf.keras.initializers.Orthogonal(gain = 1.0, seed = None)
        initializer = tf.random_normal_initializer(mean=0., stddev=0.02)

        dconv = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides_s,
                                       padding='same', kernel_initializer=initializer, use_bias=False)
        # bn = layers.BatchNormalization()
        bn = tfa.layers.InstanceNormalization()
        ac = layers.ReLU()
        self.decoder_layer = None
        
        if add:
            self.decoder_layer = tf.keras.Sequential([dconv])      
        elif apply_dropout:
            drop = layers.Dropout(rate=0.5)
            self.decoder_layer = tf.keras.Sequential([dconv, bn, drop, ac])
        else:
            self.decoder_layer = tf.keras.Sequential([dconv, bn, ac])

    def call(self, x):
        return self.decoder_layer(x)


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Resize Input
        p_layer_1 = DecoderLayer(filters=2, kernel_size=4, strides_s = 2, apply_dropout=False, add = True) 
        p_layer_2  = DecoderLayer(filters=2, kernel_size=4, strides_s = 2, apply_dropout=False, add = True)
        p_layer_3  = EncoderLayer(filters=2, kernel_size=(6,1),strides_s = (4,1), apply_batchnorm=False, add = True)
        
        self.p_layers = [p_layer_1,p_layer_2,p_layer_3]

        #encoder
        encoder_layer_1 = EncoderLayer(filters=64*1,  kernel_size=4,apply_batchnorm=False)   
        encoder_layer_2 = EncoderLayer(filters=64*2, kernel_size=4)       
        encoder_layer_3 = EncoderLayer(filters=64*4, kernel_size=4)       
        encoder_layer_4 = EncoderLayer(filters=64*8, kernel_size=4)       
        encoder_layer_5 = EncoderLayer(filters=64*8, kernel_size=4)       
        self.encoder_layers = [encoder_layer_1, encoder_layer_2, encoder_layer_3, encoder_layer_4,
                               encoder_layer_5]

        # decoder
        decoder_layer_1 = DecoderLayer(filters=64*8, kernel_size=4, apply_dropout=True)   
        decoder_layer_2 = DecoderLayer(filters=64*8, kernel_size=4,apply_dropout=True)   
        decoder_layer_3 = DecoderLayer(filters=64*8, kernel_size=4, apply_dropout=True)   
        decoder_layer_4 = DecoderLayer(filters=64*4, kernel_size=4)   
        self.decoder_layers = [decoder_layer_1, decoder_layer_2, decoder_layer_3, decoder_layer_4]

        # initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=None)
        initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
        self.last = layers.Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='tanh')

    def call(self, x):
        # pass the encoder and record xs
        for p_layer in self.p_layers:
            x = p_layer(x)

        encoder_xs = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encoder_xs.append(x)

        encoder_xs = encoder_xs[:-1][::-1]    # reverse
        assert len(encoder_xs) == 4

        # pass the decoder and apply skip connection
        for i, decoder_layer in enumerate(self.decoder_layers):
            x = decoder_layer(x)
            x = tf.concat([x, encoder_xs[i]], axis=-1)     # skip connect

        return self.last(x)        # last


class GeneratorRev(tf.keras.Model):
    def __init__(self,
                 n_path = 3):

        super(GeneratorRev, self).__init__()
        self.n_path = n_path * 2

        # Resize Input
        p_layer_1 = DecoderLayer(filters=2, kernel_size=4, strides_s=2, apply_dropout=False, add=True)
        p_layer_2 = DecoderLayer(filters=2, kernel_size=4, strides_s=2, apply_dropout=False, add=True)
        p_layer_3 = EncoderLayer(filters=2, kernel_size=(6, 1), strides_s=(4, 1), apply_batchnorm=False, add=True)

        self.p_layers = [p_layer_1, p_layer_2, p_layer_3]

        # encoder
        encoder_layer_1  = EncoderLayer(filters=64 * 1, kernel_size=4, apply_batchnorm=False)
        encoder_layer_1n = EncoderLayer(filters=64 * 2, kernel_size=4, strides_s=1)
        encoder_layer_2  = EncoderLayer(filters=64 * 2, kernel_size=4)
        encoder_layer_2n = EncoderLayer(filters=64 * 2, kernel_size=4, strides_s=1)
        encoder_layer_3  = EncoderLayer(filters=64 * 4, kernel_size=4)
        encoder_layer_3n = EncoderLayer(filters=64 * 4, kernel_size=4, strides_s=1)
        encoder_layer_4  = EncoderLayer(filters=64 * 8, kernel_size=4)
        encoder_layer_4n = EncoderLayer(filters=64 * 8, kernel_size=4, strides_s=1)
        encoder_layer_5  = EncoderLayer(filters=64 * 8, kernel_size=4)
        self.encoder_layers = [encoder_layer_1,
                               encoder_layer_1n,
                               encoder_layer_2,
                               encoder_layer_2n,
                               encoder_layer_3,
                               encoder_layer_3n,
                               encoder_layer_4,
                               encoder_layer_4n,
                               encoder_layer_5]

        # decoder
        decoder_layer_1  = DecoderLayer(filters=64 * 8, kernel_size=4, apply_dropout=True)
        decoder_layer_1n = DecoderLayer(filters=64 * 8, kernel_size=4, apply_dropout=True, strides_s=1)
        decoder_layer_2  = DecoderLayer(filters=64 * 4, kernel_size=4, apply_dropout=True)
        decoder_layer_2n = DecoderLayer(filters=64 * 4, kernel_size=4, apply_dropout=True, strides_s=1)
        decoder_layer_3  = DecoderLayer(filters=64 * 2, kernel_size=4, apply_dropout=True)
        decoder_layer_3n = DecoderLayer(filters=64 * 2, kernel_size=4, apply_dropout=True, strides_s=1)
        decoder_layer_4  = DecoderLayer(filters=64 * 2, kernel_size=4)
        decoder_layer_4n = DecoderLayer(filters=64 * 1, kernel_size=4, apply_dropout=True, strides_s=1)
        self.decoder_layers = [decoder_layer_1,
                               decoder_layer_1n,
                               decoder_layer_2,
                               decoder_layer_2n,
                               decoder_layer_3,
                               decoder_layer_3n,
                               decoder_layer_4,
                               decoder_layer_4n]

        # initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=None)
        initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
        self.last = layers.Conv2DTranspose(filters=self.n_path, kernel_size=4, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='tanh')

    def call(self, x):
        # pass the encoder and record xs
        for p_layer in self.p_layers:
            # DEC -> DEC -> ENC
            x = p_layer(x)
            # print(x.shape)

        encoder_xs = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x) ## (1, 64, 32, 2)
            # print(x.shape)
            encoder_xs.append(x)
            ## [ENC1, ENC2, ENC3, ENC4, ENC5]

        encoder_xs = encoder_xs[:-1][::-1]  # reverse
        ## [ENC5, ENC4, ENC3, ENC2]
        # assert (len(encoder_xs) == 4)

        # pass the decoder and apply skip connection
        for i, decoder_layer in enumerate(self.decoder_layers):
            # x : (1, 2, 1, 512)
            x = decoder_layer(x)
            x = tf.concat([x, encoder_xs[i]], axis=-1)  # skip connect
            # print(x.shape)


        return self.last(x)  # last
