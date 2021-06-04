import numpy as np
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
from GAN.cGANGenerator import Generator, GeneratorRev
from GAN.cGANDiscriminator import Discriminator
from GAN.data_preprocess import load_image_train, load_image_test, load_image_test_y
from tempfile import TemporaryFile
from scipy.io import loadmat, savemat
import datetime
import h5py
import hdf5storage
import skfuzzy as fuzz


"""
Discriminator loss:
The discriminator loss function takes 2 inputs; real images, generated images
real_loss is a sigmoid cross entropy loss of the real images and an array of ones(since the real images)
generated_loss is a sigmoid cross entropy loss of the generated images and an array of zeros(since the fake images)
Then the total_loss is the sum of real_loss and the generated_loss

Generator loss:
It is a sigmoid cross entropy loss of the generated images and an array of ones.
The paper also includes L2 loss between the generated image and the target image.
This allows the generated image to become structurally similar to the target image.
The formula to calculate the total generator loss = gan_loss + LAMBDA * l2_loss, where LAMBDA = 100. 
This value was decided by the authors of the paper.
"""
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss_custom(disc_real_output, disc_generated_output):
    """disc_real_output = [real_target]
       disc_generated_output = [generated_target]
    """
    # (DIS)
    real_loss = disc_real_output

    # (DIS(GEN))
    generated_loss = disc_generated_output

    total_disc_loss = tf.reduce_mean(real_loss) - tf.reduce_mean(generated_loss)

    return total_disc_loss


def generator_loss_custom(disc_generated_output, gen_output, target, l2_weight = 100):
    """
        disc_generated_output: output of Discriminator when input is from Generator
        gen_output:  output of Generator (i.e., estimated H)
        target:  target image
        l2_weight: weight of L2 loss
    """
    # GAN loss
    # gen_loss = DIS(GEN)
    # total_loss = total_loss + l2_loss
    gen_loss = disc_generated_output

    # L2 loss
    l2_loss = tf.reduce_mean(tf.abs(target - gen_output))  # loss with target...
    total_gen_loss = tf.reduce_mean(gen_loss) + l2_weight * l2_loss  ## Type : tf.tensor()
    return total_gen_loss


def discriminator_loss(disc_real_output, disc_generated_output, l2_weight = 0.0001, L2_OPT = False):
    """disc_real_output = [real_target]
       disc_generated_output = [generated_target]
    """
    # log(DIS)
    real_loss = cross_entropy(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = cross_entropy(tf.zeros_like(disc_generated_output), disc_generated_output)
    # total_loss = real_loss + generated_loss

    # real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=tf.ones_like(disc_real_output), logits=disc_real_output)  # label=1

    # log(1-DIS(GEN))
    # generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=tf.zeros_like(disc_generated_output), logits=disc_generated_output)  # label=0

    # L2 loss
    l2_loss = tf.reduce_mean(tf.abs(disc_real_output - disc_generated_output))  # loss with target...

    total_loss = real_loss + generated_loss + (L2_OPT * l2_weight * l2_loss)

    # total_disc_loss = tf.reduce_mean(real_loss) \
    #                   + tf.reduce_mean(generated_loss) \
    #                   + (l2_weight * l2_loss * L2_OPT)

    return total_loss


def generator_loss(disc_generated_output, gen_output, target, l2_weight = 100):
    """
        disc_generated_output: output of Discriminator when input is from Generator
        gen_output:  output of Generator (i.e., estimated H)
        target:  target image
        l2_weight: weight of L2 loss
    """
    # GAN loss
    # gen_loss = log(DIS(GEN))
    # total_loss = total_loss + l2_loss

    # gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=tf.ones_like(disc_generated_output), logits=disc_generated_output)

    # L2 loss
    l2_loss = tf.reduce_mean(tf.abs(target - gen_output))  # loss with target...
    # total_gen_loss = tf.reduce_mean(gen_loss) + l2_weight * l2_loss  ## Type : tf.tensor()

    return cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output) + l2_weight * l2_loss

