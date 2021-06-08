import numpy as np
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from GAN.cGANGenerator import Generator, GeneratorRev, GeneratorRev2
from GAN.cGANDiscriminator import Discriminator, DiscriminatorRev
from GAN.cGANLoss import generator_loss, generator_loss_custom, discriminator_loss_custom, discriminator_loss
from GAN.data_preprocess import load_image_train, load_image_test, load_image_test_y, load_image_train_batch
from GAN.cGANLayers import make_generator_model, make_discriminator_model
from tempfile import TemporaryFile
from scipy.io import loadmat, savemat
import datetime
import h5py
import hdf5storage

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
layers = tf.keras.layers

def generated_image(model, test_input, tar, t=0):
    """Dispaly  the results of Generator"""
    prediction = model(test_input)
    # plt.figure(figsize=(15, 15))
    display_list = [np.squeeze(test_input[:, :, :, 0]), np.squeeze(tar[:, :, :, 0]), np.squeeze(prediction[:, :, :, 0])]

    title = ['Input Y', 'Target H', 'Prediction H']

    # fig_map = plt.figure(10000)
    # for i in range(3):
    #     plt.subplot(1, 3, i+1)
    #     plt.title(title[i])
    #     plt.imshow(display_list[i])
    #     plt.axis("off")
    # fig_map.savefig(os.path.join("generated_img", "img_"+str(t)+".png"))


# @tf.function
def train_step_custom(input_image, target, l2_weight):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image)  # input -> generated_target
        disc_real_output = discriminator(target)  # [input, target] -> disc output
        disc_generated_output = discriminator(gen_output)  # [input, generated_target] -> disc output
        # print("*", gen_output.shape, disc_real_output.shape, disc_generated_output.shape)

        # calculate loss
        gen_loss = generator_loss_custom(disc_generated_output, gen_output, target, l2_weight)  # gen loss
        disc_loss = discriminator_loss_custom(disc_real_output, disc_generated_output)  # disc loss

    # gradient
    generator_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # apply gradient
    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))

    return (gen_loss, disc_loss)


# @tf.function
def train_step(bi, input_image, target, l2_weight, DISC_L2_OPT = False):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image)  # input -> generated_target
        disc_real_output = discriminator(target)  # [input, target] -> disc output
        disc_generated_output = discriminator(gen_output)  # [input, generated_target] -> disc output
        # print("******** [DIC-REAL : %.8f], [DIC-GEN : %.8f]]**********"
        #       %(disc_real_output, disc_generated_output))
        # print("*", gen_output.shape, disc_real_output.shape, disc_generated_output.shape)

        # calculate loss
        gen_loss = generator_loss(disc_generated_output, gen_output, target, l2_weight)  # gen loss
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output,
                                       l2_weight = 1.0,  L2_OPT = DISC_L2_OPT)  # disc loss

    # gradient
    generator_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # apply gradient
    # if ((bi % 3) == 0):
    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))

    # if ((bi % 5) == 0):
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(epochs, l2_weight, DISC_L2_OPT, TRAIN_SHOW_OPE = False):
    nm = []
    nm_t = []
    ep = []
    start_time = datetime.datetime.now()
    print(os.getcwd())

    for epoch in range(epochs):
        print("-----\nEPOCH:", epoch)
        # train
        for bi, (target, input_image) in enumerate(load_image_train(path)):
            # bi : index of sample
            # load_image_train : yields one H and Y...
            elapsed_time = datetime.datetime.now() - start_time
            gen_loss, disc_loss = train_step(bi,
                                             input_image,
                                             target,
                                             l2_weight,
                                             DISC_L2_OPT)

            is_nan = (tf.math.is_nan(gen_loss)) or (tf.math.is_nan(disc_loss))

            if (is_nan):
                print("nan condition detected... skip this loop...")
                break

            else:
                print("B/E: %05d / %02d, Gen loss: %.8f, Dis loss: %.8f, time: %s"
                      % (bi, epoch, gen_loss.numpy().item(), disc_loss.numpy().item(),
                         elapsed_time))
                # print("B/E:", bi, '/', epoch, ", Generator loss:", gen_loss.numpy(), ", Discriminator loss:",
                #       disc_loss.numpy(), ', time:', elapsed_time)

        # generated and see the progress
        for bii, (tar, inp) in enumerate(load_image_test(path)):
            if (bii == 100):
                generated_image(generator, inp, tar, t=epoch + 1)

        # save checkpoint
        # if (epoch + 1) % 2 == 0:
        ep.append(epoch + 1)
        # generator.save_weights(os.path.join(BASE_PATH, "weights/generator_"+str(epoch)+".h5"))
        # discriminator.save_weights(os.path.join(BASE_PATH, "weights/discriminator_"+str(epoch)+".h5"))

        (realim, inpuim) = load_image_test_y(path)
        prediction = generator(inpuim)

        error_ = np.sum((realim - prediction) ** 2, axis=None)
        real_ = np.sum(realim ** 2, axis=None)
        nmse_dB = 10 * np.log10(error_ / real_)
        nm.append(nmse_dB)

        (realim_t, inpuim_t) = load_image_train_batch(path)
        prediction_t = generator(inpuim_t)

        error_t = np.sum((realim_t - prediction_t) ** 2, axis=None)
        real_t = np.sum(realim_t ** 2, axis=None)
        nmse_dB_t = 10 * np.log10(error_t / real_t)
        nm_t.append(nmse_dB_t)

        # nm.append(np.log10(fuzz.nmse(np.squeeze(realim), np.squeeze(prediction))))

        if (TRAIN_SHOW_OPE):
            fig_temp = plt.figure(epoch + 10e3)
            plt.plot(range(1, epoch + 2), nm, 'bx-', label="Test")
            plt.plot(range(1, epoch + 2), nm_t, 'go--', label="Train")
            plt.grid(True)
            plt.legend(loc = 'best')
            plt.show()


        # Save the predicted Channel
        matfiledata = {}  # make a dictionary to store the MAT data in
        matfiledata[u'predict_Gan_0_dB_Indoor2p4_64ant_32users_8pilot'] = np.array(
            prediction)  # *** u prefix for variable name = unicode format, no issues thru Python 3.5; advise keeping u prefix indicator format based on feedback despite docs ***
        # hdf5storage.write(matfiledata, '.',
        #                   'Results\Eest_cGAN_'+str(epoch + 1)+'_0db_Indoor2p4_64ant_32users_8pilot.mat',
        #                   matlab_compatible=True)

        # plt.figure()
        # plt.plot(ep,nm,'^-r')
        # plt.xlabel('Epoch')
        # plt.ylabel('NMSE')
        # plt.show();

    return (nm, nm_t, ep, is_nan)

## Main Script Start...
l2_weight_list = [0.0]
lr_gen_list = [0.001]
lr_dis_list = [1e-3, 5e-4, 1e-4, 1e-5]
beta1_list = [0.9]
# lr_gen_list = [1e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6]
# beta1_list = [0.9, 0.8, 0.7, 0.6, 0.5]
# l2_weight_list = [0.001]
# beta1_list = [0.9, 0.8, 0.7, 0.6, 0.5]

epochs = 10
fig_num = 0
nm_list = np.zeros((len(beta1_list) * len(beta1_list), epochs + 2))
nm_val_list = []
DISC_L2_OPT = False
dropout_rate = 0.0

## DATASET Option
DATASET_CUSTUM_OPT = True
# DATASET_CUSTUM_OPT = False

for l2_weight in l2_weight_list:
    for beta1 in beta1_list:
        for lr_gen in lr_gen_list:
            for lr_dis in lr_dis_list:

                BATCH_SIZE = 1

                # model
                generator = GeneratorRev2()
                discriminator = DiscriminatorRev(dropout_rate=dropout_rate)

                # generator = make_generator_model()
                # discriminator = make_discriminator_model()

                # data path
                if (not DATASET_CUSTUM_OPT):
                    path = "../ Data_Generation/Gan_Data/Gan_0_dBIndoor2p4_64ant_32users_8pilot_ori.mat"
                    # optimizer
                    lr_gen = 2e-4
                    lr_dis = 2e-5
                    generator_optimizer = tf.compat.v1.train.AdamOptimizer(lr_gen, beta1=0.5)
                    discriminator_optimizer = tf.compat.v1.train.RMSPropOptimizer(lr_dis, epsilon=1e-10)

                else:
                    path = "../Data_Generation/Gan_Data/Gan_10_dBOutdoorSCM_3path_2scatter_re_im_chan_210608_v4.mat"
                    # optimizer
                    # lr_dis = 1e-3
                    generator_optimizer = tf.compat.v1.train.AdamOptimizer(lr_gen, beta1 = beta1)
                    discriminator_optimizer = tf.compat.v1.train.RMSPropOptimizer(lr_dis, epsilon=1e-9)

                # train
                (nm, nm_t, ep, is_nan) = train(epochs = epochs,
                                               l2_weight = l2_weight,
                                               DISC_L2_OPT = DISC_L2_OPT,
                                               TRAIN_SHOW_OPE = True)

                if (is_nan):
                    print("nan detected... skip for this params...")
                    break

                nm_val_list.append(nm)

                fig_nmse = plt.figure(fig_num, figsize=(10, 10))
                plt.plot(ep, nm, '^-r', label="Test NMSE")
                plt.plot(ep, nm_t, '^--g', label="Train NMSE")

                for x, y in zip(ep, nm):
                    if (x > 9):
                        plt.text(x, y + 0.5, "%.3f" % y,  # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
                                 fontsize=9,
                                 color='black',
                                 horizontalalignment='center',  # horizontalalignment (left, center, right)
                                 verticalalignment='bottom',
                                 rotation=90)  # verticalalignment (top, center, bottom)

                timestr = time.strftime("%Y%m%d_%H%M%S")
                # plt.text(0, 0, timestr)

                plt.xlabel('Epoch')
                plt.ylabel('NMSE (dB)')
                plt.title("Epoch - NMSE Score, [lr_gen : %.8f][lr_dis : %.8f][beta1 : %.3f][l2_weight : %.8f]"
                          % (lr_gen, lr_dis, beta1, l2_weight))
                plt.grid(True)
                plt.legend(loc='best')
                # plt.show()
                # fig_nmse.savefig("fig_temp/nmse_score_%05d_%s" % (fig_num, timestr))
                fig_nmse.savefig("fig_temp/nmse_score_%s_2epoch" % (timestr))

                fig_num = fig_num + 1