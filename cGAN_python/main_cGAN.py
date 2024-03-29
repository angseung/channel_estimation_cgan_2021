import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
from GAN.cGANGenerator import Generator
from GAN.cGANDiscriminator import Discriminator
from GAN.data_preprocess import load_image_train, load_image_test, load_image_test_y
from tempfile import TemporaryFile
from scipy.io import loadmat, savemat
import datetime
import h5py
import hdf5storage
import skfuzzy as fuzz

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
layers = tf.keras.layers


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


def discriminator_loss(disc_real_output, disc_generated_output):
    """disc_real_output = [real_target]
       disc_generated_output = [generated_target]
    """
    # log(DIS)
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(disc_real_output), logits=disc_real_output)  # label=1

    # log(1-DIS(GEN))
    generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(disc_generated_output), logits=disc_generated_output)  # label=0

    total_disc_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(generated_loss)

    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target, l2_weight=100):
    """
        disc_generated_output: output of Discriminator when input is from Generator
        gen_output:  output of Generator (i.e., estimated H)
        target:  target image
        l2_weight: weight of L2 loss
    """
    # GAN loss
    # gen_loss = log(DIS(GEN))
    # total_loss = total_loss + l2_loss
    gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(disc_generated_output), logits=disc_generated_output)

    # L2 loss
    l2_loss = tf.reduce_mean(tf.abs(target - gen_output)) # loss with target...
    total_gen_loss = tf.reduce_mean(gen_loss) + l2_weight * l2_loss ## Type : tf.tensor()
    return total_gen_loss


def generated_image(model, test_input, tar, t=0):
    """Dispaly  the results of Generator"""
    prediction = model(test_input)
    #plt.figure(figsize=(15, 15))
    display_list = [np.squeeze(test_input[:,:,:,0]), np.squeeze(tar[:,:,:,0]), np.squeeze(prediction[:,:,:,0])]
    

    title = ['Input Y', 'Target H', 'Prediction H']

    # fig_map = plt.figure(10000)
    # for i in range(3):
    #     plt.subplot(1, 3, i+1)
    #     plt.title(title[i])
    #     plt.imshow(display_list[i])
    #     plt.axis("off")
    # fig_map.savefig(os.path.join("generated_img", "img_"+str(t)+".png"))


def train_step(input_image, target, l2_weight):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image)                      # input -> generated_target
        disc_real_output = discriminator(target)  # [input, target] -> disc output
        disc_generated_output = discriminator(gen_output)  # [input, generated_target] -> disc output
        # print("*", gen_output.shape, disc_real_output.shape, disc_generated_output.shape)

        # calculate loss
        gen_loss = generator_loss(disc_generated_output, gen_output, target, l2_weight)   # gen loss
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)  # disc loss

    # gradient
    generator_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # apply gradient
    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))
    return gen_loss, disc_loss


def train(epochs, l2_weight):
    nm = []
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
            gen_loss, disc_loss = train_step(input_image, target, l2_weight)
            print("B/E:", bi, '/' , epoch, ", Generator loss:", gen_loss.numpy(), ", Discriminator loss:", disc_loss.numpy(), ', time:',  elapsed_time)
        # generated and see the progress
        for bii, (tar, inp) in enumerate(load_image_test(path)):            
            if bii == 100:
                generated_image(generator, inp, tar, t=epoch+1  )

        # save checkpoint
        # if (epoch + 1) % 2 == 0:
        ep.append(epoch + 1)
        #generator.save_weights(os.path.join(BASE_PATH, "weights/generator_"+str(epoch)+".h5"))
        #discriminator.save_weights(os.path.join(BASE_PATH, "weights/discriminator_"+str(epoch)+".h5"))
        
        realim, inpuim = load_image_test_y(path)
        prediction = generator(inpuim)

        error_ = np.sum((realim - prediction) ** 2, axis=None)
        real_ = np.sum(realim ** 2, axis=None)
        nmse_dB = 10 * np.log10(error_ / real_)
        nm.append(nmse_dB)

        # nm.append(np.log10(fuzz.nmse(np.squeeze(realim), np.squeeze(prediction))))
        
        if epoch == epochs-1:
            nmse_epoch = TemporaryFile()
            np.save(nmse_epoch, nm)
        
        # Save the predicted Channel 
        matfiledata = {} # make a dictionary to store the MAT data in
        matfiledata[u'predict_Gan_0_dB_Indoor2p4_64ant_32users_8pilot'] = np.array(prediction) # *** u prefix for variable name = unicode format, no issues thru Python 3.5; advise keeping u prefix indicator format based on feedback despite docs ***
        # hdf5storage.write(matfiledata, '.',
        #                   'Results\Eest_cGAN_'+str(epoch + 1)+'_0db_Indoor2p4_64ant_32users_8pilot.mat',
        #                   matlab_compatible=True)
        
        # plt.figure()
        # plt.plot(ep,nm,'^-r')
        # plt.xlabel('Epoch')
        # plt.ylabel('NMSE')
        # plt.show();
    
    return nm, ep

if __name__ == "__main__":
    l2_weight_list = []
    lr_gen_list = [0.002, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    beta1_list = [0.5]

    # lr_gen_list = [0.0011];
    # beta1_list = [0.8]
    epochs = 20
    fig_num = 0
    nm_list = np.zeros((len(beta1_list) * len(beta1_list), epochs + 2))
    nm_val_list = [];

    ## DATASET Option
    DATASET_CUSTUM_OPT = True
    # DATASET_CUSTUM_OPT = False

    for l2_weight in l2_weight_list:
        for beta1 in beta1_list:
            for lr_gen in lr_gen_list:

                BATCH_SIZE = 1

                # model
                generator = Generator()
                discriminator = Discriminator()

                # model summary

                # data path
                if (not DATASET_CUSTUM_OPT):
                    path = "../ Data_Generation/Gan_Data/Gan_0_dBIndoor2p4_64ant_32users_8pilot_ori.mat"
                    # optimizer
                    lr_gen = 2e-4
                    lr_dis = 2e-5
                    generator_optimizer = tf.compat.v1.train.AdamOptimizer(lr_gen, beta1=0.5)
                    discriminator_optimizer = tf.compat.v1.train.RMSPropOptimizer(lr_dis, epsilon=1e-10)

                else:
                    path = "../Data_Generation/Gan_Data/Gan_0_dBIndoor2p4_64ant_32users_8pilot_sparse.mat"
                    # optimizer
                    # lr_gen = 0.001
                    lr_dis = 2e-5
                    # beta1 = 0.9
                    generator_optimizer = tf.compat.v1.train.AdamOptimizer(lr_gen, beta1=beta1)
                    discriminator_optimizer = tf.compat.v1.train.RMSPropOptimizer(lr_dis, epsilon=1e-9)

                # train
                nm, ep = train(epochs=epochs, l2_weight=l2_weight)
                nm_val_list.append(nm)

                fig_nmse = plt.figure(fig_num)
                plt.plot(ep,nm,'^-r')

                for x, y in zip(ep, nm):
                    if (x > 9):
                        plt.text(x, y + 0.5, "%.3f" % y,  # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
                                 fontsize=9,
                                 color='black',
                                 horizontalalignment='center',  # horizontalalignment (left, center, right)
                                 verticalalignment='bottom',
                                 rotation=90)  # verticalalignment (top, center, bottom)

                plt.xlabel('Epoch')
                plt.ylabel('NMSE (dB)')
                plt.title("Epoch - NMSE Score, [lr : %.6f] [beta1 : %.3f], [l2_weight : %.3f]"
                          % (lr_gen, beta1, l2_weight))
                plt.grid(True)
                plt.show()
                fig_nmse.savefig("fig_temp/nmse_score_%05d_new" % (fig_num))

                fig_num = fig_num + 1