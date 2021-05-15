import os
import numpy as np
import hdf5storage
import h5py
import matplotlib.pyplot as plt

MAT = h5py.File('../Data_Generation/Gan_Data/Gan_0_dBIndoor2p4_64ant_32users_8pilot_sparse.mat', 'r')
CH = MAT["output_da"]
CH_test = MAT["output_da_test"]

CH_re = np.squeeze(CH[0, :, :, :]).flatten()
CH_im = np.squeeze(CH[1, :, :, :]).flatten()

CH_test_re = np.squeeze(CH_test[0, :, :, :]).flatten()
CH_test_im = np.squeeze(CH_test[1, :, :, :]).flatten()

fig_CH = plt.figure(0, figsize=(10, 10))
plt.subplot(1,2,1)
plt.hist(CH_re, bins = 10000)
plt.grid(True)
plt.title("Channel Weight Distribution - Real")
plt.subplot(1,2,2)
plt.hist(CH_im, bins = 10000)
plt.grid(True)
plt.title("Channel Weight Distribution - Imaginary")
plt.show()

MAT = h5py.File('../Data_Generation/Gan_Data/Gan_0_dBIndoor2p4_64ant_32users_8pilot_scatter.mat', 'r')
CH_sp = MAT["output_da"]
CH_sp_test = MAT["output_da_test"]

CH_re = np.squeeze(CH_sp[0, :, :, :]).flatten()
CH_im = np.squeeze(CH[1, :, :, :]).flatten()

CH_test_re = np.squeeze(CH_test[0, :, :, :]).flatten()
CH_test_im = np.squeeze(CH_test[1, :, :, :]).flatten()

fig_CH = plt.figure(1, figsize=(10, 10))
plt.subplot(1,2,1)
plt.hist(CH_re, bins = 10000)
plt.grid(True)
plt.title("Channel Weight Distribution - Real")
plt.subplot(1,2,2)
plt.hist(CH_im, bins = 10000)
plt.grid(True)
plt.title("Channel Weight Distribution - Imaginary")
plt.show()