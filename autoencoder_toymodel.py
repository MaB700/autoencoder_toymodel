# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import time
from tensorflow.python.keras.backend import bias_add
from tqdm import tqdm
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib import pylab, mlab, style, colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd

import ipywidgets as widgets
from ipywidgets import interact, fixed, interact_manual, interactive

import random as rndx

import tensorflow as tf
#os.environ['AUTOGRAPH_VERBOSITY'] = 1
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

import wandb
from wandb.keras import WandbCallback
#wandb.init(project="autoencoder_mcbm_toy_denoise")

print(tf.__version__)

# %%
# load dataset
# header gives information of parameters
# TODO: load parameters from file header
nofEvents_train = 1000
nofEvents_test = 100
cut_range = 20.0
px_x = 48
px_y = 48

def create_orderN(y_noise, order):
    if order==1:
        kernel = np.array([
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])
    if order==2:
        kernel = np.array([
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1]])
    kernel = kernel[..., tf.newaxis, tf.newaxis]
    kernel = tf.constant(kernel, dtype=np.float32)
    y_order = tf.nn.conv2d(y_noise, kernel, strides=[1, 1, 1, 1], padding='SAME')
    y_order = tf.clip_by_value(y_order, clip_value_min=0., clip_value_max=1.)
    y_order -= y_noise
    y_order = tf.clip_by_value(y_order, clip_value_min=0., clip_value_max=1.)
    return y_order

hits_train = pd.read_csv("D:/ML_data/encoder/hits_1000.csv",header=None ,comment='#', nrows=nofEvents_train).values.astype('float32')
hits_test = pd.read_csv("D:/ML_data/encoder/hits_500.csv",header=None ,comment='#', nrows=nofEvents_test).values.astype('float32')

noise_train = pd.read_csv("D:/ML_data/encoder/noise_1000.csv",header=None ,comment='#', nrows=nofEvents_train).values.astype('float32')

noise_test = pd.read_csv("D:/ML_data/encoder/noise_500.csv",header=None ,comment='#', nrows=nofEvents_test).values.astype('float32')
# hits_train = pd.read_csv("hits_100.csv",header=None ,comment='#', nrows=100).values.astype('float32')
# hits_test = pd.read_csv("hits_10.csv",header=None ,comment='#',).values.astype('float32')
# noise_train = pd.read_csv("noise_100.csv",header=None ,comment='#', nrows=100).values.astype('float32')
# noise_test = pd.read_csv("noise_10.csv",header=None ,comment='#').values.astype('float32')

hits_train = tf.reshape(hits_train, [nofEvents_train, px_y, px_x])
hits_test = tf.reshape(hits_test, [nofEvents_test, px_y, px_x])
noise_train = tf.reshape(noise_train, [nofEvents_train, px_y, px_x])
noise_test = tf.reshape(noise_test, [nofEvents_test, px_y, px_x])

hits_train = hits_train[..., tf.newaxis]
hits_test = hits_test[..., tf.newaxis]
noise_train = noise_train[..., tf.newaxis]
noise_test = noise_test[..., tf.newaxis]

#start = time.process_time()
order1_train = create_orderN(noise_train, 1)
order2_train = create_orderN(noise_train, 2)
order2_train -= order1_train
order2_train = tf.clip_by_value(order2_train, clip_value_min=0., clip_value_max=1.)

order1_test = create_orderN(noise_test, 1)
order2_test = create_orderN(noise_test, 2)
order2_test -= order1_test
order2_test = tf.clip_by_value(order2_test, clip_value_min=0., clip_value_max=1.)

#print(time.process_time() - start) # way less than 0.1 seconds with 1.1k events

# create total event , hits + noise
hits_noise_train = tf.math.add(hits_train, noise_train)
hits_noise_test = tf.math.add(hits_test, noise_test)


hits_train = tf.concat([hits_train, noise_train, order1_train, order2_train], 3)
hits_test = tf.concat([hits_test, noise_test, order1_test, order2_test], 3)

#del noise_train, noise_test, order1_train, order2_train, order1_test, order2_test #free up momory

# %% 
# load functions for loss/metrics and plot
# only hits -> 1
def get_hit_average():
    @tf.autograph.experimental.do_not_convert
    def hit_average(data, y_pred):
        y_true = data[:,:,:,0]
        nofHits = tf.math.count_nonzero(y_true, dtype=tf.float32)
        return (K.sum(y_true*y_pred[:,:,:,0])/nofHits)
    return hit_average

# hits in range 1 (like kernel 3x3) around noise pixel -> 1
def get_hit_average_order1():
    @tf.autograph.experimental.do_not_convert
    def hit_average_order1(data, y_pred):
        y_hits_in_order1 = data[:,:,:,0]*data[:,:,:,2]
        nofHitsInOrder1 = tf.math.count_nonzero(y_hits_in_order1, dtype=tf.float32)
        return (K.sum(y_hits_in_order1*y_pred[:,:,:,0])/nofHitsInOrder1)
    return hit_average_order1

# hits in range 2 (like kernel 5x5) around noise pixel -> 1
def get_hit_average_order2():
    @tf.autograph.experimental.do_not_convert
    def hit_average_order2(data, y_pred):
        y_hits_in_order2 = data[:,:,:,0]*data[:,:,:,3]
        nofHitsInOrder2 = tf.math.count_nonzero(y_hits_in_order2, dtype=tf.float32)
        return (K.sum(y_hits_in_order2*y_pred[:,:,:,0])/nofHitsInOrder2)
    return hit_average_order2       

# only noise -> 0
def get_noise_average():
    @tf.autograph.experimental.do_not_convert
    def noise_average(data, y_pred):
        y_noise = data[:,:,:,1]
        nofNoise = tf.math.count_nonzero(y_noise, dtype=tf.float32)
        return (K.sum(y_noise*y_pred[:,:,:,0])/nofNoise)
    return noise_average

# empty pmt (no hits/noise pixels!) -> 0
def get_background_average():  
    @tf.autograph.experimental.do_not_convert
    def background_average(data, y_pred):
        y_true = data[:,:,:,0]
        y_noise = data[:,:,:,1]
        y_background = tf.clip_by_value(-y_true - y_noise + tf.constant(1.0), clip_value_min=0., clip_value_max=1.)
        nofBackground = tf.math.count_nonzero(y_background, dtype=tf.float32)
        return (K.sum(K.abs(y_background*y_pred[:,:,:,0]))/nofBackground)
    return background_average 

# custom loss function to be able to use noise_train/test in loss/metrics
# data[:,:,:,0] = hits_train/hits_test , data[:,:,:,1] = noise_train/noise_test
def get_custom_loss():
    @tf.autograph.experimental.do_not_convert
    def custom_loss(data, y_pred):
        y_true = data[:,:,:,0]
        return losses.mean_squared_error(y_true, y_pred[:,:,:,0])
    return custom_loss

def single_event_plot(data, data0, nof_pixel_X, min_X, max_X, nof_pixel_Y, min_Y, max_Y, eventNo):
    plt.figure(figsize=(20, 10))
    ax = plt.subplot(1, 2, 1)
    plt.imshow(data[eventNo], interpolation='none', extent=[min_X,max_X,min_Y,max_Y], cmap='gray')
    plt.title("denoised")
    plt.colorbar()
    #plt.gray()
    ax = plt.subplot(1, 2, 2)
    cmap = colors.ListedColormap(['black','white', 'red', 'grey'])
    bounds = [0,0.1,1.25,2.5,3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(data0[eventNo], interpolation='none', extent=[min_X,max_X,min_Y,max_Y], cmap=cmap, norm=norm)
    plt.title("original")
    #plt.colorbar()
    plt.show()
    return

# %%

ks = 5
# tf.autograph.experimental.do_not_convert
# class Denoise(Model):
#   def __init__(self):
#     super(Denoise, self).__init__()
#     self.encoder = tf.keras.Sequential([
#       layers.Input(shape=(48, 48, 1)),
#       layers.Conv2D(128, (ks, ks), activation='relu',use_bias=True, padding='same'),#]) #valid
#       layers.Conv2D(64 , (ks, ks), activation='relu',use_bias=True, padding='same')])

#     self.decoder = tf.keras.Sequential([
#       layers.Conv2DTranspose(64 , kernel_size=ks, activation='relu', use_bias=True, padding='same'),
#       layers.Conv2DTranspose(128, kernel_size=ks, activation='relu', use_bias=True, padding='same'),
#       layers.Conv2D(1, kernel_size=(ks, ks), activation='tanh', padding='same')])
  
#   def call(self, x):
#     encoded = self.encoder(x)
#     decoded = self.decoder(encoded)
#     return decoded

# autoencoder = Denoise()

# alternative encoder to track model with wandb callback
model = Sequential()
model.add(Input(shape=(48, 48, 1)))
model.add(Conv2D(128, (ks, ks), activation='relu', padding='same'))
model.add(Conv2D(64, (ks, ks), activation='relu', padding='same'))
model.add(Conv2DTranspose(64 , kernel_size=ks, activation='relu', padding='same'))
model.add(Conv2DTranspose(128 , kernel_size=ks, activation='relu', padding='same'))
model.add(Conv2D(1, kernel_size=(ks, ks), activation='tanh', padding='same'))

#opt = tf.keras.optimizers.Adadelta(lr=0.1, rho=0.95, epsilon=1e-07 )
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=get_custom_loss(), metrics=[get_hit_average(), get_noise_average(), get_background_average(), get_hit_average_order1(), get_hit_average_order2()])

model.fit(hits_noise_train, hits_train,
                epochs=20,
                batch_size=100,
                shuffle=False,
                validation_data=(hits_noise_test, hits_test),
                callbacks=[WandbCallback(log_weights=True)])


# %%
# encoded = autoencoder.encoder(hits_noise_test).numpy()
# decoded = autoencoder.decoder(encoded).numpy()
# tf.shape(decoded)
# #decoded = tf.clip_by_value(tf.squeeze(decoded[:,:,:,0], [3]), clip_value_min=0.7, clip_value_max=1.)

# # single_event_plot(decoded, 48, -20.0, 20.0, 48, -20.0, 20.0, 0)
# # single_event_plot(hits_train, 48, -20.0, 20.0, 48, -20.0, 20.0, 0)
# # single_event_plot(hits_noise_train, 48, -20.0, 20.0, 48, -20.0, 20.0, 0)


# # %%
# noise_plt = tf.math.scalar_mul(2.0, hits_test[:,:,:,1])
# original_plt = tf.math.add(hits_test[:,:,:,0], tf.squeeze(noise_plt, [3]) )

#interactive_plot = widgets.interact(single_event_plot, data=fixed(noise_train), data0=fixed(2*noise_train+3*order2_train+order1_train), nof_pixel_X=fixed(px_x), min_X=fixed(-cut_range), max_X=fixed(cut_range), nof_pixel_Y=fixed(px_y), min_Y=fixed(-cut_range), max_Y=fixed(cut_range),eventNo=(0,10-1,1))

# %%
