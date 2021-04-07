# %%
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import ipywidgets as widgets
from ipywidgets import fixed

import tensorflow as tf
#os.environ['AUTOGRAPH_VERBOSITY'] = 1
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential

# load custom functions/loss/metrics
from autoencoder_functions import *

import wandb
from wandb.keras import WandbCallback
#wandb.init(project="autoencoder_mcbm_toy_denoise")

print('Tensorflow version: ' + tf.__version__)

# %%
# load dataset
# header gives information of parameters
# TODO: load parameters from file header
nofEvents_train = 1000
nofEvents_test = 100
cut_range = 20.0
px_x = 48
px_y = 48

filter_seq = [128, 64, 32]

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

order1_train = create_orderN(noise_train, 1)
order2_train = create_orderN(noise_train, 2)
order2_train -= order1_train
order2_train = tf.clip_by_value(order2_train, clip_value_min=0., clip_value_max=1.)

order1_test = create_orderN(noise_test, 1)
order2_test = create_orderN(noise_test, 2)
order2_test -= order1_test
order2_test = tf.clip_by_value(order2_test, clip_value_min=0., clip_value_max=1.)

# create total event , hits + noise
hits_noise_train = tf.math.add(hits_train, noise_train)
hits_noise_test = tf.math.add(hits_test, noise_test)

hits_train = tf.concat([hits_train, noise_train, order1_train, order2_train], 3)
hits_test = tf.concat([hits_test, noise_test, order1_test, order2_test], 3)

#del noise_train, noise_test, order1_train, order2_train, order1_test, order2_test #free up momory

# %%
ks = 5

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
                validation_data=(hits_noise_test, hits_test))#,
                #callbacks=[WandbCallback(log_weights=True)])


# %%

# # single_event_plot(hits_noise_train, 48, -20.0, 20.0, 48, -20.0, 20.0, 0)

# noise_plt = tf.math.scalar_mul(2.0, hits_test[:,:,:,1])
# original_plt = tf.math.add(hits_test[:,:,:,0], tf.squeeze(noise_plt, [3]) )

#interactive_plot = widgets.interact(single_event_plot, data=fixed(noise_train), data0=fixed(2*noise_train+3*order2_train+order1_train), nof_pixel_X=fixed(px_x), min_X=fixed(-cut_range), max_X=fixed(cut_range), nof_pixel_Y=fixed(px_y), min_Y=fixed(-cut_range), max_Y=fixed(cut_range),eventNo=(0,10-1,1))


