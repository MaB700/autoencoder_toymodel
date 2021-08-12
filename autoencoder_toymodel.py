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
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.models import Sequential


# load custom functions/loss/metrics
from autoencoder_functions import *


import wandb
from wandb.keras import WandbCallback
#wandb.init(project="autoencoder_mcbm_toy_denoise")

print('Tensorflow version: ' + tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT available")
# %%
# load dataset
# header gives information of parameters
# TODO: load parameters from file header
nofEvents_train = 10000
nofEvents_test = 1000
cut_range = 20.0
px_x = 48
px_y = 48

hits_train = pd.read_csv("./data/hits_10000_20210810-153832.csv",header=None ,comment='#', nrows=nofEvents_train).values.astype('float32')
hits_test = pd.read_csv("./data/hits_1000_20210810-153728.csv",header=None ,comment='#', nrows=nofEvents_test).values.astype('float32')
noise_train = pd.read_csv("./data/noise_10000_20210810-153832.csv",header=None ,comment='#', nrows=nofEvents_train).values.astype('float32')
noise_test = pd.read_csv("./data/noise_1000_20210810-153728.csv",header=None ,comment='#', nrows=nofEvents_test).values.astype('float32')


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

del noise_train, noise_test, order1_train, order2_train, order1_test, order2_test #free up momory

# %%
hidden_activation = 'relu'
custom_metrics = [get_hit_average(), get_noise_average(), get_background_average(),\
                    get_hit_average_order1(), get_hit_average_order2()]

model = Sequential()
model.add(Input(shape=(48, 48, 1)))

model.add(Conv2D(filters=32, kernel_size=5, strides=2 , activation=hidden_activation, padding='same'))
#model.add(Conv2D(filters=64, kernel_size=5,strides=2 ,activation=hidden_activation, padding='same'))
#model.add(Conv2D(filters=128, kernel_size=5,strides=2 ,activation=hidden_activation, padding='same'))
#model.add(Conv2D(filters=256, kernel_size=5,strides=2 ,activation=hidden_activation, padding='same'))

#model.add(BatchNormalization())
#model.add(Conv2DTranspose(filters=256 , kernel_size=5,strides=2 ,activation=hidden_activation, padding='same'))
#model.add(Conv2DTranspose(filters=128 , kernel_size=5,strides=2 ,activation=hidden_activation, padding='same'))
#model.add(Conv2DTranspose(filters=64 , kernel_size=5,strides=2 ,activation=hidden_activation, padding='same'))
model.add(Conv2DTranspose(filters=32 , kernel_size=5, strides=2 ,activation=hidden_activation, padding='same'))
#model.add(BatchNormalization())

model.add(Conv2D(1, kernel_size=5, activation='tanh', padding='same'))
model.summary()
#opt = tf.keras.optimizers.Adadelta(lr=10, rho=0.95, epsilon=1e-07 )
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=get_custom_loss(), metrics=custom_metrics, experimental_steps_per_execution=10)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model.fit(hits_noise_train, hits_train,
                epochs=1,
                batch_size=100,
                shuffle=True,
                validation_data=(hits_noise_test, hits_test),
                callbacks=[es])#,
                #callbacks=[WandbCallback(log_weights=True)])
#print('model evaluate ...\n')
#model.evaluate(hits_noise_test, hits_test, verbose=1)

original_plt = tf.math.add(hits_test[:,:,:,0], tf.math.scalar_mul(2.0, hits_test[:,:,:,1]) )
encoded = model.predict(hits_noise_test, batch_size=200)

# # single_event_plot(hits_noise_train, 48, -20.0, 20.0, 48, -20.0, 20.0, 0)


interactive_plot = widgets.interact(single_event_plot, \
                    data=fixed(tf.squeeze(encoded,[3])), data0=fixed(2*hits_test[:,:,:,1]+hits_test[:,:,:,0]), \
                    nof_pixel_X=fixed(px_x), min_X=fixed(-cut_range), max_X=fixed(cut_range), \
                    nof_pixel_Y=fixed(px_y), min_Y=fixed(-cut_range), max_Y=fixed(cut_range), eventNo=(40,80-1,1), cut=(0.,0.90,0.05))



# # %%
# #model.evaluate((hits_noise_test[14,:,:,:])[tf.newaxis,...], (hits_test[14,:,:,:])[tf.newaxis,...], verbose=1);
# def hit_average_order2(data, y_pred):
#     y_hits_in_order2 = data[14,:,:,0]*data[14,:,:,3]
#     nofHitsInOrder2 = tf.math.count_nonzero(tf.greater(y_hits_in_order2,0.01), dtype=tf.float32)
#     print(nofHitsInOrder2)
#     return (K.sum(y_hits_in_order2*y_pred[14,:,:,0])/nofHitsInOrder2), nofHitsInOrder2

# or2, nofhitsin02 = hit_average_order2(hits_test, encoded )
# print(or2)
# print(nofhitsin02)
# # %%
# print(tf.math.count_nonzero(tf.greater(hits_test[0,:,:,2],0.), dtype=tf.float32))
# #tf.greater(hits_test[0,:,:,2],0.5)
# #tf.print(hits_test[0,:,:,2], summarize=-1)
# #print(hits_test[0,:,:,2])
# # %%

# %%
