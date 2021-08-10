from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import colors
from tensorflow.keras import losses
from tensorflow.keras import backend as K


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

# only hits -> 1
def get_hit_average():
    @tf.autograph.experimental.do_not_convert
    def hit_average(data, y_pred):
        y_true = data[:,:,:,0]
        nofHits = tf.math.count_nonzero(tf.greater(y_true,0.01), dtype=tf.float32)
        return (K.sum(y_true*y_pred[:,:,:,0])/nofHits)
    return hit_average

# hits in range 1 (like kernel 3x3) around noise pixel -> 1
def get_hit_average_order1():
    @tf.autograph.experimental.do_not_convert
    def hit_average_order1(data, y_pred):
        y_hits_in_order1 = data[:,:,:,0]*data[:,:,:,2]
        nofHitsInOrder1 = tf.math.count_nonzero(tf.greater(y_hits_in_order1,0.01), dtype=tf.float32)
        return (K.sum(y_hits_in_order1*y_pred[:,:,:,0])/nofHitsInOrder1)
    return hit_average_order1

# hits in range 2 (like kernel 5x5) around noise pixel -> 1
def get_hit_average_order2():
    @tf.autograph.experimental.do_not_convert
    def hit_average_order2(data, y_pred):
        y_hits_in_order2 = data[:,:,:,0]*data[:,:,:,3]
        nofHitsInOrder2 = tf.math.count_nonzero(tf.greater(y_hits_in_order2,0.01), dtype=tf.float32)
        return (K.sum(y_hits_in_order2*y_pred[:,:,:,0])/nofHitsInOrder2)
    return hit_average_order2       

# only noise -> 0
def get_noise_average():
    @tf.autograph.experimental.do_not_convert
    def noise_average(data, y_pred):
        y_noise = data[:,:,:,1]
        nofNoise = tf.math.count_nonzero(tf.greater(y_noise,0.01), dtype=tf.float32)
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
        #return losses.binary_crossentropy(y_true, y_pred[:,:,:,0])
        return losses.mean_squared_error(y_true, y_pred[:,:,:,0])
    return custom_loss

def single_event_plot(data, data0, nof_pixel_X, min_X, max_X, nof_pixel_Y, min_Y, max_Y, eventNo, cut=0.):
    plt.figure(figsize=(20, 10))
    ax = plt.subplot(1, 2, 1)
    plt.imshow(tf.cast(data[eventNo] > cut, data[eventNo].dtype) * data[eventNo], interpolation='none', extent=[min_X,max_X,min_Y,max_Y], cmap='gray')
    plt.title("denoised")
    #y = tf.maximum(data[eventNo], 0.5)
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
