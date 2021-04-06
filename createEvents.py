# %%
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from tqdm import tqdm
import random as rndx
import matplotlib.pyplot as plt
from matplotlib import colors
import sklearn
from sklearn.datasets import make_circles
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import time


nofEvents = 10000

minX = -20.0
maxX = 20.0
nofPixelX = 48

minY = -20.0
maxY = 20.0
nofPixelY = 48

# %%
def create_rnd_ring(params=[0,0,0]):
    rndx.seed()
    
    xshift=rndx.uniform(minX,maxX)
    yshift=rndx.uniform(minY,maxY)
    radius=rndx.uniform(3.0,5.0)
    weight_of_overlapped_rings=rndx.uniform(0.0,1.0)
    X, Y = make_circles(noise=0.05,factor=0.1, n_samples=(16,3))
    if params[0]==0:
        for i in range(len(X)):
            X[i][0] *=radius
            X[i][0] +=xshift
            X[i][1] *=radius
            X[i][1] +=yshift
    elif params[0]!=0 and weight_of_overlapped_rings<0.5:
        xshift_overlap=params[0] + rndx.uniform(-1.8*params[2],1.8*params[2])
        yshift_overlap=params[1] + rndx.uniform(-1.8*params[2],1.8*params[2])
        radius_overlap=params[2]*rndx.uniform(0.8,1.2)
        for i in range(len(X)):
            X[i][0] *=radius_overlap
            X[i][0] +=xshift_overlap
            X[i][1] *=radius_overlap
            X[i][1] +=yshift_overlap
    else:
        for i in range(len(X)):
            X[i][0] *=radius
            X[i][0] +=xshift
            X[i][1] *=radius
            X[i][1] +=yshift
    params= [xshift,yshift,radius]
    return X, Y, params

def create_single_event():
    rndx.seed()
    nofRings=rndx.randint(1,2)
    X, Y, params = create_rnd_ring()
    for _ in range(0,nofRings-1): 
        X1, Y1, params = create_rnd_ring(params)
        firstX = np.array(X)
        secondX = np.array(X1)
        X = np.concatenate((firstX, secondX),axis=0)
        
        firstY = np.array(Y)
        secondY = np.array(Y1)
        Y = np.concatenate((firstY,secondY),axis=0)
    return X, Y 

def get_pixel_nr(x, y, nof_pixel_X, min_X, max_X, nof_pixel_Y, min_Y, max_Y):
    pixel_w = ((max_X-min_X)/nof_pixel_X)
    pixel_h = ((max_Y-min_Y)/nof_pixel_Y)
    for i in range(nof_pixel_X):
        if (x >= (min_X + i*pixel_w) and x < (min_X + (i+1)*pixel_w)):
            for j in range(nof_pixel_Y):
                if (y <= (max_Y - j*pixel_h) and y > (max_Y - (j+1)*pixel_h)):
                    return (nof_pixel_X*j)+ i + 1
    return -1

def build_pixel_hits_noise(nof_pixel_X, min_X, max_X, nof_pixel_Y, min_Y, max_Y):
    hit_count = []
    for _ in range(nof_pixel_X*nof_pixel_Y):
        hit_count.append([0,0])
    X, Y = create_single_event()
    for i in range(len(X)):
        if Y[i] == 0: # "real" points
            px_nr = get_pixel_nr(X[i][0], X[i][1], nof_pixel_X, min_X, max_X, nof_pixel_Y, min_Y, max_Y)
            #print(px_nr)
            if px_nr == -1:
                continue
            hit_count[px_nr-1][0] += 1
        elif Y[i] == 1: # noise points
            px_nr = get_pixel_nr(X[i][0], X[i][1], nof_pixel_X, min_X, max_X, nof_pixel_Y, min_Y, max_Y)
            if px_nr == -1:
                continue
            hit_count[px_nr-1][1] += 1
    #add random noise 
    rndx.seed()
    nofRndNoise = rndx.randint(1,4)
    for _ in range(nofRndNoise):
        rndPxNo = rndx.randint(0, (nof_pixel_X*nof_pixel_Y)-1)
        hit_count[rndPxNo][1] += 1
    for i in range(len(hit_count)):
        if hit_count[i][0]==0 and hit_count[i][1]==0:
            hit_count[i][0] = 0
            hit_count[i][1] = 0
            continue
        if hit_count[i][0] >= hit_count[i][1]:
            hit_count[i][0] = 1
            hit_count[i][1] = 0
            continue
        if hit_count[i][0] < hit_count[i][1]:
            hit_count[i][0] = 0
            hit_count[i][1] = 1
    hits = []
    noise = []
    for i in range(len(hit_count)):
        hits.append(hit_count[i][0])
        noise.append(hit_count[i][1])
    hits = tf.reshape(hits, [nof_pixel_Y, nof_pixel_X]) 
    noise = tf.reshape(noise, [nof_pixel_Y, nof_pixel_X]) 
    
    return hits, noise    


def create_events(nof_Events ,nof_pixel_X, min_X, max_X, nof_pixel_Y, min_Y, max_Y):
    x_hits = []
    x_noise = []
    print('Building Events ...\n')
    for i in tqdm(range(nof_Events)):
        x_h, x_n = build_pixel_hits_noise(nof_pixel_X, min_X, max_X, nof_pixel_Y, min_Y, max_Y)
        x_hits.append(x_h)
        x_noise.append(x_n)
    print('Done!\n')
    return x_hits, x_noise

def fill_filter(px_nr_x, px_nr_y, order):
    sub_filter_order1 = tf.zeros([nofPixelY, nofPixelX], dtype=tf.int32).numpy()
    sub_filter_order2 = tf.zeros([nofPixelY, nofPixelX], dtype=tf.int32).numpy()
    if order == 1:
        for j in range(3):
            if (px_nr_y+j-1) > (nofPixelY-1) or (px_nr_y+j-1) < 0:
                continue
            for i in range(3):
                if (px_nr_x+i-1) > (nofPixelX-1) or (px_nr_x+i-1) < 0:
                    continue
                sub_filter_order1[px_nr_y+j-1][px_nr_x+i-1] += 1
        return sub_filter_order1
    if order == 2:
        for j in range(5):
            if (px_nr_y+j-2) > (nofPixelY-1) or (px_nr_y+j-2) < 0:
                continue
            for i in range(5):
                if (px_nr_x+i-2) > (nofPixelX-1) or (px_nr_x+i-2) < 0:
                    continue
                sub_filter_order2[px_nr_y+j-2][px_nr_x+i-2] += 1
        return sub_filter_order2

            
def create_noise_distance_filter(noise):
    filter_order1 = tf.zeros([nofPixelY, nofPixelX], dtype=tf.int32).numpy()
    filter_order2 = tf.zeros([nofPixelY, nofPixelX], dtype=tf.int32).numpy()
    # TODO: fix runtime
    #start2 = time.process_time()
    for j in range(nofPixelY):
        for i in range(nofPixelX):
            if noise[j][i] > 0 :
                filter_order1 += fill_filter(i, j, 1)
                filter_order2 += fill_filter(i, j, 2)
    #print(time.process_time() - start2)
    filter_order1 = np.clip(filter_order1 , 0, 1)
    filter_order2 = np.clip(filter_order2 , 0, 1)
    filter_order1 -= noise
    filter_order2 -= noise + filter_order1
    return filter_order1, filter_order2

def all_events_filter(noise, nofEvents):
    filter1 = []
    filter2 = []
    print('Building Filter ...\n')
    for i in tqdm(range(nofEvents)):
        f1, f2 = create_noise_distance_filter(noise[i])
        filter1.append(f1)
        filter2.append(f2)
    print('Done!\n')
    return filter1, filter2

def single_event_plot(data, data0, data1, nof_pixel_X, min_X, max_X, nof_pixel_Y, min_Y, max_Y, eventNo):
    plt.figure(figsize=(15, 15))
    cmap = colors.ListedColormap(['white','green', 'red'])
    bounds = [1,1.1,2.25,3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(data[eventNo]+data0[eventNo]*2+data1[eventNo]*3, interpolation='none', extent=[min_X,max_X,min_Y,max_Y], cmap=cmap, norm=norm)
    plt.colorbar()
    plt.show()
    return
# %%
hits, noise = create_events(nofEvents, nofPixelX, minX, maxX, nofPixelY, minY, maxY)
order1, order2 = all_events_filter(noise, nofEvents)
#print(order1)
# %%
#single_event_plot(noise, order1, order2, nofPixelX, minX, maxX, nofPixelY, minY, maxY, 1)
# %%
print('Saving to file...\n')
hits = pd.DataFrame((tf.reshape(hits,[nofEvents,-1])).numpy())
noise = pd.DataFrame((tf.reshape(noise,[nofEvents,-1])).numpy())
order1 = pd.DataFrame((tf.reshape(order1,[nofEvents,-1])).numpy())
order2 = pd.DataFrame((tf.reshape(order2,[nofEvents,-1])).numpy())

header = '#nofEvents='+str(nofEvents)+', nofPixelX='+str(nofPixelX)+', min_X='+str(minX)+', max_X='+str(maxX)+', nofPixelY='+str(nofPixelY)+', min_Y='+str(minY)+', max_Y='+str(maxY)+'\n'
path = 'D:/ML_data/encoder/'
#path = ''
if not os.path.isdir(path):
    os.mkdir(path)
filename_hits = path + 'hits_'+str(nofEvents)+'.csv'
filename_noise = path + 'noise_'+str(nofEvents)+'.csv'
filename_order1 = path + 'order1_'+str(nofEvents)+'.csv'
filename_order2 = path + 'order2_'+str(nofEvents)+'.csv'

with open(filename_hits, 'w') as f:
    f.write(header)
hits.to_csv(filename_hits,index=False, header=None, mode='a')

with open(filename_noise, 'w') as f:
    f.write(header)
noise.to_csv(filename_noise,index=False, header=None, mode='a')

with open(filename_order1, 'w') as f:
    f.write(header)
order1.to_csv(filename_order1,index=False, header=None, mode='a')

with open(filename_order2, 'w') as f:
    f.write(header)
order2.to_csv(filename_order2,index=False, header=None, mode='a')

print('Done!')
# np.savetxt(filename_hits, tf.reshape(hits,[nofEvents,-1]), fmt='%d', header=header) 
# np.savetxt(filename_noise, tf.reshape(noise,[nofEvents,-1]), fmt='%d', header=header) 






# %%
