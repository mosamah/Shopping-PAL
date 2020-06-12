from __future__ import print_function
import plaidml.keras
plaidml.keras.install_backend()

# first lets import the useful stuff
import tensorflow as tf
import keras

## Import usual libraries
from keras.layers import Dense
from keras.models import Sequential,Model,load_model
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Deconvolution2D, Cropping2D
from keras.layers import Input, Add, Dropout, Permute, add
from keras.losses import categorical_crossentropy
from keras import optimizers

# import seaborn as sns
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
import numpy as np
import pickle
import  cv2


import scipy.io
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import glob
import os
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray, rgb2hsv

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

# preprocessing data
from scipy.io import loadmat
from scipy.io import savemat
import h5py


from enum import Enum
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import json
import sys

from skimage.color import rgb2gray,rgb2hsv
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise
import numpy as np

from skimage.exposure import histogram
from matplotlib.pyplot import bar
import cv2
from scipy.io import loadmat
from scipy.io import savemat
import pickle
from skimage import filters
from scipy import ndimage as ndi
from skimage.feature import canny

from skimage.morphology import binary_dilation,binary_erosion

def BGR2RGB(img):
    RGBimg=np.zeros((img.shape[0],img.shape[1],3),np.uint8)
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]
    RGBimg[:,:,0]=R
    RGBimg[:,:,1]=G
    RGBimg[:,:,2]=B
    return RGBimg

def bgr2rgb(img):
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    img_rgb = np.copy(img)
    img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2] = r, g, b
    return img_rgb

def returnHSV(image):
    # img = cv2.imread(image)
    img= BGR2RGB(image)
    imgRGB = img
    imgHSV = rgb2hsv(img)
    imgHSL = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype('float')
    imgHSL[:,:,0] = imgHSL[:,:,0] * 2
    imgHSL[:,:,1] = imgHSL[:,:,1] / 255
    imgHSL[:,:,2] = imgHSL[:,:,2] / 255
    #white, silver, gray, black, red, maroon, yellow, olive, lime, green, aqua, teal,blue, navy, fuscia, purple
    #rgb_colors = np.array([[1,1,1],[0.75,0.75,0.75],[0.5,0.5,0.5],[0,0,0],[1,0,0],[0.5,0,0],[1,1,0],[0.5,0.5,0],[0,1,0],[0,0.5,0],[0,1,1],[0,0.5,0.5],[0,0,1],[0,0,0.5],[1,0,1],[0.5,0,0.5]])
    rgb_colors = np.array([[255,255,255],[192, 192,192],[128,128,128],[0,0,0],[255,0,0],[128,0,0],[255,255,0],[128,128,0],[0,255,0],[0,128,0],[0,255,255],[0,128,128],[0,0,255],[0,0,128],[255,0,255],[128,0,128]])
    base_colorsHSV =np.array( [[[[0,0,1]]],[[[0,0,0.75]]], [[[0,0,0.5]]], [[[0,0,0]]], [[[0,1,1]]], [[[0,1,0.5]]], [[[0.16666667,1,1]]], [[[0.16666667,1,0.5]]],[[[0.33333333,1,1]]], [[[0.33333333,1,0.5]]], [[[0.5,1,1]]], [[[0.5,1,0.5]]], [[[0.66666667,1,1]]], [[[0.66666667,1,0.5]]], [[[0.83333333,1,1]]],[[[0.83333333,1,0.5]]]])

    #white, beige, silver, gray, black, red, maroon, yellow, gold, orange, olive, lime, green, aqua, teal,blue,
    #  navy, fuscia, purple,pink,brown
    rgb_colors = np.array([[255,255,255],[245,245,220],[192, 192,192],[128,128,128],[0,0,0],[255,0,0],[128,0,0],
                           [255,255,0],[255, 265, 0],[207, 181, 59],[128,128,0],[0,255,0],[0,128,0],[0,255,255],
                           [0,128,128],[0,0,255],[0,0,128],[255,0,255],[128,0,128]
                           ])
    base_colorsHSV =np.array( [[[[0,0,1]]],[[[0.11764706, 0.1,0.96]]],[[[0,0,0.75]]],
                               [[[0,0,0.5]]], [[[0,0,0]]], [[[0,1,1]]], [[[0,1,0.5]]],
                               [[[0.16666667,1,1]]], [[[0.09803922,1,1]]], [[[0.0745098,1,1]]],
                               [[[0.16666667,1,0.5]]],[[[0.33333333,1,1]]],
                               [[[0.33333333,1,0.5]]], [[[0.5,1,1]]], [[[0.5,1,0.5]]], [[[0.66666667,1,1]]],
                               [[[0.66666667,1,0.5]]], [[[0.83333333,1,1]]],[[[0.83333333,1,0.5]]]
                               ])


    distHSV = np.zeros((imgHSV.shape[0],imgHSV.shape[1], len(base_colorsHSV)), float)
    i = 0
    for base in base_colorsHSV:
        dh = 0.5 - np.absolute(np.absolute(imgHSV[:,:,0] - base[0,0,0]) - 0.5)
        ds = imgHSV[:,:,1] - base[0,0,1]
        dv = imgHSV[:,:,2] - base[0,0,2]
        distHSV[:,:,i] = (dh*dh*0.5) + (ds*ds*0.25) + (dv*dv*0.25)
        i += 1
    labelsHSV = np.argmin(distHSV, axis=-1)
    imageHSV = np.zeros(imgHSV.shape, np.uint8)
    imageHSV[:,:] = rgb_colors[labelsHSV]
    # show_images([imgRGB, imageHSV],["Original Image", "imageHSV"])
    return labelsHSV

def compute_piece_color(img):
    color_names=['white', 'beige', 'silver', 'gray', 'black', 'red', 'maroon', 'yellow', 'gold',
                 'orange', 'olive', 'lime','green', 'aqua', 'teal','blue', 'navy', 'fuscia', 'purple']
    # print(color_names)
    # piece=img[30:70,40:60]
    piece=img
    # show_images([bgr2rgb(piece),bgr2rgb(img)])
    piece_colors=returnHSV(piece)
    cols,freqs=np.unique(piece_colors,return_counts=True)
    pixel_sum=sum(freqs)

    # print(pixel_sum)
    col_freqs={}
    i=0
    for c in cols:
        col_freqs[c]=freqs[i]
        i=i+1
    col_freqs = sorted(col_freqs.items(), key=lambda kv: kv[1],reverse=True)
    print(col_freqs)
    bk=piece_colors[0,0]
    print("bk: ",bk)

    most_freq=col_freqs[0][1]
    if (most_freq/pixel_sum) < 0.8:
        if bk==col_freqs[0][0]:
            piece_color=color_names[col_freqs[1][0]]
        else:
            piece_color=color_names[col_freqs[0][0]]
    else:
        piece_color=color_names[col_freqs[0][0]]

    # print("piece col: ",piece_color)
    # sds=np.zeros(len(color_names))
    # for i in range(len(col_freqs)):
    #     idx= col_freqs[i][0]
    #     sds[idx]=col_freqs[i][1]
    # sds=sds/pixel_sum
    #
    # print(np.sum(sds))
    # print(sds)
    return piece_color

def show_images(images, titles=None, main_title=None,colors=None,labels=None):
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        gs = GridSpec(6,1)

        plt.imshow(image)

        if(colors is not None):

          ax2 = fig.add_subplot(gs[-1,:])
          handles = [
              Rectangle((0,0),1,1, color = tuple((v/255 for v in c))) for c in colors
          ]
          ax2.legend(handles,labels, mode='expand', ncol=3)
          ax2.axis('off')

        a.set_title(title)
        n += 1
    if main_title is not None:
        fig.suptitle(main_title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()
    return

name='Watercolor Painting Print Dip Hem Chiffon Top'
img_path='C:\\Users\\mosama\\PycharmProjects\\GP\\crawled_images\\Shein Images\\Shein Crawled Blouses\\'+name+'.jpg'

img=cv2.imread(img_path,cv2.IMREAD_COLOR)
imgrey=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)


# show_images([bgr2rgb(img)])
img= (cv2.resize(cv2.imread(img_path, cv2.IMREAD_COLOR),(96,96),
                 interpolation=cv2.INTER_NEAREST))

compute_piece_color(img)

# edges=filters.sobel(imgrey)

# edges=canny(imgrey)
# fill_coins = ndi.binary_fill_holes(edges)
# fill_coins = ndi.binary_fill_holes(fill_coins)
#
# final=binary_dilation(fill_coins)
# final=binary_erosion(final)
# t=filters.threshold_otsu(img)
# print(edges)
# mask=np.copy(img)
# mask[img>t]=255
# mask[img<t]=0
# show_images([bgr2rgb(img),mask,edges,fill_coins,final])
# print(t)

# col,sds=compute_piece_color(img)
# print(col)
# print(sds)