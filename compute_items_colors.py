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

import csv
import webcolors
# Show the figures / plots inside the notebook
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec


import skimage.io as io

# Show the figures / plots inside the notebook
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

import numpy as np
import random
import matplotlib.pyplot as plt
from enum import Enum

def random_color():
    levels = range(32, 256, 32)
    return tuple(int(random.choice(levels)) for _ in range(3))



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

def BGR2RGB(img):
    RGBimg=np.zeros((img.shape[0],img.shape[1],3),np.uint8)
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]
    RGBimg[:,:,0]=R
    RGBimg[:,:,1]=G
    RGBimg[:,:,2]=B
    return RGBimg

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

    #white, beige, silver, gray, black, red, maroon, yellow, gold, orange, olive, lime, green, aqua, teal,blue, navy, fuscia, purple
    rgb_colors = np.array([[255,255,255],[245,245,220],[192, 192,192],[128,128,128],[0,0,0],[255,0,0],[128,0,0],
                           [255,255,0],[255, 265, 0],[207, 181, 59],[128,128,0],[0,255,0],[0,128,0],[0,255,255],
                           [0,128,128],[0,0,255],[0,0,128],[255,0,255],[128,0,128]])
    base_colorsHSV =np.array( [[[[0,0,1]]],[[[0.11764706, 0.1,0.96]]],[[[0,0,0.75]]],
                               [[[0,0,0.5]]], [[[0,0,0]]], [[[0,1,1]]], [[[0,1,0.5]]],
                               [[[0.16666667,1,1]]], [[[0.09803922,1,1]]], [[[0.0745098,1,1]]],
                               [[[0.16666667,1,0.5]]],[[[0.33333333,1,1]]],
                               [[[0.33333333,1,0.5]]], [[[0.5,1,1]]], [[[0.5,1,0.5]]], [[[0.66666667,1,1]]],
                               [[[0.66666667,1,0.5]]], [[[0.83333333,1,1]]],[[[0.83333333,1,0.5]]]])


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


def compute_item_colors(img,labels,discarded_tags=[0,2,3,8,9,10,14,15,17,19,20,21],t=3):
    # color_names=['white', 'silver', 'gray', 'black', 'red', 'maroon', 'yellow', 'olive', 'lime',
    # 'green', 'aqua', 'teal','blue', 'navy', 'fuscia', 'purple']

    color_names=['white', 'beige', 'silver', 'gray', 'black', 'red', 'maroon', 'yellow', 'gold',
                 'orange', 'olive', 'lime','green', 'aqua', 'teal','blue', 'navy', 'fuscia', 'purple']

    label_list=['bk','T-shirt','bag','belt','blazer','blouse','coat','dress','face','hair','hat','jeans',
                'legging','pants','scarf','shoe','shorts','skin','skirt','socks','stocking','sunglass','sweater']

    colors = returnHSV(img)

    tags_counts={}
    tags,counts=np.unique(labels,return_counts=True)
    i=0
    for t in tags:
        tags_counts[t]=counts[i]
        i=i+1
    tags=list(tags)
    for t in discarded_tags:
        if t in tags:
            del tags_counts[t]
            tags.remove(t)

    # print(tags_counts)
    process_colors=np.copy(colors)
    colored_labels=[]
    pieces_sds=[]
    for piece in tags:
        piece_name=label_list[piece]
        piece_pixels=tags_counts[piece]
        sds=np.zeros(len(color_names))
        piece_colors=process_colors[labels==piece]
        cols,freqs=np.unique(piece_colors,return_counts=True)
        freqs=freqs/piece_pixels
        col_freqs={}
        i=0
        for c in cols:
            col_freqs[c]=freqs[i]
            i=i+1
        col_freqs = sorted(col_freqs.items(), key=lambda kv: kv[1],reverse=True)
        piece_color=color_names[col_freqs[0][0]]
        colored_label=piece_color+'-'+piece_name
        colored_labels.append(colored_label)
        for i in range(len(col_freqs)):
            idx= col_freqs[i][0]
            sds[idx]=col_freqs[i][1]
        pieces_sds.append((piece_name,sds))

    return colored_labels,pieces_sds

