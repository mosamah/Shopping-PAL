
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
import colorsys

from enum import Enum
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

def bgr2rgb(img):
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    img_rgb = np.copy(img)
    img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2] = r, g, b
    return img_rgb

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


def compute_dist(a,b):
  dist = np.linalg.norm(a-b)
  return dist
from colour import Color


# print(Color(rgb=(230/255, 255/255, 255/255)))
colors_dict={'WHITE':(255, 255, 255),
            'SILVER':(192, 192, 192),
            'GRAY'	:(128, 128, 128),
            'BLACK'	:(0, 0, 0),
            'RED'	:(255, 0, 0),
            'MAROON':(128, 0, 0),
            'YELLOW':(255, 255, 0),
            'OLIVE':(128, 128, 0),
            'LIME':(0, 255, 0),
            'GREEN':(0, 128, 0),
            'AQUA':(0, 255, 255),
            'TEAL':(0, 128, 128),
            'BLUE':(0, 0, 255),
            'NAVY':(0, 0, 128),
            'FUCHSIA':(255, 0, 255),
            'PURPLE':(128, 0, 128)}

# img=cv2.imread('col.jpeg',cv2.IMREAD_COLOR)
# # img=bgr2rgb(img)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# show_images([img])
# print(img)
# h,s,v = img[:, :, 0], img[:, :, 1], img[:, :, 2]
# tst=np.array([h[90,450],s[90,450],v[90,450]])
# print(tst)
#
# min=55555
#
# for c in colors_dict:
#     (r,g,b)=colors_dict[c]
#     (h,s,v)=colorsys.rgb_to_hsv(r,g,b)
#     c_value=(h,s,v)
#     print(c_value)
#     c_value=np.array(list(c_value))
#
#     c_name=c
#     dist=compute_dist(c_value,tst)
#     if dist < min:
#         min=dist-0
#
#         cl=c_name
#
# print(cl)


import webcolors

def closest_colours(rs,gs,bs):

    colors_dis=[]
    ColorsLabels=webcolors.html4_hex_to_names.items()
    print(ColorsLabels)
    ColorsLabels=[c[1] for c in ColorsLabels]
    print(ColorsLabels)


    pixels_no=len(gs)
    for key, name in webcolors.html4_hex_to_names.items():

        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rds= (rs-r_c)**2
        gds= (gs-g_c)**2
        bds= (bs-b_c)**2

        colors_dis.append(rds+gds+bds)


    colors_dis=np.array(colors_dis)
    colors_dis=np.reshape(colors_dis,(pixels_no,16))
    # print(colors_dis.shape)
    min_colors_idx=np.argmin(colors_dis,axis=1)
    print(min_colors_idx)

    colors_idx,colors_counts=np.unique(min_colors_idx,return_counts=True)

    freqs=colors_counts/pixels_no
    print(colors_idx)
    print(freqs)
    ColorsLabels=np.array(ColorsLabels)
    colors=ColorsLabels[colors_idx]
    return colors,freqs

# def get_colour_name(requested_colour):
#     try:
#         closest_name = actual_name = closest_colour(requested_colour)
#     except ValueError:
#         closest_name = closest_colour(requested_colour)
#         actual_name = None
#     return actual_name, closest_name



def compute_color_freq(BGRImage):

    red=BGRImage[:,:,2]
    # red=red[modifiedtags==255]

    green=BGRImage[:,:,1]
    # green=green[modifiedtags==255]

    blue=BGRImage[:,:,0]
    # blue=blue[modifiedtags==255]

    colors,freqs=closest_colours(red,green,blue)

    return colors, freqs


