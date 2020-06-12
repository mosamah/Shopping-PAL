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
from color_picker_HSV import compute_color_freq

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
import numpy as np
import random
import matplotlib.pyplot as plt
from enum import Enum


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


def random_color():
    levels = range(32, 256, 32)
    return tuple(int(random.choice(levels)) for _ in range(3))


def bgr2rgb(img):
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    img_rgb = np.copy(img)
    img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2] = r, g, b
    return img_rgb


def HSLImage_2 (BGRImage):
    HSLImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2HLS).astype('float')
    Hue = HSLImage[:,:,0] * 2
    Sat = HSLImage[:,:,1] / 255
    Light = HSLImage[:,:,2] / 255

    imgRows = HSLImage.shape[0]
    imgCols = HSLImage.shape[1]

    black =  (Light >= 0) & (Light < 0.125)  #0
    blackCount = np.sum(black.astype('int'))

    white = (Light >= 0.875) & (Light <= 1 )  #1
    whiteCount = np.sum(white.astype('int'))

    dark_gray = (Sat >=0) & (Sat < 0.2 ) & (Light >= 0.125) & (Light <0.5) #2
    dark_grayCount = np.sum(dark_gray.astype('int'))

    light_gray  =  (Sat >=0) & (Sat < 0.2 ) & (Light >= 0.5) & (Light < 0.875) #3
    light_grayCount = np.sum(light_gray.astype('int'))

    dark_red   = np.logical_or((Hue >= 0 ) & (Hue < 18),(Hue >=340)) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #4
    dark_redCount = np.sum(dark_red.astype('int'))

    light_red = np.logical_or((Hue >= 0 ) & (Hue < 18),(Hue >=340)) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #5
    light_redCount = np.sum(light_red.astype('int'))

    brown = (Hue >= 18 ) & (Hue < 45) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #6
    brownCount = np.sum(brown.astype('int'))

    light_orange = (Hue >= 18 ) & (Hue < 45) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #7
    light_orangeCount = np.sum(light_orange.astype('int'))

    dark_yellow = (Hue >= 45 ) & (Hue < 75) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5)  #8
    dark_yellowCount = np.sum(dark_yellow.astype('int'))

    light_yellow = (Hue >= 45 ) & (Hue < 75) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #9
    light_yellowCount = np.sum(light_yellow.astype('int'))

    dark_green = (Hue >= 75 ) & (Hue <155 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #10
    dark_greenCount = np.sum(dark_green.astype('int'))

    light_green = (Hue >= 75 ) & (Hue <155 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #11
    light_greenCount = np.sum(light_green.astype('int'))

    dark_cyan = (Hue >= 155 ) & (Hue <200 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #12
    dark_cyanCount = np.sum(dark_cyan.astype('int'))

    light_cyan = (Hue >= 155 ) & (Hue <200 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #13
    light_cyanCount = np.sum(light_cyan.astype('int'))

    dark_blue = (Hue >= 200 ) & (Hue <260 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #14
    dark_blueCount = np.sum(dark_blue.astype('int'))

    light_blue = (Hue >= 200 ) & (Hue <260 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #15
    light_blueCount = np.sum(light_blue.astype('int'))

    dark_purple = (Hue >= 260 ) & (Hue <310 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #16
    dark_purpleCount = np.sum(dark_purple.astype('int'))

    light_purple = (Hue >= 260 ) & (Hue <310 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #17
    light_purpleCount = np.sum(light_purple.astype('int'))

    dark_pink = (Hue >= 310 ) & (Hue <340 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #18
    dark_pinkCount = np.sum(dark_pink.astype('int'))

    light_pink = (Hue >= 310 ) & (Hue <340 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #19
    light_pinkCount = np.sum(light_pink.astype('int'))

    ColorsLabels = np.array(['black','white','darkGray','lightGray',
                        'darkRed','lightRed','brown','Orange',
                        'darkYellow','lightYellow',
                        'darkGreen','lightGreen',
                        'darkCyan','lightCyan',
                        'darkBlue','lightBlue',
                        'darkPurple','lightPurple',
                        'darkPink','lightPink'])

    ColorsRatios = np.array([blackCount,whiteCount,
                        dark_grayCount,light_grayCount,
                        dark_redCount,light_redCount,
                        brownCount,light_orangeCount,
                        dark_yellowCount,light_yellowCount,
                        dark_greenCount,light_greenCount,
                        dark_cyanCount,light_cyanCount,
                        dark_blueCount,light_blueCount,
                        dark_purpleCount,light_purpleCount,
                        dark_pinkCount,light_pinkCount])

    return ColorsLabels, ColorsRatios

def HSLImage (BGRImage,modifiedtags):
    HSLImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2HLS).astype('float')
    Hue = HSLImage[:,:,0] * 2
    Sat = HSLImage[:,:,1] / 255
    Light = HSLImage[:,:,2] / 255

    imgRows = HSLImage.shape[0]
    imgCols = HSLImage.shape[1]

    piece = np.zeros((imgRows,imgCols), bool)
    piece[:,:]=False
    piece[modifiedtags == 255] = True
    totalPixels = np.sum(piece.astype('int'))

    ColorsMat = np.zeros((imgRows,imgCols), np.uint8)

    black =  (Light >= 0) & (Light < 0.125)  #0
    black = black & piece
    blackCount = np.sum(black.astype('int')) / totalPixels

    white = (Light >= 0.875) & (Light <= 1 )  #1
    whiteCount = np.sum(white.astype('int')) /totalPixels

    dark_gray = (Sat >=0) & (Sat < 0.2 ) & (Light >= 0.125) & (Light <0.5) #2
    dark_grayCount = np.sum(dark_gray.astype('int')) / totalPixels

    light_gray  =  (Sat >=0) & (Sat < 0.2 ) & (Light >= 0.5) & (Light < 0.875) #3
    light_grayCount = np.sum(light_gray.astype('int')) / totalPixels

    dark_red   = np.logical_or((Hue >= 0 ) & (Hue < 18),(Hue >=340)) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #4
    dark_redCount = np.sum(dark_red.astype('int')) / totalPixels

    light_red = np.logical_or((Hue >= 0 ) & (Hue < 18),(Hue >=340)) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #5
    light_redCount = np.sum(light_red.astype('int')) / totalPixels

    brown = (Hue >= 18 ) & (Hue < 45) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #6
    brownCount = np.sum(brown.astype('int')) / totalPixels

    light_orange = (Hue >= 18 ) & (Hue < 45) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #7
    light_orangeCount = np.sum(light_orange.astype('int')) / totalPixels

    dark_yellow = (Hue >= 45 ) & (Hue < 75) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5)  #8
    dark_yellowCount = np.sum(dark_yellow.astype('int')) / totalPixels

    light_yellow = (Hue >= 45 ) & (Hue < 75) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #9
    light_yellowCount = np.sum(light_yellow.astype('int')) / totalPixels

    dark_green = (Hue >= 75 ) & (Hue <155 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #10
    dark_greenCount = np.sum(dark_green.astype('int')) / totalPixels

    light_green = (Hue >= 75 ) & (Hue <155 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #11
    light_greenCount = np.sum(light_green.astype('int')) / totalPixels

    dark_cyan = (Hue >= 155 ) & (Hue <200 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #12
    dark_cyanCount = np.sum(dark_cyan.astype('int')) / totalPixels

    light_cyan = (Hue >= 155 ) & (Hue <200 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #13
    light_cyanCount = np.sum(light_cyan.astype('int')) / totalPixels

    dark_blue = (Hue >= 200 ) & (Hue <260 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #14
    dark_blueCount = np.sum(dark_blue.astype('int')) / totalPixels

    light_blue = (Hue >= 200 ) & (Hue <260 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #15
    light_blueCount = np.sum(light_blue.astype('int')) / totalPixels

    dark_purple = (Hue >= 260 ) & (Hue <310 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #16
    dark_purpleCount = np.sum(dark_purple.astype('int')) / totalPixels

    light_purple = (Hue >= 260 ) & (Hue <310 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #17
    light_purpleCount = np.sum(light_purple.astype('int')) / totalPixels

    dark_pink = (Hue >= 310 ) & (Hue <340 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #18
    dark_pinkCount = np.sum(dark_pink.astype('int')) / totalPixels

    light_pink = (Hue >= 310 ) & (Hue <340 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #19
    light_pinkCount = np.sum(light_pink.astype('int')) / totalPixels


    ColorsMat[black] = 1
    ColorsMat[white] = 2
    ColorsMat[dark_gray] = 3
    ColorsMat[light_gray]= 4
    ColorsMat[dark_red] = 5
    ColorsMat[light_red] =6
    ColorsMat[brown] = 7
    ColorsMat[light_orange] = 8
    ColorsMat[dark_yellow] = 9
    ColorsMat[light_yellow] = 10
    ColorsMat[dark_green] = 11
    ColorsMat[light_green] = 12
    ColorsMat[dark_cyan] = 13
    ColorsMat[light_cyan] = 14
    ColorsMat[dark_blue] = 15
    ColorsMat[light_blue] = 16
    ColorsMat[dark_purple] = 17
    ColorsMat[light_purple] = 18
    ColorsMat[dark_pink] = 19
    ColorsMat[light_pink] = 20

    ColorsLabels = np.array(['null','black','white','darkGray','lightGray',
                        'darkRed','lightRed','brown','Orange',
                        'darkYellow','lightYellow',
                        'darkGreen','lightGreen',
                        'darkCyan','lightCyan',
                        'darkBlue','lightBlue',
                        'darkPurple','lightPurple',
                        'darkPink','lightPink'])

    ColorsRatios = np.array([0,blackCount,whiteCount,
                        dark_grayCount,light_grayCount,
                        dark_redCount,light_redCount,
                        brownCount,light_orangeCount,
                        dark_yellowCount,light_yellowCount,
                        dark_greenCount,light_greenCount,
                        dark_cyanCount,light_cyanCount,
                        dark_blueCount,light_blueCount,
                        dark_purpleCount,light_purpleCount,
                        dark_pinkCount,light_pinkCount])

    colors = [ColorsLabels[i] for i in np.unique(ColorsMat)]
    frequencies = [ColorsRatios[i] for i in np.unique(ColorsMat)]

    return colors, frequencies


def get_piece_color(image):
    return

name='Mock Neck Bell Sleeve Button Back Colorblock Blouse'
img_path='C:\\Users\\mosama\\PycharmProjects\\GP\\crawled_images\\Shein Images\\Shein Crawled Blouses\\'+name+'.jpg'
img= (cv2.resize(cv2.imread(img_path, cv2.IMREAD_COLOR),(224,224),
                 interpolation=cv2.INTER_NEAREST))

modified_image=np.copy(img)
modified_image=modified_image[90:130,90:130]

tag=np.copy(modified_image)
tag[:,:]=255
print(tag[0,0,0])
tag=tag[:,:,0]
print(tag)


show_images([bgr2rgb(img),bgr2rgb(modified_image),tag])
# pieceColors,colorsFrequencies=HSLImage(modified_image,tag)
pieceColors,colorsFrequencies=HSLImage_2(modified_image)
print(pieceColors)
print(colorsFrequencies)

pieceColors=np.array(pieceColors)
colorsFrequencies=np.array(colorsFrequencies)

colors_no=1
max_colors_no=np.argsort(colorsFrequencies)[-colors_no:]
pieceColors=pieceColors[max_colors_no]
colorsFrequencies=colorsFrequencies[max_colors_no]

print(pieceColors,colorsFrequencies)
