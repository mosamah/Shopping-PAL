# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:49:33 2019

@author: yousra
"""
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

# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
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
    ColorsMat[np.logical_or(dark_gray,light_gray)] = 3
    ColorsMat[np.logical_or(dark_red,light_red)] = 4
    ColorsMat[brown] = 5
    ColorsMat[light_orange] = 6
    ColorsMat[np.logical_or(dark_yellow,light_yellow)] = 7
    ColorsMat[np.logical_or(dark_green,light_green)] = 8
    ColorsMat[np.logical_or(dark_cyan,light_cyan)] = 9
    ColorsMat[np.logical_or(dark_blue,light_blue)] = 10
    ColorsMat[np.logical_or(dark_purple,light_purple)] = 11
    ColorsMat[np.logical_or(dark_pink,light_pink)] = 12

    ColorsLabels = np.array(['null','black','white','Gray',
                             'Red','brown','Orange','Yellow',
                             'Green','Cyan','Blue','Purple',
                             'Pink'])
    
    #colors counts
    grayCount=dark_grayCount+light_grayCount
    redCount=dark_redCount+light_redCount
    yellowCount=dark_yellowCount+light_yellowCount
    greenCount=dark_greenCount+light_greenCount
    cyanCount=dark_cyanCount+light_cyanCount
    blueCount=dark_blueCount+light_blueCount
    purpleCount=dark_purpleCount+light_purpleCount
    pinkCount=dark_pinkCount+light_pinkCount
    
    ColorsRatios = np.array([0,blackCount,whiteCount,grayCount,
                             redCount,brownCount,light_orangeCount,yellowCount,
                             greenCount,cyanCount,blueCount,purpleCount,
                             pinkCount])
     
    colors = [ColorsLabels[i] for i in np.unique(ColorsMat)]
    colors.remove('null')
    frequencies = [ColorsRatios[i] for i in np.unique(ColorsMat)]
    frequencies.remove(0)

    return colors, frequencies

def getImageColors(image,matPixelLevel,labelsList,discardedTags,colorsThreshold,colors_no=1):
    imageColors = []
    ColorsFile = open("ColorsFile.csv",'w')
    writer = csv.DictWriter(ColorsFile, fieldnames=["piece", "colors"])
    tags = matPixelLevel.get('groundtruth')  #array
    modifiedtags = np.copy(tags,np.uint8)
    uniqueTags = np.unique(tags)
    for tag in uniqueTags:
        pieceLabel = labelsList[tag]
        if(tag in discardedTags):
            continue
        modifiedtags[tags == tag] = 255
        modifiedtags[tags != tag] = 0
        newSegmentedImg = np.zeros((modifiedtags.shape[0],modifiedtags.shape[1],3),np.uint8)
        newSegmentedImg[:,:,0] = modifiedtags
        newSegmentedImg[:,:,1] = modifiedtags
        newSegmentedImg[:,:,2] = modifiedtags
    
        newImage = np.bitwise_and(newSegmentedImg.astype('uint8'), image.astype('uint8'))
        pieceColors,colorsFrequencies = HSLImage(newImage,modifiedtags)

        pieceColors=np.array(pieceColors)
        colorsFrequencies=np.array(colorsFrequencies)
        
        #get max color frequencies
        max_colors_no=np.argsort(colorsFrequencies)[-colors_no:]
        pieceColors=pieceColors[max_colors_no]
        colorsFrequencies=colorsFrequencies[max_colors_no]
        
        thresholdedColors =  pieceColors[colorsFrequencies>0.1]
        thresholdedFreq = colorsFrequencies[colorsFrequencies>0.1]
        writer.writerows([{"piece": pieceLabel, "colors": thresholdedColors}])
        pieceColors = ''
        for color in thresholdedColors:
            pieceColors = color + '-' + pieceColors 
        pieceColors = pieceColors + pieceLabel
        imageColors.append(pieceColors)
        
    return imageColors





def fetchDS(img_dir,label_dir,csv_file,categories,discarded_tags,
            colors_threshold,ds_type,colors_no=2,colors_only=0):
    data_path = os.path.join(img_dir,'*g')
    imgs = glob.glob(data_path)
    print("NEWWWW############################################################################ ")
    count=0
    for f1 in imgs:
        img = cv2.imread(f1,cv2.IMREAD_COLOR)
        if(ds_type==0):
            img_name=f1.split('\\')[6].split('.')[0]
        else:
            img_name=f1.split('\\')[1].split('.')[0]
        mat_label=loadmat(label_dir+'\\'+img_name+'.mat')
        
        #call function extract colors with pieces
        items=getImageColors(img,mat_label,categories,discarded_tags,colors_threshold,colors_no)
        if(colors_only == 1):
            items=set([it.split('-')[0] for it in items])                      
        items=' '.join(items)    
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([items])
        count=count+1
        print("img no:",count)
    return 

#fetching data sets
#CFPD    
with open('I:\\4th year\\2nd sem\\gp\\CFPD\\categories.csv', 'r') as f:
  reader = csv.reader(f)
  reader_list = list(reader)
  categs=[''.join(x) for x in reader_list]
  categs=[x.lower() for x in categs]
  categs[12]=categs[12]+'s'
  categs[15]=categs[15]+'s'
  categs[20]=categs[20]+'s'
discarded_tags=[0,8,9,10,17,21]
fetchDS('I:\\4th year\\2nd sem\\gp\\CFPD\\image',
        'I:\\4th year\\2nd sem\\gp\\CFPD\\img-labels',
        'Pieces_Stats_Colors_Only.csv',categs,discarded_tags,
        colors_threshold=0.15,ds_type=0,colors_no=1,colors_only=1)

#CCP
discarded_tags = [0,1,8,9,15,17,18,19,20,23,29,30,34,41,47,56,57]
labelsList = loadmat('label_list.mat').get('label_list')[0]
categs=[''.join(x) for x in labelsList]
fetchDS('photos','annotations\pixel-level','Pieces_Stats_Colors_Only.csv'
        ,categs,discarded_tags,colors_threshold=0.15,ds_type=1,
        colors_no=1,colors_only=1)


