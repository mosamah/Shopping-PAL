# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:56:03 2019

@author: mosama
"""
from __future__ import print_function
import plaidml.keras
plaidml.keras.install_backend()

# first lets import the useful stuff
import tensorflow as tf
import keras

# Imports components from Keras
from keras.layers import Dense
from keras.models import Sequential,Model
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Deconvolution2D, Cropping2D
from keras.layers import Input, Add, Dropout, Permute, add
from keras.losses import categorical_crossentropy
from keras import optimizers

import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


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

# loads and resizes photos into a list
def loadImagesFromFolder(folder_dir,imSize=224,imgCount=-1):
    data_path = os.path.join(folder_dir,'*g')
    files = glob.glob(data_path)
    imgs=[]
    if(imgCount == -1):
        for f1 in files:
            #read img
            imgBef = cv2.imread(f1,cv2.IMREAD_COLOR)
            img=cv2.resize(imgBef,(imSize,imSize),interpolation = cv2.INTER_NEAREST)
            imgs.append(img)
    else:
        count=0
        for f1 in files:
            if(count==imgCount):
                break
            #read img
            imgBef = cv2.imread(f1,cv2.IMREAD_COLOR)
            img=cv2.resize(imgBef,(imSize,imSize),interpolation = cv2.INTER_NEAREST)
            imgs.append(img)
            count+=1
    return imgs

#loads and resizes labels of 1004 photos
def loadLabels(folder_dir,imSize=224,imgCount=-1,nClasses=59):
    data_path = os.path.join(folder_dir,'*mat')
    files = glob.glob(data_path)
    labels=[]
    if(imgCount == -1):
        for f1 in files:
            #read img
            mat = loadmat(f1)
            label2D = mat['groundtruth']
            #img2D=np.reshape(img2D,(224,224))
            label=np.zeros((imSize,imSize,nClasses))
            label2D=cv2.resize(label2D,(imSize,imSize),interpolation = cv2.INTER_NEAREST)
            #change to N classes D 
            for i in range(imSize):
                for j in range(imSize):    
                    label[i,j,label2D[i,j]]=1
            labels.append(label)
    else:
        count=0
        for f1 in files:
            if(count==imgCount):
                break
            #read img
            mat = loadmat(f1)
            label2D = mat['groundtruth']
            #img2D=np.reshape(img2D,(224,224))
            label=np.zeros((imSize,imSize,nClasses))
            label2D=cv2.resize(label2D,(imSize,imSize),interpolation = cv2.INTER_NEAREST)
            #change to N classes D 
            for i in range(imSize):
                for j in range(imSize):    
                    label[i,j,label2D[i,j]]=1
            labels.append(label)
            count+=1
    return labels

#build FCN8
def buildFCN(model,imgSize=224,categories=21):

  #os
  model.add(Permute((1,2,3),input_shape = (imgSize,imgSize,3)))
                                  # Downsampling path #
  #1st block
  #Adding convolution layers
  model.add(Convolution2D(64,kernel_size = (3,3),padding = "same",activation = "relu",name = "block1_conv1"))
  model.add(Convolution2D(64,kernel_size = (3,3),padding = "same",activation = "relu",name = "block1_conv2"))

  #Addding max pooling layer
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name = 'block1_pool'))

  #2nd block
  #Adding convolution layers
  model.add(Convolution2D(128,kernel_size = (3,3),padding = "same",activation = "relu",name = "block2_conv1"))
  model.add(Convolution2D(128,kernel_size = (3,3),padding = "same",activation = "relu",name = "block2_conv2"))

  #Addding max pooling layer
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name = 'block2_pool'))


  #3rd block
  #Adding convolution layers
  model.add(Convolution2D(256,kernel_size = (3,3),padding = "same",activation = "relu",name = "block3_conv1"))
  model.add(Convolution2D(256,kernel_size = (3,3),padding = "same",activation = "relu",name = "block3_conv2"))
  model.add(Convolution2D(256,kernel_size = (3,3),padding = "same",activation = "relu",name = "block3_conv3"))

  #Addding max pooling layer
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name = 'block3_pool'))

  #4th block
  #Adding convolution layers
  model.add(Convolution2D(512,kernel_size = (3,3),padding = "same",activation = "relu",name = "block4_conv1"))
  model.add(Convolution2D(512,kernel_size = (3,3),padding = "same",activation = "relu",name = "block4_conv2"))
  model.add(Convolution2D(512,kernel_size = (3,3),padding = "same",activation = "relu",name = "block4_conv3"))

  #Addding max pooling layer
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name = 'block4_pool'))

  #5th block
  #Adding convolution layers
  model.add(Convolution2D(512,kernel_size = (3,3),padding = "same",activation = "relu",name = "block5_conv1"))
  model.add(Convolution2D(512,kernel_size = (3,3),padding = "same",activation = "relu",name = "block5_conv2"))
  model.add(Convolution2D(512,kernel_size = (3,3),padding = "same",activation = "relu",name = "block5_conv3"))

  #Adding max pooling layer
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name = 'block5_pool'))

  model.add(Convolution2D(4096,kernel_size=(7,7),padding = "same",activation = "relu",name = "fc6"))

  #Replacing fully connnected layers of VGG Net using convolutions
  model.add(Convolution2D(4096,kernel_size=(1,1),padding = "same",activation = "relu",name = "fc7"))

  # Gives the classifications scores for each of the N classes including background
  model.add(Convolution2D(categories,kernel_size=(1,1),padding="same",activation="relu",name = "score_fr"))

  #Save convolution size
  desiredSize=model.layers[-1].output_shape[2]
                              # Upsampling  #

  #First deconv layer
  model.add(Deconvolution2D(categories,kernel_size=(4,4),strides = (2,2),padding = "valid"))
  actualSize=model.layers[-1].output_shape[2]
  extra=actualSize-2*desiredSize

  #Cropping to get correct size
  model.add(Cropping2D(cropping=((0,extra),(0,extra))))

  Conv_size = model.layers[-1].output_shape[2]

  #Conv to be applied on Pool4
  skip_con1 = Convolution2D(categories,kernel_size=(1,1),padding = "same",activation=None, name = "score_pool4")

  #Addig skip connection which takes adds the output of Max pooling layer 4 to current layer
  Summed = add(inputs = [skip_con1(model.layers[14].output),model.layers[-1].output])

  #Upsampling output of first skip connection
  x = Deconvolution2D(categories,kernel_size=(4,4),strides = (2,2),padding = "valid",activation=None,name = "score4")(Summed)
  x = Cropping2D(cropping=((0,2),(0,2)))(x)


  #Conv to be applied to pool3
  skip_con2 = Convolution2D(categories,kernel_size=(1,1),padding = "same",activation=None, name = "score_pool3")

  #Adding skip connection which takes output og Max pooling layer 3 to current layer
  Summed = add(inputs = [skip_con2(model.layers[10].output),x])

  #Final Up convolution which restores the original image size
  Up = Deconvolution2D(categories,kernel_size=(16,16),strides = (8,8),
                       padding = "valid",activation = None,name = "upsample")(Summed)

  #Cropping the extra part obtained due to transpose convolution
  final = Cropping2D(cropping = ((0,8),(0,8)))(Up)

  return Model(model.input, final)

#test arch of FCN Main
fcn8= Sequential()

#build arch of fcn and return model
fcn8=buildFCN(fcn8,224,59)

#compile model
fcn8.compile(loss=categorical_crossentropy,
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])

print(fcn8.layers[-1].output_shape)
fcn8.summary()

#preprocessing data
imgs=loadImagesFromFolder("photos",224,10)
print(len(imgs))
labels=loadLabels("annotations\pixel-level",224,10,59)
print(len(labels))

imgs=np.array(imgs)
labels=np.array(labels)
print(imgs.shape)
print(labels.shape)

#training model
fcn8.fit(x=imgs, y=labels, batch_size=None, epochs=3, verbose=1)
