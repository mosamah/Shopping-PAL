
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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

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

#data set processing
import idx2numpy
import numpy as np

def process_MNIST(data_path='I:\\4th year\\2nd sem\\gp\\GP Datasets\\fashionmnist\\train-images-idx3-ubyte',
                  labels_path='I:\\4th year\\2nd sem\\gp\\GP Datasets\\fashionmnist\\train-labels-idx1-ubyte'):

    classes=10
    imsize=28
    label_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt',
    'Sneaker','Bag','Ankle boot']

    imgs = idx2numpy.convert_from_file(data_path)
    labels = idx2numpy.convert_from_file(labels_path)

    #normalize
    data=[]
    for im in imgs:
      im=np.reshape(im,(imsize,imsize,1))
      data.append(im/255.0)

    print(len(imgs),len(labels))
    # show_images([imgs[0]])
    print(labels[0])

    hot_encoded_labels=[]
    for l in labels:
      lb=np.zeros(classes)
      lb[l]=1
      hot_encoded_labels.append(lb)

    labels=hot_encoded_labels
    print(labels[0])

    data=np.asarray(data)
    labels=np.asarray(labels)

    train_data=data[:50000]
    train_labels=labels[:50000]
    print(train_data.shape)
    print(train_labels.shape)

    val_data=data[50000:]
    val_labels=labels[50000:]

    print(len(val_data))
    return train_data,train_labels,val_data,val_labels

def build_ONet(classes=10,input_shape=(28,28,1)):
  chanDim = -1
  model=Sequential()
  model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=input_shape))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(Conv2D(32, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  # second CONV => RELU => CONV => RELU => POOL layer set
  model.add(Conv2D(64, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(Conv2D(64, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  # first (and only) set of FC => RELU layers
  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  # softmax classifier
  model.add(Dense(classes))
  model.add(Activation("softmax"))

  model.summary()
  # return the constructed network architecture
  return model

#process dataset
train_data,train_labels,val_data,val_labels=process_MNIST()

#build and compile model
ONet=build_ONet(classes=10)
ONet.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])

#train model
ONet.fit(x=train_data,y=train_labels,epochs=25,validation_data=(val_data,val_labels),
         verbose=1)

#save model
ONet.save('pieces_identification.h5')


