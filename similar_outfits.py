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


from keras.utils import Sequence
from enum import Enum
import os
import glob
import numpy as np
import cv2
from scipy.io import loadmat
import pymongo

import json
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


def compute_dist(a,b):
  dist = np.linalg.norm(a-b)
  return dist

def compute_similarity_vector(img_path,model_path='C:\\Users\\mosama\\PycharmProjects\\GP\\VGG_similiarity_items.h5'):


    #read image
    img= (cv2.resize(cv2.imread(img_path, cv2.IMREAD_COLOR),(224,224),
                     interpolation=cv2.INTER_NEAREST))/255.0

    show_images([img],['query image'])
    #load model
    model=load_model(model_path)

    #predict sim vector
    similarity_vector= model.predict(np.array([img]), batch_size=None, verbose=0)[0]
    # print(similarity_vector)
    return similarity_vector



def getTopN_1(img_file=None,N=10):
    query_vector=compute_similarity_vector(img_file)
    connection = pymongo.MongoClient("localhost",27017)
    db = connection["GP"]
    names = db.list_collection_names()
    # print(names)
    topN=[]
    count_topN=0
    for name in names:
        # print(name)
        coll = db[name]
        # qr="{ Vector: { $exists: true} }"
        docs=coll.find({ "Vector" : { "$exists" : "false" } },{"_id":0})
        # print(docs.count())
        cnt=docs.count()
        docs=list(docs)
        docs=np.array(docs)
        for d in docs:

           sim_vector=np.array(d['Vector'])
           dist = compute_dist(query_vector,sim_vector)
           if count_topN < N :
              topN.append((d,dist,name))
              topN.sort(key=lambda x: x[1])
              count_topN=count_topN+1
           else:
              if topN[-1][1] > dist:
                del topN[-1]
                topN.append((d,dist,name))
                topN.sort(key=lambda x: x[1])

    # topN=[x[0] for x in topN ]
    return topN  


def getTopN(img_file=None,N=10):
    query_vector=compute_similarity_vector(img_file)
    connection = pymongo.MongoClient("localhost",27017)
    db = connection["GP"]
    coll = db.get_collection("dataset")
    # print(names)
    topN=[]
    count_topN=0

    docs=coll.find({},{"_id":0})
    # print(docs.count())
    cnt=docs.count()
    docs=list(docs)
    docs=np.array(docs)
    for d in docs:
       sim_vector=np.array(d['sim_vector'])
       dist = compute_dist(query_vector,sim_vector)
       if count_topN < N :
          topN.append((d,dist))
          topN.sort(key=lambda x: x[1])
          count_topN=count_topN+1
       else:
          if topN[-1][1] > dist:
            del topN[-1]
            topN.append((d,dist))
            topN.sort(key=lambda x: x[1])

    # topN=[x[0] for x in topN ]
    return topN



query_image_path=sys.argv[1]
# query_image_path='I:\\4th year\\2nd sem\\gp\\CFPD\\image\\1.jpg'
docs=getTopN(query_image_path)

# for x in docs:
#     doc=x[0]
#     title=doc['Title']
#     name=x[2]
#     name.capitalize()
#     path='C:\\Users\\mosama\\PycharmProjects\\GP\\crawled_images\\Shein Images\\Shein Crawled '+name+'\\'+title+'.jpg'
#     doc['path']=path

docs=[x[0] for x in docs ]
print(json.dumps(docs))