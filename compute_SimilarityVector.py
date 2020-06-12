
# from __future__ import print_function
# import plaidml.keras
# plaidml.keras.install_backend()
#
# # first lets import the useful stuff
# import tensorflow as tf
# import keras
#
# ## Import usual libraries
# from keras.layers import Dense
# from keras.models import Sequential,Model,load_model
# from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Deconvolution2D, Cropping2D
# from keras.layers import Input, Add, Dropout, Permute, add
# from keras.losses import categorical_crossentropy
# from keras import optimizers
#
# # import seaborn as sns
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# import keras, sys, time, warnings
# from keras.models import *
# from keras.layers import *
# import numpy as np
# import pickle
# import  cv2
#
#
# from keras.utils import Sequence
# from enum import Enum
import os
import glob
import numpy as np
import cv2
from scipy.io import loadmat
import pymongo

# class ResizeType(Enum):
#     UNCHANGED = 0
#     SQUASH = 1
#     RESIZE_AND_CROP = 2
#     RESIZE_AND_PAD = 3
#
#
# class LabelType(Enum):
#     UNCHANGED = 0
#     BINARY = 1
#
#
# class BatchGenerator(Sequence):
#     def __init__(self,
#                  img_dir,
#                  target_dir=None,
#                  batch_size=64,
#                  label_list=None,
#                  label_type=LabelType.UNCHANGED,
#                  resize_type=ResizeType.SQUASH,
#                  img_size=(224, 224),
#                  start_point=0.0,
#                  end_point=1.0):
#         img_path = os.path.join(img_dir, '*g')
#         img_files = glob.glob(img_path)
#         img_files = list(set(img_files))
#         img_files = sorted(img_files,
#                            key=lambda item: (int(item.partition(' ')[0]) if item[0].isdigit() else float('inf'),
#                                              item.lower()))
#         if len(img_files) == 0:
#             raise Exception('empty img_dir')
#
#
#
#         self.img_files = img_files
#         self.target_files = []
#
#
#
#         self.label_type = label_type
#         self.resize_type = resize_type
#         self.img_size = img_size
#         self.batch_size = batch_size
#         self.label_list = label_list
#         print("\n", self.img_files[0])
#
#     def __len__(self):
#         return int(np.ceil(len(self.img_files) / float(self.batch_size)))
#
#     def __getitem__(self, idx):
#         x = []
#         y = []
#         batch_img_files = self.img_files[idx * self.batch_size:(idx + 1) * self.batch_size]
#         if self.resize_type == ResizeType.SQUASH:
#
#
#             for img_file in batch_img_files:
#                 im=cv2.imread(img_file, cv2.IMREAD_COLOR)
#                 if im is not None:
#                     # print(im.shape)
#                     img=(cv2.resize(im,self.img_size,
#                                 interpolation=cv2.INTER_NEAREST)) / 255.0
#                     x.append(img)
#                     title=img_file.split('\\')[1].split(".")[0]
#                     y.append(title)
#                     # print(title)
#
#
#         return np.array(x), np.array(y)
#
#
# #This file is for computing similarity vector for crawled images
# #from the database and saving them back in the database
#
# #update image info in the db
# def updateDB(coll, caption, vector):
#     connection = pymongo.MongoClient("localhost",27017)
#     db = connection["GP"]
#     x = db.get_collection(coll)
#     title = caption
#     new = {"Vector": vector}
#     x.update_one({"Title":title}, {"$set": new}, upsert=False)
#
# #load model from path
# def _load_model(model_path):
#   model=load_model(model_path)
#   return model
#
# #compute similarity vector
# def compute_similarity_vector(img_path,model_path='VGG_similiarity_items.h5'):
#     title=img_path.split('\\')[1].split(".")[0]
#
#     #read image
#     img= (cv2.resize(cv2.imread(img_path, cv2.IMREAD_COLOR),(224,224),
#                      interpolation=cv2.INTER_NEAREST))
#
#     #load model
#     model=_load_model(model_path)
#
#     #predict sim vector
#     similarity_vector= model.predict(np.array([img]), batch_size=None, verbose=1)[0]
#     print(similarity_vector)
#     updateDB('blazers',title,similarity_vector.tolist())
#     return
#
#
# def compute_similarity_vector_batch(batch,model_path='VGG_similiarity_items.h5'):
#
#     #load model
#     model=_load_model(model_path)
#
#     #predict sim vector
#     similarity_vectors= model.predict(batch, batch_size=16, verbose=1)
#     return similarity_vectors
#
# def read_images_from_file(img_dir):
#     data_path = os.path.join(img_dir,'*g')
#     files = glob.glob(data_path)
#
#     imgs_no=len(files)
#     coll=img_dir.split(' ')[-1].lower()
#     print("new folder######################## ",coll)
#
#     imgs_gen=BatchGenerator(img_dir,batch_size=16)
#
#     # for i in range(int(imgs_no/32)):
#     print(int(imgs_no/16))
#     x,y=imgs_gen.__getitem__(0)
#     print(x.shape)
#     predicts=compute_similarity_vector_batch(x)
#
#     idx=0
#     print("updating...")
#     for p in predicts:
#         print(y[idx])
#         print(p)
#         updateDB(coll,y[idx],p.tolist())
#         idx=idx+1
#
#
#     # cnt=1
#     # for f1 in files:
#     #     print(cnt)
#     #     cnt=cnt+1
#     #
#     #
#     #
#     #     #read img
#     #     sim_vector=compute_similarity_vector(f1)
#     #
#     #     #get title
#     #     title=f1.split('\\')[1].split(".")[0]
#     #     print(title)
#     #     updateDB(coll,title,sim_vector.tolist())
#
#     return
#
# def read_image_from_folder(folder_dir):
#     for folder in os.listdir(folder_dir):
#         read_images_from_file(folder_dir+'/'+folder)
#     return

# read_image_from_folder('crawled_images/Shein Images')
# read_images_from_file('crawled_images/Shein Images/Shein Crawled Blazers')

# x=compute_similarity_vector('crawled_images/Shein Images/Shein Crawled Blazers/Appliques Mesh Sleeve Buttoned Blazer.jpg')
# print(x)

from threading import Thread
from time import sleep
import subprocess


for folder in os.listdir('C:\\Users\mosama\\PycharmProjects\GP\\crawled_images\\Shein Images'):

    print("new folder##################### ",folder)
    img_dir='C:\\Users\\mosama\\PycharmProjects\GP\\crawled_images\\Shein Images\\'+folder
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)


    imgs_no=len(files)
    coll=img_dir.split(' ')[-1].lower()
    print("new folder######################## ",coll)
    banned=['pants','shorts','skirts','sweaters','Tops']
    if(coll in banned):
        continue
    for img_file in files:
        im=cv2.imread(img_file, cv2.IMREAD_COLOR)
        if im is not None:
            print(img_file)
            arg_file=img_file.replace(" ",'#')
            os.system("cd C:\\Users\\mosama\\Anaconda3\\envs\\fash && python C:\\Users\\mosama\\PycharmProjects\\GP\\sim_process.py "+arg_file+" "+coll)




