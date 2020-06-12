from __future__ import print_function
import plaidml.keras
plaidml.keras.install_backend()


## Import usual libraries
from keras.layers import Dense
from keras.models import Sequential,Model,load_model
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Deconvolution2D, Cropping2D
from keras.layers import Input, Add, Dropout, Permute, add
from keras.losses import categorical_crossentropy
from keras import optimizers


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *

import cv2
from compute_items_colors import *
from fcn_clean_labels import clean_labels
from save_FCNModel import FCN8

import json
class ImgMode(Enum):
    LABEL = 1
    OHE = 2


def viewable_img(image, mode=ImgMode.LABEL, view=False):
    img = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    if mode == ImgMode.LABEL:
        img[:, :] = random_colors[image]
    elif mode == ImgMode.OHE:
        label_img = np.argmax(image, axis=2)
        img[:, :] = random_colors[label_img]
    if view:
        show_images([img])
    return img

def bgr2rgb(img):
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    img_rgb = np.copy(img)
    img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2] = r, g, b
    return img_rgb

def ohe2label(ohe):
    label_img = np.argmax(ohe, axis=-1) #K.cast(K.argmax(ohe, axis=-1), np.uint8)#
    return label_img

random_colors = np.zeros((100, 3), np.uint8)
for color_idx in range(100):
    random_colors[color_idx, :] = random_color()
    random_colors[0, :] = [255, 255, 255]


random_colors = np.zeros((100, 3), np.uint8)
for color_idx in range(100):
    random_colors[color_idx, :] = random_color()
    random_colors[0, :] = [255, 255, 255]

# print("main: ",random_colors[1])
random_square=np.zeros((10,10,3), np.uint8)
random_square=random_colors[1]
# print(random_square)


def get_colored_items_list(img_file):
    img= (cv2.resize(cv2.imread(img_file, cv2.IMREAD_COLOR),(224,224),
                     interpolation=cv2.INTER_NEAREST))
    col_img=np.copy(img)
    img=img/255.0
    label_list=labels=['bk','T-shirt','bag','belt','blazer','blouse','coat','dress','face','hair','hat','jeans','legging','pants','scarf','shoe',
    'shorts','skin','skirt','socks','stocking','sunglass','sweater']
    fcn8=load_model('C:\\Users\\mosama\\PycharmProjects\\GP\\fcn.h5')
    # fcn8 = FCN8(nClasses     = 23,
    #          input_height = 224,
    #          input_width  = 224)
    # fcn8.summary()

    # fcn8.load_weights('weights_fcn_without_overfitting.h5')

    pl = fcn8.predict(np.array([img]), batch_size=None, verbose=0)[0]

    pr = ohe2label(pl)
    predicted_labels=clean_labels(pr)

    p = viewable_img(predicted_labels, mode=ImgMode.LABEL, view=False)

    original=viewable_img(pr, mode=ImgMode.LABEL, view=False)

    show_images([bgr2rgb(img),original,p],titles=['query image','initial segmentation','after cleaning'],colors=random_colors[:23],labels=labels)


    items_list,sds=compute_item_colors(col_img,predicted_labels)

    caption=' '.join(items_list)
    show_images([bgr2rgb(img)],[caption])

    return items_list


query_image_path=sys.argv[1]
# print(query_image_path)
# query_image_path='C:\\Users\\mosama\\PycharmProjects\\GP\\GP_Photos_Multer\\temp\\1558638261309-photo.jpg'
# query_image_path='C:\\Users\\mosama\\PycharmProjects\\GP\\GP_Photos_Multer\\temp\\1558641694456-photo.jpg'
doc={}
doc['path']=query_image_path
doc['labels']=get_colored_items_list(query_image_path)
print(json.dumps(doc))
# img_name='Pearl Beading Balloon Sleeve Cardigan'
# img_file='C:\\Users\\mosama\\PycharmProjects\GP\\crawled_images\\Shein Images\\Shein Crawled Sweaters\\'+img_name+'.jpg'
# print(get_colored_items_list(img_file))



