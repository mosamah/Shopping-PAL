from scipy.signal import medfilt

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


def get_unique_cnts(pl):
  unique_elems,cnts=np.unique(pl,return_counts=True)
  # print(unique_elems)
  # print(cnts)

  cnt_dict={}
  c=0
  for el in unique_elems:
    cnt_dict[el]=cnts[c]
    c=c+1


  cnt_dict = sorted(cnt_dict.items(), key=lambda item: item[1],reverse=True)
  # print(cnt_dict)

  labels_cnt={}
  for k,v in cnt_dict:
    labels_cnt[k]=v
  return labels_cnt


def clean_labels(predicted_labels):
  pl=np.copy(predicted_labels)
#   pl=medfilt(pl).astype('int')
  labels_cnt= get_unique_cnts(pl)

  #if dress is maximum variables
  dress_sub={}
  dress_ban=[1,4,5,6,7,11,12,13,16,18,22]

  top_sub={}
  top_ban=[1,5,22] #tshirt and blouse, sweater

  jack_sub={}
  jack_ban=[4,6] #blazer, coat

  down_sub={}
  down_ban=[11,12,13,16,18] #leggings,jeans,pants,shorts,skirt

  #dresses check
  for it in labels_cnt:
    if it in dress_ban:
      dress_sub[it]=labels_cnt[it]

  # print("dress sub: ",dress_sub)
  if dress_sub:
    max_top=max(dress_sub.items(), key=lambda item: item[1])[0]
    # print("max: ",max_top)
    if max_top == 7:
      for lab in dress_sub:
        pl[pl==lab]=7


  #tops check
  labels_cnt= get_unique_cnts(pl)
  for it in labels_cnt:
    if it in top_ban:
      top_sub[it]=labels_cnt[it]

  # print("top sub: ",top_sub)
  if top_sub:
    max_top=max(top_sub.items(), key=lambda item: item[1])[0]
    # print("max: ",max_top)
    pl[pl==7]=max_top
    for lab in top_sub:
      pl[pl==lab]=max_top


  #jackets check
  labels_cnt= get_unique_cnts(pl)
  for it in labels_cnt:
    if it in jack_ban:
      jack_sub[it]=labels_cnt[it]

  # print("jack sub: ",jack_sub)
  if jack_sub:
    max_top=max(jack_sub.items(), key=lambda item: item[1])[0]
    # print("max: ",max_top)
    for lab in jack_sub:
      pl[pl==lab]=max_top


  #down check
  labels_cnt= get_unique_cnts(pl)
  for it in labels_cnt:
    if it in down_ban:
      down_sub[it]=labels_cnt[it]

  # print("down sub: ",down_sub)
  if down_sub:
    max_top=max(down_sub.items(), key=lambda item: item[1])[0]
    # print("max: ",max_top)
    for lab in down_sub:
      pl[pl==lab]=max_top

  pl=medfilt(pl).astype('int')
  return pl
