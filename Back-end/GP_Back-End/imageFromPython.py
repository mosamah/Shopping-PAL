# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:59:59 2019

@author: yousra
"""

import sys
import base64
from scipy.io import loadmat
import csv
import NodeAndPython_ColorModelTest
import cv2
import numpy as np
import os
### Get image
imagePath = 'C:/Users/yousra/GP_Photos_Multer/temp/0001.jpg'
#imagePath = sys.argv[1]
image = cv2.imread(imagePath, cv2.IMREAD_COLOR)

matFilePath = 'C:/Users/yousra/GP_Photos_Multer/CCP Dataset/CCP mat files/0001.mat'
#matFilePath = sys.argv[2]
matPixelLevel = loadmat(matFilePath)  #dictionary
discardedTags = [0,1,8,9,15,17,18,19,20,23,29,30,34,41,47,56,57]

labelsList = []
with open('C:/Users/yousra/GP_Photos_Multer/CCP Dataset/labels.csv', 'r') as labelsFile:
    reader = csv.reader(labelsFile)
    for row in reader:
        if row!= [] :
            labelsList.append(row[0])

### Operate on image
imageColors = NodeAndPython_ColorModelTest.getImageColors(image,matPixelLevel,labelsList,discardedTags)  
print(imageColors)


os.remove(imagePath)
  