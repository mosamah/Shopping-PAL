# -*- coding: utf-8 -*-
"""
Created on Thu May 16 00:27:38 2019

@author: yousra
"""
import base64
import sys
import os

imagePath = sys.argv[1]
#imagePath = 'C:/Users/yousra/GP_Photos_Multer/temp/Dresses2.jpg'
with open(imagePath,'rb') as imageFile:
    encodedString = base64.b64encode(imageFile.read())
    print(encodedString)
    
os.remove(imagePath)