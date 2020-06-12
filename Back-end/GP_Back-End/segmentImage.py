# -*- coding: utf-8 -*-
"""
Created on Sat May 25 20:48:58 2019

@author: yousra
"""
import json
import sys
imagePath = sys.argv[1]
#imagePath = 'C:/Users/yousra/GP_Photos_Multer/temp/Dresses2.jpg'
labelsArray = ['White Blouse', 'Black Jacket', 'Jeans Pants']
jsonDoc = {'path': imagePath, 'labelsList': labelsArray}
print(json.dumps(jsonDoc))