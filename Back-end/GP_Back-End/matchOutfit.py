# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:55:46 2019

@author: yousra
"""
import sys
from pymongo import MongoClient
import json

#imagePath = 'C:/Users/yousra/GP_Photos_Multer/Dresses/Dresses2.jpg'
imageLabel = sys.argv[1]
client = MongoClient ('localhost', 27017)
db = client['GP_Database']
jacketsCollection = db['Jackets']
jackets = []
for x in jacketsCollection.find({},{ "_id": 0}):
  jackets.append(x)
  
print(json.dumps(jackets))