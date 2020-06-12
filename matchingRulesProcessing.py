# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:51:57 2019

@author: Salma Ibrahim
"""

import csv
import pymongo

connection = pymongo.MongoClient("localhost",27017)
db = connection["GP"]
x = db.get_collection("matching_rules")


with open('matching_rules.csv', 'r') as rf:
    reader = csv.reader(rf, delimiter=',')
    for col in reader:
        leftAndRight = col[1].split("=>")
        print(len(leftAndRight))
        if(len(leftAndRight) == 2):
            removeBracesL = leftAndRight[0].split("{")
            removedBracesL = removeBracesL[1].split("}")
            left = removedBracesL[0].split(",")
            #print(left)
            removeBracesR = leftAndRight[1].split("{")
            removedBracesR = removeBracesR[1].split("}")
            right = removedBracesR[0].split(",")
            print(right)
            for item in right:
                left.append(item)
                newRule = {"rule": left}
                x.insert_one(newRule)

            
            
            