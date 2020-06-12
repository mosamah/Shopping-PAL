import csv
import pymongo
import sys
import json
import cv2
from compute_items_colors import show_images
import numpy as np
def bgr2rgb(img):
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    img_rgb = np.copy(img)
    img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2] = r, g, b
    return img_rgb
def find_matching_pieces(piece_name):
    connection = pymongo.MongoClient("localhost",27017)
    db = connection["GP"]
    matching_rules = db.get_collection("matching_rules")
    data=db.get_collection("dataset")

    trend_rules=matching_rules.find({"rule":piece_name},{"_id":0})
    trend_rules=list(trend_rules)
    # print(trend_rules)
    maxlen=4
    cnt=0
    if(len(trend_rules)==0):
        trend_imgs=data.find({"labels":piece_name},{"_id":0})
        trend_imgs=list(trend_imgs)

        showed_imgs=[]
        captions=[]
        for doc in trend_imgs:
            if cnt ==maxlen:
                break
            img=cv2.imread(doc['path'],cv2.IMREAD_COLOR)
            showed_imgs.append(bgr2rgb(img))
            labels=doc['labels']
            labels=' '.join(labels)
            captions.append(labels)
            cnt=cnt+1

    else:
        trend_imgs=[]
        showed_imgs=[]
        captions=[]
        for doc in trend_rules:
            # print("hi")
            rule=list(doc['rule'])
            # print(rule)
            imgs=data.find({"labels":{"$all":rule}},{"_id":0})
            imgs=list(imgs)
            for doc in imgs:
                if cnt== maxlen:
                    break
                img=cv2.imread(doc['path'],cv2.IMREAD_COLOR)
                showed_imgs.append(bgr2rgb(img))
                labels=rule
                labels='->'.join(labels)
                captions.append(labels)
                cnt=cnt+1
            trend_imgs=trend_imgs+imgs
            # print(trend_imgs)


    show_images(showed_imgs,captions)
    print(json.dumps(trend_imgs))

piece=sys.argv[1]
# piece='white-shorts'
# piece='silver-T-shirt'
# piece="silver-T-shirt"
# print(piece)
# piece=str(piece)
#
# piece=piece.replace(' ','')
# piece=piece.replace('\"','')
# piece=piece.replace("\'",'')

# piece='\"'+piece+'\"'
# print(piece)
# piece=piece.replace('')
find_matching_pieces(piece)
