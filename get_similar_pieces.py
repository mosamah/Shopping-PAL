import csv
import pymongo
import sys
import json
import pickle
import numpy as np
def compute_dist_(SDSI,SDSF):
    manhattanDistance = np.sum(np.abs(SDSI - SDSF))
    # normalize manhattanDistance to range [0,1]
    manhattanDistance = (manhattanDistance - np.min(manhattanDistance))/(np.max(manhattanDistance)-np.min(manhattanDistance))
    return manhattanDistance
def compute_dist(a,b):
  dist = np.linalg.norm(a-b)
  return dist

def find_similar_pieces(piece_name,N=20):
    connection = pymongo.MongoClient("localhost",27017)
    db = connection["GP"]

    data=db.get_collection(piece_name)
    with open('C:\\Users\\mosama\\PycharmProjects\\GP\\col_temp.pkl', 'rb') as input:
        piece_color = pickle.load(input)

    # with open('C:\\Users\\mosama\\PycharmProjects\\GP\\sds_temp.pkl', 'rb') as input:
    #     piece_sds = pickle.load(input)

    #Todo zawed get top N similar zy elsimilarity

    #
    # topN=[]
    # count_topN=0
    #
    # docs=data.find({},{"_id":0})
    # # print(docs.count())
    # cnt=docs.count()
    # docs=list(docs)
    # docs=np.array(docs)
    # for d in docs:
    #    sdsn=np.array(d['sds'])
    #    dist = compute_dist(piece_sds,sdsn)
    #    if count_topN < N :
    #       topN.append((d,dist))
    #       topN.sort(key=lambda x: x[1])
    #       count_topN=count_topN+1
    #    else:
    #       if topN[-1][1] > dist:
    #         del topN[-1]
    #         topN.append((d,dist))
    #         topN.sort(key=lambda x: x[1])
    #
    # trend_rules=[x[0] for x in topN ]

    trend_rules=data.find({'color': piece_color},{"_id":0}).limit(50)
    trend_rules=list(trend_rules)

    trend_imgs=[]
    for doc in trend_rules:
        name=piece_name.capitalize()
        doc['path']='C:\\Users\\mosama\\PycharmProjects\\GP\\crawled_images\\Shein Images\\Shein Crawled '+name+'\\'+doc['Title']+'.jpg'
        trend_imgs.append(doc)
    print(json.dumps(trend_imgs))

piece=sys.argv[1]
# piece="dresses"
find_similar_pieces(piece)
