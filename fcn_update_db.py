import pymongo
# from fcn_predict_image import get_colored_items_list

import glob
import os
import pickle
import csv

def fill_db():
    with open('items_gt_1.pkl', 'rb') as input:
        docs = pickle.load(input)
    connection = pymongo.MongoClient("localhost",27017)
    db = connection["GP"]
    dataset_coll = db.get_collection("dataset")
    print(len(docs))
    cnt=1
    for doc in docs:
        print(cnt)
        print(doc)
        cnt=cnt+1
        print(doc['path'])
        doc['path']=doc['path'].replace('gdrive/My Drive','I:/4th year/2nd sem/gp')
        doc['path']=doc['path'].replace('/','\\')
        path=doc['path']
        print(path)
        new = {"sds": doc['sds']}
        dataset_coll.update_one({"path":path}, {"$set": new}, upsert=False)

        new = {"labels": doc['labels']}
        dataset_coll.update_one({"path":path}, {"$set": new}, upsert=False)

        # dataset_coll.insert_one(doc)
    return
def produce_csv(csv_file='items_final.csv'):
    with open('items_gt_1.pkl', 'rb') as input:
        docs = pickle.load(input)

    for doc in docs:
        labels=doc['labels']
        items=' '.join(labels)
        # items=items.lower()
        print(items)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([items])


# def process(img_dir='I:\\4th year\\2nd sem\\gp\\CFPD\\image'):
#
#     #establish connection
#     connection = pymongo.MongoClient("localhost",27017)
#     db = connection["GP"]
#     dataset_coll = db.get_collection("dataset")
#
#     #fetch images in file
#     data_path = os.path.join(img_dir,'*g')
#     files = glob.glob(data_path)
#
#     cnt=1
#     for img_file in files:
#         print("cnt: ",cnt)
#         cnt=cnt+1
#         doc={}
#         doc['path']=img_file
#         doc['labels']=get_colored_items_list(img_file)
#         print(doc)
#         break
#         # dataset_coll.insert_one(doc)
#     return


fill_db()
produce_csv()