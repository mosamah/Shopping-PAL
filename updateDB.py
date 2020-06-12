# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html


#class OurfirstscraperPipeline(object):
#    def process_item(self, item, spider):
#        return item


# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


import pymongo

def updateDB(coll, caption, vector):

    connection = pymongo.MongoClient("localhost",27017)
    db = connection["gpDatabase"]
    x = db.get_collection(coll)
    title = caption
    new = {"Vector": vector}
    x.update_one({"Title":title}, {"$set": new}, upsert=False)
    
    
updateDB("jumia", "xyz", "vector1")

        

  
        
