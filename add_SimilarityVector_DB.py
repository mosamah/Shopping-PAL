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

from scrapy.conf import settings
from scrapy.exceptions import DropItem
from scrapy import log


class OurfirstscraperPipeline(object):
#############
    def __init__(self):
        connection = pymongo.MongoClient(
            settings['MONGODB_SERVER'],
            settings['MONGODB_PORT']
        )
        db = connection[settings['MONGODB_DB']]
        self.collection = db[settings['MONGODB_COLLECTION']]
        

        
    def process_item(self, item, spider):
        #print("zehe2t", spider.name)
        for data in item:
            if not data:
                raise DropItem("Missing data!")
        
        #self.collection.update({'title': item['title']},dict(item), upsert=True)
        #TRY THIS ONE
        #collection.update({}, {"$set" : {"Date":datetime.strptime('Date', '%b %d, %Y').date()}})
        #posts4 = db4.posts
        #post_id4 = posts4.update({'id' : usr.get('id')}, dict4, upsert = True)
        #spider.settings.get('MONGODB_COLLECTION').insert(dict(item))
        spiderName = spider.name
        #print(spiderName)
        connection = pymongo.MongoClient(settings['MONGODB_SERVER'],settings['MONGODB_PORT'])
        db = connection[settings['MONGODB_DB']]
        if(spiderName == 'jumia'):
            self.collection = db[settings['JUMIA_COLLECTION']]
        if(spiderName == 'souq'):
            self.collection = db[settings['SOUQ_COLLECTION']]
        if(spiderName == 'sheinBlouses'):
            self.collection = db[settings['BLOUSES_COLLECTION']]
        if(spiderName == 'sheinSkirts'):
            self.collection = db[settings['SKIRTS_COLLECTION']]
        if(spiderName == 'sheinSweaters'):
            self.collection = db[settings['SWEATERS_COLLECTION']]
        if(spiderName == 'sheinBlazers' or spiderName == 'jumiaBlazers'):
            self.collection = db[settings['BLAZERS_COLLECTION']]
        if(spiderName == 'sheinDresses'):
            self.collection = db[settings['DRESSES_COLLECTION']]
        if(spiderName == 'sheinJackets'):
            self.collection = db[settings['JACKETS_COLLECTION']]
            #db.createCollection(settings['JACKETS_COLLECTION'], {capped:true, size:100000, max:100});
        if(spiderName == 'sheinJeans'):
            self.collection = db[settings['JEANS_COLLECTION']]
        if(spiderName == 'sheinJumpsuits'):
            self.collection = db[settings['JUMPSUITS_COLLECTION']]
        if(spiderName == 'sheinPants'):
            self.collection = db[settings['PANTS_COLLECTION']]
        if(spiderName == 'sheinShorts'):
            self.collection = db[settings['SHORTS_COLLECTION']]
        if(spiderName == 'sheinTops'):
            self.collection = db[settings['TOPS_COLLECTION']]
        #self.collection.insert(dict(item))
       # _filter = item.get('imgSrc')
        #if _filter:
        #    self.collection.update_one(_filter, dict(item), upsert=True)
        if "Image Source" in item:
            imgSrc = item.get("Image Source")
            self.collection.update_one({"imgSrc":imgSrc}, {"$set": item}, upsert=True)
        else:
            self.collection.insert(dict(item))
        log.msg("Question added to MongoDB database! ",
                level=log.DEBUG, spider=spider)
        #print("Salma", spider.name)
        return item

