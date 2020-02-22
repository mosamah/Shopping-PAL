library(arules)
library (arulesViz)
library (ggplot2)

#clean environment 
rm(list=ls())
#set working directory
setwd("C:\\Users\\DELL\\Desktop\\Spring 2019\\Big Data\\Project\\BDA-Project")

dfm<-read.transactions("Pieces_Stats_1color.csv")
summary(dfm)

#showing most frequent items
itemFrequencyPlot(dfm,topN=5)

## Mine itemsets with minimum support of 0.001 and 5 or less items
rules<- apriori(dfm,parameter=list(supp=0.001,conf=0.3,minlen=2,maxlen=5,target="rules"))
summary(rules)
inspect(rules)

#taking rules with lift near to 1
minLiftRules<-subset(rules,lift<1.1 & lift>0.95)
minLiftRules <-sort(minLiftRules,by="lift")
inspect(minLiftRules )

#removing rules with lift less than cerain number
rules_by_lift<-subset(rules,lift>1.5)
rules_by_lift <-sort(rules_by_lift,by="lift")
inspect(rules_by_lift )
