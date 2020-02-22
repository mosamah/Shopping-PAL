library (arulesViz)
library (ggplot2)

#clean environment 
rm(list=ls())
#set working directory
setwd("D:\\Big Data\\Project\\Final\\BDA-project-master\\BDA-Project")
dfm<-read.transactions("clothes_only.csv")

#most frequent 5 items 
itemFrequencyPlot(dfm,topN=5,col="blue")

#extracting rules using eclat algo
## Mine itemsets with minimum support of 0.001 and 5 or less items
itemsets <- eclat(dfm,parameter = list(supp = 0.001, maxlen = 5))
itemsets
## Create rules from the itemsets with minimum conf=0.03
rules <- ruleInduction(itemsets, dfm, confidence = .3)
rules
inspect(rules )

#taking rules with lift near to 1
minLiftRules<-subset(rules,lift<1.1 & lift>0.95)
minLiftRules <-sort(minLiftRules,by="lift")
inspect(minLiftRules )



#removing rules with lift less than cerain number
rules_by_lift<-subset(rules,lift>1.15)
rules_by_lift <-sort(rules_by_lift,by="lift")
inspect(rules_by_lift )

 
## calculate a single measure and add it to the quality slot
#quality(rules) <- cbind(quality(rules), 
 #                    hyperConfidence = interestMeasure(rules, measure = "hyperConfidence", 
  #                                                       transactions = dfm))

#inspect(head(rules, by = "leverage"))

## calculate several measures
m <- interestMeasure(rules_by_lift, c("conviction","rulePowerFactor","lift"), 
                     transactions = dfm)
inspect(rules)
rsub<-head(rules,n=20,by="lift")
inspect(rsub)
m
msub<-head(m,n=20,by="conviction")
msub
plot(msub$conviction)
plot(msub$rulePowerFactor)
plot(msub$lift)

#subrules <- rules[quality(rules)$confidence > 0.3]
#inspect(subrules)

#plotting rules

#scatter plot 
plot(rules_by_lift, measure = c("support", "confidence"), shading = "lift")

#With this value the color of the points represents the length (order) of the rule. This is used for two-key plots. ... If reorder is set to TRUE, then the itemsets on the x and y-axes are reordered to bring rules with similar values for the interest measure closer together and make the plot clearer.
plot(rules_by_lift, method = "two-key plot",reorder=TRUE)

# save a plot as a html page
p1<-plotly_arules(rules_by_lift, measure = c("support", "confidence"), shading = "lift")
p2<-plotly_arules(rules_by_lift, method = "two-key plot",reorder=TRUE)
htmlwidgets::saveWidget(p1, "scatterPlotPieces.html", selfcontained = FALSE)
htmlwidgets::saveWidget(p2, "TwokeyplotPieces.html", selfcontained = FALSE)
browseURL("scatterPlotPieces.html")
browseURL("TwokeyplotPieces.html")


#matrix plot 
plot(rules_by_lift, method = "matrix", measure = "lift")
p3<-plotly_arules(rules_by_lift, method = "matrix", measure = "lift")
htmlwidgets::saveWidget(p3, "matrixPlotPieces.html", selfcontained = FALSE)
browseURL("matrixPlotPieces.html")

#Grouped Matrix plot
plot(rules_by_lift, method = "grouped")

#graph for 20 rules (sorting by lift)
subrules<-head(rules,n=20,by="lift")
plot(subrules, method = "graph")
saveAsGraph(head(subrules, n = 20, by = "lift"), file = "rulesPieces.graphml")

