#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:00:27 2023

@author: chenguanchen
"""
from apyori import apriori
#Read Data from file
import csv
Datalist=[]
with open("/Users/chenguanchen/dm/data.csv", newline='') as csvFile: 
    rows = csv.reader(csvFile)
    for row in rows:
        #print(row)
        Datalist.append(row)
#Print data
#print(Datalist)
#Run Apriori
#association_rules = apriori(Datalist,min_support=0.16, min_confidence=0.2, min
association_rules = apriori(Datalist, min_support=0.6, min_confidence=0.9, max_length=3) 
association_results = list(association_rules)
print("##frequent itemsets")
for itemList in association_results:
    pair = itemList[0]
    sup = itemList[1]
    items = [x for x in pair]
    print(items,'sup=', sup)
print("##association rule")
for itemList in association_results:
    pair = itemList[0]  
    items = [x for x in pair]
    if(len(items)>1):
     print(items)
     try:
        LstrOld=itemList[2][1][0]
        LstrNew = {x.replace('frozenset(f', '').replace(')', '') for x in LstrOld}
        RstrOld=itemList[2][1][1]
        RstrNew = {x.replace('frozenset({', '').replace('}', '') for x in RstrOld}
        print("Rule: " + str(LstrNew) + "-> " + str(RstrNew))
        print("Support: " + str(itemList[1]))
        print("Confidence: " + str(itemList[2][1][2]))
        print("Lift:" +str(itemList[2][1][3]))
        
        print()
        
        LstrOld=itemList[2][2][0]
        LstrNew = {x.replace('frozenset(f', '').replace(')', '') for x in LstrOld}
        RstrOld=itemList[2][1][1]
        RstrNew = {x.replace('frozenset({', '').replace('}', '') for x in RstrOld}
        print("Rule: " + str(LstrNew) + "-> " + str(RstrNew))
        print("Support: " + str(itemList[1]))
        print("Confidence: " + str(itemList[2][2][2]))
        print("Lift:" +str(itemList[2][2][3]))
        
        print("=======================")
     except IndexError:
        pass