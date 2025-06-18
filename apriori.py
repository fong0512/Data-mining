#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:39:12 2023

@author: chenguanchen
"""

#關聯規則Apriori例
#Read Data from file
import csv
Datalist=[]
with open( "/Users/chenguanchen/dm/data.csv", newline='') as csvFile:
   rows = csv.reader(csvFile)
   for row in rows:
       #print(row)
       Datalist.append(row)
#Print data
print("===Transactions===")
print(Datalist)
from mlxtend.preprocessing import TransactionEncoder
te=TransactionEncoder()
te_ary=te.fit(Datalist).transform(Datalist)
import pandas as pd
df=pd.DataFrame(te_ary,columns=te.columns_)
#print(df)
print("===frequent itemsets===")
from mlxtend.frequent_patterns import apriori
FPs=apriori(df,min_support=0.6,use_colnames=True)
#print(FPs)
FPs['Length']=FPs['itemsets'].apply(lambda x: len(x))
print(FPs)
print("===association rule===")
from mlxtend.frequent_patterns import association_rules
ARs=association_rules(FPs,metric="confidence", min_threshold=0.9)
print(ARs)
#顯示特定欄位
print(ARs[[ "antecedents","consequents", "support", "confidence", "lift"]])
#輪出至
ARs.to_csv("/Users/chenguanchen/dm/dataAr")