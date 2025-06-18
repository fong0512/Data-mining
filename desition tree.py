#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 23:24:18 2023

@author: chenguanchen
"""

import pandas as pd 
df_Train = pd.read_csv("/Users/chenguanchen/dm/heart _Train.csv")

features = list(df_Train.columns[:13])
X_train = df_Train[features]
y_train = df_Train["target"]
#print(y_Train)

#建立模型

from sklearn import tree 
#model = tree.DecisionTreeClassifier(splitter='random')
#指定參數
model = tree.DecisionTreeClassifier(min_samples_leaf=1000) 

#訓練
model.fit(X_train,y_train)

#預測
pred = model.predict(X_train)

#輸出混亂矩陣 顯示準確度
from sklearn.metrics import confusion_matrix , classification_report
print("輸出混亂矩陣 顯示準確度：使用訓練資料")
print(confusion_matrix(y_train,pred))
print(classification_report(y_train,pred))

#預測 評估模型好壞:使用測試資料
df_Test = pd.read_csv("/Users/chenguanchen/dm/heart_Test.csv")
features = list(df_Test.columns[:13])
X_test = df_Test[features]
y_test = df_Test["target"]

#預測 評估模型好壞:使用訓練資料當測試資料
pred = model.predict(X_test)
#prob = model.predict_proba(X_test)

print ("輸出混亂矩陣 顯示準確度:使用測試資料")
print (confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

#模型輸出 陽春圖形
#tree.plot_tree(model)

#模型輸出 精緻圖形
import graphviz
dot_data = tree.export_graphviz(
           model,#(決策樹模型)
           out_file = None,
           feature_names=features,#模型中對應標籤名稱
           filled = True,
           impurity = False,
           rounded = True
           )
import os
os.environ["PATH"] += os.pathsep + '/Users/chenguanchen/dm'

graph = graphviz.Source(dot_data)#選擇可視化的dot數據
graph.format = 'png'
graph.render('Dtree.gv')

import matplotlib.pyplot as plt #plt 用於顯示圖片
import matplotlib.image as mpimg #mpimg 用於讀取圖片

lena = mpimg.imread('Dtree.gv.png') #讀取程式碼處於同一目錄下的 lena.png
#此時 lena 就已經是np.array了 可以對她進行任意處理
lena.shape #(512,512,3)
plt.imshow(lena) #顯示圖片
plt.axis('off') #不顯示座標軸
plt.show()