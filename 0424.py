#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:27:33 2023

@author: chenguanchen
"""

#機器學替演算法（ALP） 應用於Parkinsons資料集
import pandas as pd
df_Train = pd.read_csv("/Users/chenguanchen/Downloads/Parkinsons_Train.csv")
features = list(df_Train.columns[:22])
X_train= df_Train[features]
y_train = df_Train["status"]
#print (y Train)
#建立模型
#package
from sklearn.neural_network import MLPClassifier
model = MLPClassifier (solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,3),random_state=1)
model.fit(X_train,y_train)
#測，評估模型好壞；使用訓練資料當測試資料
pred = model.predict(X_train)
#書出混亂矩陣，顯示準確率
from sklearn. metrics import confusion_matrix,classification_report
print("輸出混亂矩陣,顯示準確率：使用訓練資料")
print (confusion_matrix(y_train,pred))
print(classification_report(y_train,pred))
#預測，評估槙型好壞；使用測試資料
df_Test = pd.read_csv("/Users/chenguanchen/Downloads/Parkinsons_Test.csv")
features = list(df_Test.columns[:22])
X_test= df_Test[features]
y_test = df_Test["status"]
#預測，評估模型好壞；使用訓練資料當測試資料
pred = model.predict(X_test)
prob=model.predict_proba(X_test)
print("輸出混亂矩陣，顯示準確率，使用測試資料")
print (confusion_matrix(y_test,pred))
print(classification_report (y_test,pred))
#將預測结果與真實資料 合饼成DataFrame
from pandas import DataFrame
df_PredResult=DataFrame ({"Pred": pred, "Real" : y_test})
 #print("DataFrame: Pred, Real") 
 #print(df PredResult)
#將預測結果與真寶資料 合併成DataFrame
from pandas import DataFrame
#將predicted Test 轉換成 dataframe
df_PredProb=DataFrame (model.predict_proba(X_test))

#print(df_PredProb)
#axis=1 水平合饼
df_myResult=pd.concat([df_PredResult, df_PredProb], axis=1) 
#print(df_myResult)
# Compute Roc curve and Roc area for each class
#(y_test，pred）必須是〔e，1〕義换； 1->日，Y->1
df=DataFrame ({"Pred": pred,"Real":y_test})
df["PredNew"]=0
df.loc[df["Pred"] == "Y", ["PredNew"]] = 1
df["RealNew"]=0
df.loc[df["Real"] == "Y", ["RealNew"]] = 1
#使用轉換後的資料
y_test=df["RealNew"] 
pred=df["PredNew"]
# Compute ROC curve and ROC area for each class
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt
fpr,tpr,threshold = roc_curve(y_test,pred)#計算真正率和假正率
roc_auc = auc(fpr,tpr)###計算auc的值
plt.figure()
lw = 2
#plt.figure(figsize=(10,10))
plt.plot(fpr,tpr,color='darkorange',
lw= lw,label='ROC curve (area = %0.4f)'% roc_auc)#假正率為横座標，真正率為縱座標做曲線
plt.plot([0,1],[0,1], color='navy', lw=lw,linestyle='--')
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05])
plt.xlabel ('False Positive Rate')
plt.ylabel ('True Positive Rate')
plt.title ('ROC Curve')
#plt.legend (1oc="1over right")#標籤位置
plt.legend()
plt.show()
print("ROC auc area-%.4f" % (roc_auc))
#keep probabilities for the positive outcome only
#Ir_probs = Ir_probsts, 11
# predict class values
#yhat =- model.predict (X_test)
from sklearn.metrics import precision_recall_curve
#Ir_precision, Ir_recall, Ir_precision, Ir_recall,
#lr_precisión_recall_curve (y_test,lr_probs)
lr_precision, lr_recall, _=precision_recall_curve (y_test, pred)
from sklearn.metrics import f1_score
lr_f1, lr_auc=f1_score(y_test, pred), auc(lr_recall,lr_precision)
# summarize scores
#print ( MP(AIN)： f1-%-3f, PRC_ auc area-%.35' $ (1r_t1， 1r_auc))＃f1 乃是1abel-l的五
print ("PRC_auc area=%.4f" % (lr_auc))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
#pIt-figure (figsize=(10,10)）
#pIt•plot( [e, 1], Ino ski11, no_ski11], 1inestyle= --1, 1abel- No ski1L)
plt.plot ([0,1], [1,0], color='navy', lw=lw,linestyle='--')
plt.plot(lr_recall,lr_precision,color='darkorange',
lw=lw,label='PRC curve (area = %0.4f)' %lr_auc)
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05]) 
#axis labels
plt.xlabel ('Recall')
plt.ylabel ('Precision')
plt.title('PRC Curve')
# show the legend 
plt.legend()
#show the plot 
plt.show
