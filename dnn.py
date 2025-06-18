#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 00:54:50 2023

@author: chenguanchen
"""

import pandas as pd
df_Train = pd.read_csv("/Users/chenguanchen/dm/heart _Train.csv")

features = list(df_Train.columns[:13])
df_X = df_Train[features]
df_y = df_Train["target"]
#print(y_train) 

from sklearn.model_selection import train_test_split
X,y = df_X.values,df_y.values

test_size=0.3
#X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)
print("test_size=",test_size)

#建立模型
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(22,input_dim=13,activation='relu'))

model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
#from keras.layers import Dropout
#model.add(Dropout(0,2))
#model.add(Dense(9,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#compile the keras model
#model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])

#訓練
model.fit(X_train,y_train)

#預測，評估模型好壞;使用訓練資料當測試資料
pred = model.predict(X_train) #pred:0~2需轉換
#pred = model.predict_classes(X_train)

#將預測結果與真實資料 合併成DataFrame
from pandas import DataFrame
pred_df = DataFrame(pred)

#將預測結果轉整數(0,1)
#pred = round(pred_df,0)
#print(pred)

from sklearn.metrics import confusion_matrix,classification_report
print("輸出混亂矩陣，顯示準確率:使用訓練資料")
print(confusion_matrix(y_train,pred))
print(classification_report(y_train,pred))

#預測，評估模型好壞;使用訓練資料當測試資料
pred = model.predict(X_test)
#prob = model.predict_proba(X_test)

pred_df =DataFrame(pred)
#將預測結果轉整數(0,1)
#pred = round(pred_df,0)
#print(pred)
print("輸出混亂矩陣，顯示準確率:使用測試資料")
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

# Compute ROC curve and ROC area for each class
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr,tpr,threshold = roc_curve(y_test,pred) ###計算真正率和假正率
roc_auc = auc(fpr,tpr) ###計算auc的值
plt.figure()
lw = 2
#plt.figure(figsize=(10,10))
plt.plot(fpr,tpr,color='darkorange',
lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)###假正率為橫座標，真正率為縱座標做曲線
plt.plot([0,1], [0,1], color='navy', lw=lw,linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

#plt.legend(loc="lower right")#標籤位置
plt.legend()
plt.show()
print("ROC_auc area=%.4f" % (roc_auc))

#lr_probs = model.predict_proba(X_test)
#print(lr_probs)
# keep probabilities for the postive outcome only
#lr_probs = lr_probs[:,1]
# predict class values
#yhat = model.predict(X_test)
from sklearn.metrics import precision_recall_curve
#lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, pred)


from sklearn.metrics import f1_score
lr_f1,lr_auc=f1_score(y_test,pred),auc(lr_recall,lr_precision)
#summarize scores
#print('MLP(ANN):f1=%.3f PRC_auc area=%.3d'%(lr_f1,lr_arc))#f1乃是label=1的f1的f1
print("PRC_auc area=%.4f"%(lr_auc))

# plot the precision-recall curves
no_skill = len(y_test[y_test==1])/len(y_test)

#plt.figure(figsize=(10,10))
#plt.plot([0,1],[no_slill,no_skill],linestyle='--',label='no skill')
plt.plot([0,1],[1,0],color='navy',lw=lw, linestyle='--')
plt.plot(lr_recall, lr_precision, color='darkorange',
lw=lw, label='PRC curve (area = %0.4f)' % lr_auc)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])

# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRC Curve')
#show the legend
plt.legend()
#show the plot
plt.show()