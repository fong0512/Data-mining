#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:33:50 2023

@author: chenguanchen
"""
import pandas as pd 
df_Train=pd.read_csv("/Users/chenguanchen/Downloads/Parkinsons.csv")

features=list(df_Train.columns[:22])
df_X=df_Train[features]
df_y=df_Train["status"]

from sklearn.model_selection import train_test_split
X,y=df_X.values,df_y.values

test_size=0.3
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size)
print("test_size=",test_size)

#build model
from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,random_state=0)

#Bagging
from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier(base_estimator=model,n_estimators=10,
bootstrap=True,bootstrap_features=True,max_features=3,max_samples=0.7)

#Train
model.fit(X_train,y_train)
#predict use train data to test
pred=model.predict(X_train)

#dataFrame
from pandas import DataFrame
pred_df=DataFrame(pred)
#predict end to int
pred=round(pred_df,0)
print(pred)

#out put matrix display accuracy
from sklearn.metrics import confusion_matrix,classification_report
print("out put matrix display accuracy using train data")
print(classification_report(y_train, pred))

#predict use test data
pred=model.predict(X_test)

pred_df=DataFrame(pred)

#predict end to int
pred=round(pred_df,0)
print(pred)

print("out put matrix display accuracy using test data")
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred))

#compute ROC curve and ROC area for each class
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

fpr,tpr,threshold = roc_curve(y_test, pred)###計算真正率和假正率
roc_auc=auc(fpr,tpr)###計算auc的值
plt.figure()
lw=2
plt.plot(fpr,tpr,color='darkorange',
lw=lw,label='ROC curve (area=%0.4f) ' % roc_auc)###假正率為橫坐標 真正率為縱坐標 做曲線
plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend()
plt.show()
print("ROC_auc area=%.4f"% (roc_auc))

from sklearn.metrics import precision_recall_curve
lr_precision,lr_recall,_=precision_recall_curve(y_test, pred)

from sklearn.metrics import f1_score
lr_fl,lr_auc=f1_score(y_test,pred),auc(lr_recall,lr_precision)
print("ROC_auc area=%.4f"% (lr_auc))

no_skill=len(y_test[y_test==1])/len(y_test)

plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
plt.plot(lr_recall,lr_precision,color='darkorange',
lw=lw,label='ROC curve (area=%0.4f) ' % lr_auc)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRC Curve')
