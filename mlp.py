#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 23:44:36 2023

@author: chenguanchen
"""

import pandas as pd
df_Train = pd.read_csv("/Users/chenguanchen/dm/heart _Train.csv")

features = list(df_Train.columns[:13])
X_train = df_Train[features]
y_train = df_Train["target"]
#print(y_train) 

#建立模型
#package
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,3),random_state=1)

#訓練
model.fit(X_train,y_train)

#預測，評估模型好壞;使用訓練資料當測試資料
pred = model.predict(X_train)

#輸出混亂矩陣，顯示準確率
from sklearn.metrics import confusion_matrix,classification_report
print("輸出混亂矩陣，顯示準確率:使用訓練資料")
print(confusion_matrix(y_train,pred))
print(classification_report(y_train,pred))


#預測，評估模型好壞;使用測試資料
df_Test = pd.read_csv("/Users/chenguanchen/dm/heart_Test.csv")
features = list(df_Test.columns[:13])
X_test = df_Test[features]
y_test = df_Test["target"]

#預測，評估模型好壞;使用訓練資料當測試資料
pred = model.predict(X_test)
prob = model.predict_proba(X_test)

print("輸出混亂矩陣，顯示準確度:使用測試資料")
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

#將預測結果與真實資料 合併成DataFrame
from pandas import DataFrame
#將predict_Test轉換成 dataframe
df_PredProb = DataFrame(model.predict_proba(X_test))
#print(df_myResult)

# Compute ROC curve and ROC area for each class
#(y_test,pred)必須是[0,1] 轉換； N->0, Y->1
df = DataFrame({"Pred" : pred , "Real" : y_test})
df["PredNew"] = 0
df.loc[df["Pred"] == "Y" ,["PredNew"]] = 1
df["RealNew"] = 0
df.loc[df["Real"] == "Y" ,["RealNew"]] = 1
#使用轉換後的資料
y_test = df["RealNew"]
pred = df["PredNew"]

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