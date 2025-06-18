#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 09:35:51 2023

@author: chenguanchen
"""
import pandas as pd
df_Train = pd.read_csv("/Users/chenguanchen/Downloads/Parkinsons.csv")
features=list(df_Train.columns[:22])
df_X=df_Train[features]
df_y=df_Train["status"]

from sklearn.model_selection import train_test_split
X,y=df_X.values,df_y.values
    


test.size=0.3
#X_train,X_test,y_train,y_test=train.test.split(X,ytest_size=0.3,random_state=0)
X_train,X_test,y_train,y_test=train.test.split(X,y,test_size=test_size)
print("test_size",test_size)

#建立模型
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(22,input_dim=22,activation="relu"))

model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
#from keras.layers import Dropout
#model.add(Dropout(0,2))
#model.add(Dense(9,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
#compile the keras model
#model.compile(loss='binary_crossentropy',optimize='adam',metrics=['accuracy'])
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])

#訓練
model.fit(X_train,y_train)
#預測，評估模型好壞 使用預測資料當測試資料
pred= model.predict（X＿train)
#pred= model.predict_classes（X＿train)


#將預測結果與真實資料合併成dataframe
from pandas import DataFrame
pred_df=Dataframe(pred)
#將預測結果轉整數（0/1)
#pred= round(pred_df,0)
#print(pred)




