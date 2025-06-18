#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:39:11 2023

@author: chenguanchen
"""
#機器學習演算法(Kmeans)範例應用於iris資料集
from sklearn import cluster, datasets
#讀入鳶尾花資料
iris = datasets.load_iris()
iris_X = iris.data
print(iris_X)
#KMeans演算法及預測
Cluster_fit = cluster.KMeans(n_clusters = 3).fit(iris_X) 
#Hierarchical Clustering 演算法及 預測
#cluster_fit = cluster.AgglomerativeClustering(linkage = 'ward', affinity = 'euclidean', n _clusters =
#分群結果
cluster_labels = Cluster_fit.labels_
iris_y = iris.target
#將預測結果與真實資料併
from pandas import DataFrame
myPredResult=DataFrame({"Pred": cluster_labels, "Real":iris_y})
print("DataFrame: Pred, Real")
myPredResult["Ans"]=0
TP=0
for i in range(len(myPredResult)):
    if myPredResult.iat[i,0]==myPredResult.iat[i,1]:
        myPredResult.iat[i,2]=1
        TP=TP+1
    else:
        myPredResult.iat[i,2]=0
print(myPredResult)
print("資料Accuracy_Cal", round(TP/len(myPredResult),2))
#sepal length. sepal width, petal length. petal width
#from sklearn.cluster import KMeans
#km= KMeans(n_clusters-3) #K=2群
#y_pred =km.fit_predict(iris_X)
#直接引用 上半部分群預測結果
y_pred =Cluster_fit.labels_
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
#sepal length, sepal width
plt.scatter(iris_X[:,0], iris_X[:,1],c=y_pred)#C是第三维度 已色做維度
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('Cluster k-means')
plt.show()
#petal length, petal width
plt.scatter(iris_X[:, 2], iris_X[:, 3], c=y_pred)#c是第三维度 已顏色做維度
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('Cluster k-means')
plt.show()
#km.cluster_centers_#名群中心點(8,Y)的位置
######Real
plt.figure(figsize=(10,6))
#sepal length, sepal width 
plt.scatter(iris_X[:,0], iris_X[:,1],c=iris_y)#c是第三维度已色做维度
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('Cluster Real')
plt.show()
#petal length, petal width 
plt.scatter(iris_X[:,2], iris_X[:,3], c=iris_y)#c是第三维度已顏色做维度
plt.xlabel('petal Length')
plt.ylabel('petal width')
plt.title('Cluster Real')
plt.show()
#km.cluster_centers_#各群中心(X,Y)的位置
