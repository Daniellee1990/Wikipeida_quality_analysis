#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:13:17 2018

@author: lixiaodan
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_with_all_features.csv')
#wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_without_network.csv')

labels = wikidata["page_class"]
for i in range(labels.shape[0]):
    if labels[i] == 'FA' or labels[i] == 'AC':
        labels.loc[i] = '0'
    elif labels[i] == 'GA' or labels[i] == 'BC':
        labels.loc[i] = '1'
    elif labels[i] == 'ST' or labels[i] == 'SB':
        labels.loc[i] = '2'

labels = labels.convert_objects(convert_numeric=True)

### preprocess features
features = wikidata.iloc[:, 1:-1]
min_max_scaler = preprocessing.MinMaxScaler()
features_minmax = min_max_scaler.fit_transform(features)

### split data into training set and label set
X_train, X_test, y_train, y_test = train_test_split(features_minmax, labels, test_size=0.2, random_state=42)
samples_num = y_test.shape[0]

### Decision tree #####
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)
dtree_accuracy = dtree_model.score(X_test, y_test)
print(dtree_accuracy)
# creating a confusion matrix
cm_dt = confusion_matrix(y_test, dtree_predictions)

### svm ###
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test) 
# model accuracy for X_test  
svm_accuracy = svm_model_linear.score(X_test, y_test)
print(svm_accuracy) 
# creating a confusion matrix
cm_svm = confusion_matrix(y_test, svm_predictions)

### KNN ###
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 
# accuracy on X_test
knn_accuracy = knn.score(X_test, y_test)
print( knn_accuracy ) 
# creating a confusion matrix
knn_predictions = knn.predict(X_test) 
cm_knn = confusion_matrix(y_test, knn_predictions) 

# training a Naive Bayes classifier
gnb = GaussianNB().fit(X_train, y_train)
gnb_predictions = gnb.predict(X_test)
# accuracy on X_test
gnb_accuracy = gnb.score(X_test, y_test)
print(gnb_accuracy)
# creating a confusion matrix
gnb_cm = confusion_matrix(y_test, gnb_predictions)
 
"""
predictions_one_vs_rest = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)

predictions_one_vs_one = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)

clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
prediction_outputCode = clf.fit(X_train, y_train).predict(X_test)

correct_onevsone = 0
correct_onevsrest = 0
correct_output = 0
y_test = np.array(y_test)
for i in range(samples_num):
    if predictions_one_vs_rest[i] == y_test[i]:
        correct_onevsrest = correct_onevsrest + 1
    if predictions_one_vs_one[i] == y_test[i]:
        correct_onevsone = correct_onevsone + 1
    if prediction_outputCode[i] == y_test[i]:
        correct_output = correct_output + 1
        
print("Accuracy for one vs one classifier")
acc_oneVsone = float(correct_onevsone) / samples_num
print(acc_oneVsone)

print("Accuracy for one vs rest classifier")
acc_oneVsrest = float(correct_onevsrest) / samples_num
print(acc_oneVsrest)

print("Accuracy for output code classifier")
acc_output = float(correct_output) / samples_num
print(acc_output)
"""