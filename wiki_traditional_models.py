#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:13:17 2018

@author: lixiaodan
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC


#wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_with_all_features.csv')
wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_without_network.csv')

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
features = wikidata.iloc[:, 0:-1]
min_max_scaler = preprocessing.MinMaxScaler()
features_minmax = min_max_scaler.fit_transform(features)

### split data into training set and label set
X_train, X_test, y_train, y_test = train_test_split(features_minmax, labels, test_size=0.2, random_state=42)
samples_num = y_test.shape[0]

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