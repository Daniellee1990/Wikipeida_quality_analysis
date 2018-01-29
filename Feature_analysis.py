#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 10:22:28 2018

@author: lixiaodan
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import np_utils
import deep_learning_models
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_with_all_features.csv')
#wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_without_network.csv')
#wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_without_hist_net.csv')
colnames = list(wikidata)
#print(colnames)

labels = wikidata["page_class"]
for i in range(labels.shape[0]):
    if labels[i] == 'FA' or labels[i] == 'AC' or labels[i] == 'GA':
        labels.loc[i] = '0'
    elif labels[i] == 'BC' or labels[i] == 'ST' or labels[i] == 'SB':
        labels.loc[i] = '1'
    """
    if labels[i] == 'FA' or labels[i] == 'AC':
        labels.loc[i] = '0'
    elif labels[i] == 'GA' or labels[i] == 'BC':
        labels.loc[i] = '1'
    elif labels[i] == 'ST' or labels[i] == 'SB':
        labels.loc[i] = '2'
    """

labels = labels.convert_objects(convert_numeric=True)
onehotlabels = np_utils.to_categorical(labels)

### preprocess features
features = wikidata.iloc[:, 1:]
feature_names = list(features)
min_max_scaler = preprocessing.MinMaxScaler()
features_minmax = min_max_scaler.fit_transform(features)

### split data into training set and label set
#X_train, X_test, y_train, y_test = train_test_split(features_minmax, onehotlabels, test_size=0.4, random_state=42)

### adjust the dataset dimension
# reshape X to be [samples, time steps, features]
#X_train_LSTM = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
#X_test_LSTM = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

"""
### create the deep learning models
epochs = 15
batch_size = 195 #195 best now #190 # 100 # 250
dropoutRate = 0.2

model6, hist6 = deep_learning_models.stacked_LSTM(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs)
importances = hist6.feature_importances_
"""

"""
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(features_minmax, labels)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
X = features_minmax

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
       #color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
"""
features_analysis_result = dict()

for i in range(len(feature_names)):
    cur_features = features
    cur_features = cur_features.drop(feature_names[i], axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    cur_features = min_max_scaler.fit_transform(cur_features)
    cur_feature_name = feature_names[i]
    ### split data into training set and label set
    X_train, X_test, y_train, y_test = train_test_split(cur_features, onehotlabels, test_size=0.4, random_state=42)
    #X_train, X_test, y_train, y_test = train_test_split(features_minmax, labels, test_size=0.4, random_state=42)
    
    ### adjust the dataset dimension
    # reshape X to be [samples, time steps, features]
    X_train_LSTM = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_LSTM = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    
    ### create the deep learning models
    epochs = 15
    batch_size = 195 #195 best now #190 # 100 # 250
    dropoutRate = 0.2
    
    ## stacked LSTM
    model6, hist6 = deep_learning_models.stacked_LSTM(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs)
    prediction6 = model6.predict(X_test_LSTM)
    prediction6_re = np.argmax(prediction6, axis=1)
    stacked_LSTM_precision = deep_learning_models.getAccuracy(prediction6, y_test)
    print("Precision for stacked LSTM")
    print(cur_feature_name)
    print(stacked_LSTM_precision)
    features_analysis_result[cur_feature_name] = stacked_LSTM_precision
    
    
    
     