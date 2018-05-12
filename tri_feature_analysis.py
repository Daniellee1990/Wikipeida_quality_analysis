#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 20:39:39 2018

@author: lixiaodan
"""

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

wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_with_all_features.csv')
#wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_without_network.csv')
#wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_without_hist_net.csv')
colnames = list(wikidata)
#print(colnames)

labels = wikidata["page_class"]
for i in range(labels.shape[0]):
        if labels[i] == 'FA' or labels[i] == 'AC':
            labels.loc[i] = 0
        elif labels[i] == 'GA' or labels[i] == 'BC':
            labels.loc[i] = 1
        elif labels[i] == 'ST' or labels[i] == 'SB':
            labels.loc[i] = 2        

labels = labels.convert_objects(convert_numeric=True)
onehotlabels = np_utils.to_categorical(labels)

### preprocess features
features = wikidata.iloc[:, 1:]
feature_names = list(features)
min_max_scaler = preprocessing.MinMaxScaler()
features_minmax = min_max_scaler.fit_transform(features)
features_analysis_result = dict()
accuracies = list()

for i in range(len(feature_names)):
    cur_features = features
    cur_features = cur_features.drop(feature_names[i], axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    cur_features = min_max_scaler.fit_transform(cur_features)
    cur_feature_name = feature_names[i]
    ### split data into training set and label set
    X_train, X_test, y_train, y_test = train_test_split(cur_features, onehotlabels, test_size=0.4, random_state=42)
    transformed_y = deep_learning_models.transformResult(y_test)
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
    model, hist = deep_learning_models.stacked_LSTM(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs)
    prediction = model.predict(X_test_LSTM)
    transformed_pre = deep_learning_models.transformResult(prediction)
    prediction_re = np.argmax(prediction, axis=1)
    accuracy, precision, recall, F1, fbeta = deep_learning_models.getAccuracyMulti(transformed_pre, transformed_y)
    
    #stacked_LSTM_precision = deep_learning_models.getAccuracy(prediction, y_test)
    print("Precision for stacked LSTM")
    print(cur_feature_name)
    print(accuracy)
    accuracies.append(accuracy)
    features_analysis_result[cur_feature_name] = accuracy